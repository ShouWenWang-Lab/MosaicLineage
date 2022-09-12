import gzip
import os

import numpy as np
import pandas as pd
import scipy.sparse as ssp
import seaborn as sns
import yaml
from Bio import SeqIO
from Bio.Seq import Seq
from matplotlib import pyplot as plt
from scipy.io import loadmat

import carlinhf.analysis_script as analysis
import carlinhf.util as util

#########################################

## We put CARLIN-specific operations here

#########################################


# define seqeunces and primers for QC. For each 5' sequence, we skip the first 2 bp, and for 3' sequence, we skip the last 2 bp, as they are prone to errors
CC_5prime = "AGCTGTACAAGTAAGCGGC"
CC_3prime = "AGAATTCTAACTAGAGCTCGCTGATCAGCCTCGACTGTGCCTTCT"
CC_CARLIN = "CGCCGGACTGCACGACAGTCGACGATGGAGTCGACACGACTCGCGCATACGATGGAGTCGACTACAGTCGCTACGACGATGGAGTCGCGAGCGCTATGAGCGACTATGGAGTCGATACGATACGCGCACGCTATGGAGTCGAGAGCGCGCTCGTCGACTATGGAGTCGCGACTGTACGCACACGCGATGGAGTCGATAGTATGCGTACACGCGATGGAGTCGAGTCGAGACGCTGACGATATGGAGTCGATACGTAGCACGCAGACGATGGGAGCT"

TC_5prime = "TCGGTACCTCGCGAA"
TC_3prime = "GTCTTGTCGGTGCCTTCTAGTT"
TC_CARLIN = "TCGCCGGAGTCGAGACGCTGACGATATGGAGTCGACACGACTCGCGCATACGATGGAGTCGCGAGCGCTATGAGCGACTATGGAGTCGATAGTATGCGTACACGCGATGGAGTCGACTACAGTCGCTACGACGATGGAGTCGATACGATACGCGCACGCTATGGAGTCGCGACTGTACGCACACGCGATGGAGTCGACTGCACGACAGTCGACGATGGAGTCGATACGTAGCACGCAGACGATGGGAGCGAGAGCGCGCTCGTCGACTATGGA"

RC_5prime = "GTACAAGTAAAGCGGCC"
RC_3prime = "GTCTGCTGTGTGCCTTCTAGTT"
RC_CARLIN = "GCGCCGGCGAGCGCTATGAGCGACTATGGAGTCGACACGACTCGCGCATACGATGGAGTCGACTACAGTCGCTACGACGATGGAGTCGATACGATACGCGCACGCTATGGAGTCGACTGCACGACAGTCGACGATGGAGTCGATACGTAGCACGCAGACGATGGGAGCGAGTCGAGACGCTGACGATATGGAGTCGATAGTATGCGTACACGCGATGGAGTCGCGACTGTACGCACACGCGATGGAGTCGAGAGCGCGCTCGTCGACTATGGA"


#########################################

## Call CARLIN alleles from raw data

#########################################


def consensus_sequence(df):
    X = np.array([np.array(bytearray(x, encoding="utf8")) for x in df])
    return bytes(np.median(X, axis=0).astype("uint8")).decode("utf8")


def obtain_read_dominant_sequences(
    df_input, cell_bc_key="cell_bc", clone_key="clone_id"
):
    """
    Find the candidate sequence with the max read count within each group
    """

    df_input["CARLIN_length"] = df_input[clone_key].apply(lambda x: len(x))
    df_CARLIN_lenth = (
        df_input.groupby([cell_bc_key, clone_key, "CARLIN_length"])
        .agg(read=("read", "sum"))
        .reset_index()
    )

    df_dominant_fraction = (
        df_CARLIN_lenth.groupby([cell_bc_key])
        .agg(
            read=("read", "max"),
            max_read_ratio=("read", lambda x: np.max(x) / np.sum(x)),
        )
        .reset_index()
    )

    df_out = df_CARLIN_lenth.merge(
        df_dominant_fraction, on=[cell_bc_key, "read"], how="inner"
    )
    return df_input.drop(["read", "CARLIN_length"], axis=1).merge(
        df_out, on=[cell_bc_key, clone_key]
    )


def CARLIN_analysis(
    df_input, cell_bc_key="cell_bc", clone_key="clone_id", read_ratio_threshold=0.6
):
    """
    This function is similar to the CARLIN pipeline, that for each tag, we find the dominant CARLIN sequence as the right candidate.
    At the moment, I believe that second part (obtain consensus sequence is not used, as there is only one sequence left after the
    dominant sequence selection.
    """

    # df_dominant_fraction=calculate_read_fraction_for_dominant_sequences(df_input,cell_bc_key=cell_bc_key,clone_key=clone_key)
    df_tmp = obtain_read_dominant_sequences(
        df_input, cell_bc_key=cell_bc_key, clone_key=clone_key
    )
    df_final = df_tmp[df_tmp["max_read_ratio"] > read_ratio_threshold]

    # obtain consensus sequences
    df_final = df_final.groupby(cell_bc_key).agg(
        consensuse_CARLIN=(clone_key, consensus_sequence), read=("read", "sum")
    )
    df_final["CARLIN_length"] = df_final["consensuse_CARLIN"].apply(lambda x: len(x))
    return df_final


def CARLIN_raw_reads(data_path, sample, protocol="scLimeCat"):
    """
    Load raw fastq files. This function will depend on what protocol is used.
    """
    # supported_protocol = ["scLimeCat", "sc10xV3"]
    # if not (protocol in supported_protocol):
    #     raise ValueError(f"Only support protocols: {supported_protocol}")

    if protocol.startswith("sc"):
        seq_list = []
        seq_quality = []
        with gzip.open(f"{data_path}/{sample}_L001_R2_001.fastq.gz", "rt") as handle:
            for record in SeqIO.parse(handle, "fastq"):
                seq_list.append(str(record.seq))
                quality_tmp = record.letter_annotations["phred_quality"]
                seq_quality.append(quality_tmp)

        tag_list = []
        tag_quality = []
        with gzip.open(f"{data_path}/{sample}_L001_R1_001.fastq.gz", "rt") as handle:
            for record in SeqIO.parse(handle, "fastq"):
                tag_list.append(str(record.seq))
                quality_tmp = record.letter_annotations["phred_quality"]
                tag_quality.append(quality_tmp)

        if protocol == "scLimeCat":
            bc_len = 8
            umi_len = 8
        elif protocol == "sc10xV3":
            bc_len = 16
            umi_len = 12
        else:
            raise ValueError(f"{protocol} must be among scLimeCat, sc10xV3")

        df_seq = pd.DataFrame(
            {
                "Tag": tag_list,
                "Seq": seq_list,
                "Seq_quality": seq_quality,
                "Tag_quality": tag_quality,
            }
        )
        df_seq["cell_bc"] = df_seq["Tag"].apply(lambda x: x[:bc_len])
        df_seq["cell_bc_quality_mean"] = df_seq["Tag_quality"].apply(
            lambda x: np.mean(x[:bc_len])
        )
        df_seq["cell_bc_quality_min"] = df_seq["Tag_quality"].apply(
            lambda x: np.min(x[:bc_len])
        )
        df_seq["library"] = sample
        df_seq["cell_id"] = df_seq["library"] + "_" + df_seq["cell_bc"]
        df_seq["umi"] = df_seq["Tag"].apply(lambda x: x[bc_len : (bc_len + umi_len)])
        df_seq["umi_quality_mean"] = df_seq["Tag_quality"].apply(
            lambda x: np.mean(x[bc_len : (bc_len + umi_len)])
        )
        df_seq["umi_quality_min"] = df_seq["Tag_quality"].apply(
            lambda x: np.min(x[bc_len : (bc_len + umi_len)])
        )
        df_seq["umi_id"] = df_seq["cell_bc"] + "_" + df_seq["umi"]
        df_seq["clone_id"] = df_seq["Seq"]
        df_seq["clone_id_quality_mean"] = df_seq["Seq_quality"].apply(
            lambda x: np.mean(x)
        )
        df_seq["clone_id_quality_min"] = df_seq["Seq_quality"].apply(
            lambda x: np.min(x)
        )
        df_seq = df_seq.drop(["Tag", "Seq", "Seq_quality", "Tag_quality"], axis=1)

    elif protocol.startswith("Bulk"):
        if "UMI" in protocol:
            UMI_length = int(protocol.split("UMI")[0][-2:])
        else:
            UMI_length = 12

        seq_list = []
        quality = []
        handle = f"{data_path}/{sample}.trimmed.pear.assembled.fastq"
        for record in SeqIO.parse(handle, "fastq"):
            seq_list.append(str(record.seq))
            quality_tmp = record.letter_annotations["phred_quality"]
            quality.append(quality_tmp)

        df_seq = pd.DataFrame({"quality": quality, "Seq": seq_list})
        df_seq["cell_bc"] = df_seq["Seq"].apply(lambda x: x[:UMI_length])
        df_seq["cell_bc_quality_min"] = df_seq["quality"].apply(
            lambda x: np.min(x[:UMI_length])
        )
        df_seq["cell_bc_quality_mean"] = df_seq["quality"].apply(
            lambda x: np.mean(x[:UMI_length])
        )
        df_seq["library"] = sample
        df_seq["cell_id"] = df_seq["library"] + "_" + df_seq["cell_bc"]
        df_seq["umi"] = ""
        df_seq["umi_id"] = df_seq["cell_bc"] + "_" + df_seq["umi"]
        df_seq["clone_id"] = df_seq["Seq"].apply(
            lambda x: util.reverse_compliment(x[UMI_length:])
        )
        df_seq["clone_id_quality_min"] = df_seq["quality"].apply(
            lambda x: np.min(x[UMI_length:])
        )
        df_seq["clone_id_quality_mean"] = df_seq["quality"].apply(
            lambda x: np.mean(x[UMI_length:])
        )
        df_seq = df_seq.drop(["quality", "Seq"], axis=1)
    else:
        raise ValueError("un supported cfg")

    return df_seq


def CARLIN_preprocessing(
    df_input,
    template="cCARLIN",
    ref_cell_barcodes=None,
    seq_5prime_upper_N=None,
    seq_3prime_upper_N=None,
):
    """
    Filter the raw reads. This pipeline should be independent of whether this is bulk or single-cell CARLIN

    Parameters
    ----------
    df_input: pd.DataFrame
        input data, from CARLIN_raw_reads
    template: str
        {'cCARLIN','Tigre','Rosa'}
    ref_cell_barcodes:
        Reference cell barcode list, for filtering
    seq_5prime_upper_N:
        Control the number of 5prime bps for QC. Default: use all bps.
    seq_3prime_upper_N:
        Control the number of 5prime bps for QC. Default: use all bps.

    Returns
    -------
    df_output:
        A dataframe of sequences that pass QC
    """

    # seq_5prime:
    #     5' end sequences, for QC. Only reads contain exactly this sequence will pass QC.
    #     The end of the 5' end sequences mark the beginning of CARLIN sequences.
    # seq_3prime:
    # 3' end sequences, for QC. Only reads contain exactly this sequence will pass QC.
    #     The beginning of the 3' end sequences mark the end of CARLIN sequences.

    if template.startswith("cCARLIN"):
        seq_5prime = CC_5prime
        seq_3prime = CC_3prime
        CARLIN_seq_CC = CC_CARLIN
    elif template.startswith("Tigre"):
        seq_5prime = TC_5prime
        seq_3prime = TC_3prime
        CARLIN_seq_TC = TC_CARLIN
    elif template.startswith("Rosa"):
        seq_5prime = RC_5prime
        seq_3prime = RC_3prime
        CARLIN_seq_RC = RC_CARLIN
    else:
        raise ValueError("template must start with {'cCARLIN','Tigre','Rosa'}")

    if seq_5prime_upper_N is not None:
        seq_5prime = seq_5prime[-seq_5prime_upper_N:]
    if seq_3prime_upper_N is not None:
        seq_3prime = seq_3prime[:seq_3prime_upper_N]

    df_output = df_input.copy()
    df_output["Valid"] = df_output["clone_id"].apply(
        lambda x: (seq_5prime in x) & (seq_3prime in x)
    )
    tot_fastq_N = len(df_output)
    print("Total fastq:", tot_fastq_N)
    df_output = df_output.query("Valid==True")
    valid_3_5_prime_N = len(df_output)
    print(
        f"Fastq with vaid 3 and 5 prime: {valid_3_5_prime_N} ({valid_3_5_prime_N/tot_fastq_N:.2f})"
    )
    if ref_cell_barcodes is not None:
        df_output = df_output[df_output["cell_bc"].isin(ref_cell_barcodes)]
        valid_BC_N = len(df_output)
        print(f"Fastq with valid barcodes: {valid_BC_N} ({valid_BC_N/tot_fastq_N:.2f})")

    df_output["clone_id"] = df_output["clone_id"].apply(
        lambda x: x.split(seq_5prime)[1].split(seq_3prime)[0]
    )
    df_output["unique_id"] = (
        df_output["cell_id"] + "_" + df_output["umi_id"] + "_" + df_output["clone_id"]
    )
    df_tmp = (
        df_output.groupby("unique_id").agg(read=("unique_id", "count")).reset_index()
    )
    return (
        df_output.merge(df_tmp, on="unique_id")
        .drop(["Valid", "unique_id"], axis=1)
        .drop_duplicates()
    )


#########################################

## extract information from CARLIN output

#########################################


def get_SampleList(root_path):
    with open(f"{root_path}/config.yaml", "r") as stream:
        file = yaml.safe_load(stream)
        SampleList = file["SampleList"]
    return SampleList


def load_allele_info(data_path):
    """
    Convert allele and frequency information to a pd.DataFrame.

    data_path: point to specific sample folder, e.g. path/to/results_read_cutoff_3/{sample}
    """
    pooled_data = loadmat(os.path.join(data_path, "allele_annotation.mat"))
    allele_freqs = pooled_data["allele_freqs"].flatten()
    alleles = [xx[0][0] for xx in pooled_data["AlleleAnnotation"]]
    return pd.DataFrame({"allele": alleles, "UMI_count": allele_freqs})


def load_allele_frequency_statistics(data_path: str, SampleList: list):
    """
    Return an allele-grouped frequency count table

    data_path: should be at the level of samples, e.g., path/to/results_read_cutoff_3
    """

    df_list = []
    for sample in SampleList:
        df_temp = load_allele_info(os.path.join(data_path, sample))
        df_list.append(df_temp)
        print(f"{sample}: {len(df_temp)}")
    df_raw = pd.concat(df_list).reset_index()
    df_raw["sample_count"] = 1
    df_new = df_raw.groupby("allele", as_index=False).agg(
        {"UMI_count": "sum", "sample_count": "sum"}
    )
    return df_new


def extract_CARLIN_info(
    data_path,
    SampleList,
):
    """
    Extract CARLIN information, like alleles, colonies, UMI count info

    data_path:
        The root dir to all the samples, e.g. path/to/results_read_cutoff_3
    SampleList:
        The list of desired samples to load
    """

    tmp_list = []
    for sample in SampleList:
        base_dir = os.path.join(data_path, sample)
        df_tmp = load_allele_info(base_dir)
        df_tmp["sample"] = sample.split("_")[0]
        df_tmp["mouse"] = sample.split("-")[0]

        df_allele = pd.read_csv(
            data_path + f"/{sample}/AlleleAnnotations.txt",
            sep="\t",
            header=None,
            names=["allele"],
        )
        df_CB = pd.read_csv(
            data_path + f"/{sample}/AlleleColonies.txt",
            sep="\t",
            header=None,
            names=["CB"],
        )
        df_allele["CB"] = df_CB
        df_allele["CB_N"] = df_allele["CB"].apply(lambda x: len(x.split(",")))

        if os.path.exists(data_path + f"/{sample}/Actaul_CARLIN_seq.txt"):
            df_CARLIN = pd.read_csv(
                data_path + f"/{sample}/Actaul_CARLIN_seq.txt",
                sep="\t",
                header=None,
                names=["CARLIN"],
            )

            df_allele["CARLIN"] = df_CARLIN["CARLIN"].apply(
                lambda x: "".join(x.split("-"))
            )
            df_allele["CARLIN_length"] = df_allele["CARLIN"].apply(lambda x: len(x))

        df_tmp = df_tmp.merge(df_allele, on="allele")
        tmp_list.append(df_tmp)
    df_all = pd.concat(tmp_list)

    # add clone_size information for each allele
    df_tmp = df_all.copy()
    df_tmp["CB_tmp"] = df_tmp["CB"].str.split(",")
    df_tmp = df_tmp.explode("CB_tmp")
    df_tmp["cell_id"] = df_tmp["sample"] + "_" + df_tmp["CB_tmp"]
    df_clone_size = (
        df_tmp.groupby("allele")
        .agg(clone_size=("cell_id", lambda x: len(set(x))))
        .reset_index()
    )
    df_all = df_all.merge(df_clone_size, on="allele")
    return df_all


def CARLIN_output_to_cell_by_barcode_long_table(df_input):
    """
    Convert output from extract_CARLIN_info (a wide table)
    to a two column (cell, and clone_id) long table.

    Warn: this is suitable only within a library.

    A alternative method:
    ```python
    df_input['CB']=df_input['CB'].str.split(',')
    df_output=df_input.explode('CB').reset_index(drop=True).rename(columns={'CB':'cell_bc','CARLIN':'clone_id'})
    ```
    """

    CB_flat = []
    Clone_id_flat = []
    for j in range(len(df_input)):
        df_series = df_input.iloc[j]
        tmp = [x for x in df_series["CB"].split(",")]
        CB_flat += tmp
        Clone_id_flat += [df_series["CARLIN"] for _ in tmp]

    df_ref_flat = pd.DataFrame({"cell_bc": CB_flat, "clone_id": Clone_id_flat})

    return df_ref_flat


def merge_three_locus(
    data_path_CC,
    data_path_RC,
    data_path_TC=None,
    sample_type_CC="CC",
    sample_type_TC="RC",
    sample_type_RC="TC",
):
    """
    Merge the 3 locus (CC,TC,RC) according to the sample info and
    convert to a wide table

    data_path_CC:
        Path to CC locus root dir that contains all sample sub-folder,
        e.g. path/to/results_read_cutoff_3
    sample_type_CC
    """

    df_CC = pd.read_csv(
        f"{data_path_CC}/merge_all/refined_results.csv", index_col=0
    ).sort_values("sample")
    df_CC = df_CC[df_CC["sample"] != "merge_all"]
    idx = np.argsort(df_CC["sample"])
    df_CC = df_CC.iloc[idx]
    df_CC["sample_id"] = np.arange(len(df_CC))
    df_CC["Type"] = sample_type_CC
    df_RC = pd.read_csv(
        f"{data_path_RC}/merge_all/refined_results.csv", index_col=0
    ).sort_values("sample")
    df_RC = df_RC[df_RC["sample"] != "merge_all"]
    idx = np.argsort(df_RC["sample"])
    df_RC = df_RC.iloc[idx]
    df_RC["sample_id"] = np.arange(len(df_RC))
    df_RC["Type"] = sample_type_TC

    x = "total_alleles"
    df_CC[f"{x}_norm_fraction"] = df_CC[x] / df_CC[x].sum()
    df_RC[f"{x}_norm_fraction"] = df_RC[x] / df_RC[x].sum()
    x = "singleton"
    df_CC[f"{x}_norm_fraction"] = df_CC[x] / df_CC[x].sum()
    df_RC[f"{x}_norm_fraction"] = df_RC[x] / df_RC[x].sum()

    if data_path_TC is not None:
        df_TC = pd.read_csv(
            f"{data_path_TC}/merge_all/refined_results.csv", index_col=0
        ).sort_values("sample")
        df_TC = df_TC[df_TC["sample"] != "merge_all"]
        idx = np.argsort(df_TC["sample"])
        df_TC = df_TC.iloc[idx]
        df_TC["sample_id"] = np.arange(len(df_TC))
        df_TC["Type"] = sample_type_RC

        x = "total_alleles"
        df_TC[f"{x}_norm_fraction"] = df_TC[x] / df_TC[x].sum()
        x = "singleton"
        df_TC[f"{x}_norm_fraction"] = df_TC[x] / df_TC[x].sum()

        df_all = pd.concat([df_CC, df_RC, df_TC])
        df_sample_association = (
            df_CC.filter(["sample_id", "sample"])
            .merge(df_TC.filter(["sample_id", "sample"]), on="sample_id")
            .merge(df_RC.filter(["sample_id", "sample"]), on="sample_id")
        )
    else:
        df_all = pd.concat([df_CC, df_RC])
        df_sample_association = df_CC.filter(["sample_id", "sample"]).merge(
            df_RC.filter(["sample_id", "sample"]), on="sample_id"
        )

    df_sample_association = df_sample_association.rename(
        columns={"sample_x": "CC", "sample_y": "TC", "sample": "RC"}
    )
    return df_all, df_sample_association


def extract_lineage(x):
    """
    We expect the structure like 'LL731-LF-B'
    """
    x = "-".join(x.split("-")[:3])  # keep at most 3 entries
    if ("MPP3" in x) and ("MPP3-4" not in x):
        return "MPP3-4".join(x.split("MPP3"))
    else:
        return x


def rename_lib(x):
    # this is for CARLIN data
    if "_" in x:
        x = x.split("_")[0]

    if ("CC" in x) or ("TC" in x) or ("RC" in x):
        return x[:-3]
    else:
        return x


def extract_plate_ID(x):
    # this is for single-cell Limecat protocl
    return x[:-3]


def add_metadata(df_sc_data, plate_map=None):
    """
    Annotate single-cell CARLIN data
    """

    if "library" in df_sc_data.columns:
        # CARLIN like data, based on library
        df_sc_data["library"] = df_sc_data["library"].apply(rename_lib)
        df_sc_data["sample"] = df_sc_data["library"]
        df_sc_data["plate_ID"] = df_sc_data["sample"]
    elif "sample" in df_sc_data.columns:
        # plate-based single-cell data
        df_sc_data["plate_ID"] = df_sc_data["sample"].apply(extract_plate_ID)
    else:
        raise ValueError("library or sample not found")

    df_sc_data["mouse"] = df_sc_data["sample"].apply(lambda x: x.split("-")[0])

    if plate_map is not None:
        df_sc_data["plate_ID"] = df_sc_data["plate_ID"].map(plate_map)

    df_sc_data["RNA_id"] = df_sc_data["plate_ID"] + "_RNA_" + df_sc_data["cell_bc"]
    df_sc_data["clone_id"] = df_sc_data["locus"] + "_" + df_sc_data["clone_id"]
    df_sc_data["allele"] = df_sc_data["locus"] + "_" + df_sc_data["allele"]
    return df_sc_data


def generate_sc_CARLIN_from_CARLIN_output(df_all):
    if "locus" not in df_all.columns:
        df_all["locus"] = "locus"

    if "CARLIN" not in df_all.columns:
        df_all["CARLIN"] = df_all["allele"]

    df_merge = df_all.fillna(0)
    df_merge["library"] = df_merge["sample"]
    df_merge["clone_size"] = df_merge["CB"].apply(lambda x: len(x.split(",")))
    df_merge["CB"] = df_merge["CB"].str.split(",")
    df_merge = (
        df_merge.explode("CB")
        .reset_index(drop=True)
        .rename(columns={"CB": "cell_bc", "CARLIN": "clone_id"})
    )
    df_sc_CARLIN = add_metadata(df_merge)

    df_sc_CARLIN["lineage"] = df_merge["library"].apply(extract_lineage)

    return df_sc_CARLIN
