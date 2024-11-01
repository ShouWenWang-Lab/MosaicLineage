import gzip
import os

import numpy as np
import pandas as pd
import scipy.sparse as ssp
import yaml
from Bio import SeqIO
from scipy.io import loadmat

import mosaiclineage.larry as larry
import mosaiclineage.util as util

#########################################

## We put CARLIN-specific operations here

#########################################


# define seqeunces and primers for QC. For each 5' sequence, we skip the first 2 bp, and for 3' sequence, we skip the last 2 bp, as they are prone to errors
CA_5prime = (
    "AGCTGTACAAGTAAGCGGC"  # full primer: GAGCTGTACAAGTAAGCGGC (single-cell & bulk)
)
CA_3prime = "AGAATTCTAACTAGAGCTCGCTGATCAGCCTCGACTGTGCCTTCT"  # full primer: AGAATTCTAACTAGAGCTCGCTGATCAGCCTCGACTGTGCCTTCTAGTTGC (only for bulk DARLIN protocol)
CA_CARLIN = "CGCCGGACTGCACGACAGTCGACGATGGAGTCGACACGACTCGCGCATACGATGGAGTCGACTACAGTCGCTACGACGATGGAGTCGCGAGCGCTATGAGCGACTATGGAGTCGATACGATACGCGCACGCTATGGAGTCGAGAGCGCGCTCGTCGACTATGGAGTCGCGACTGTACGCACACGCGATGGAGTCGATAGTATGCGTACACGCGATGGAGTCGAGTCGAGACGCTGACGATATGGAGTCGATACGTAGCACGCAGACGATGGGAGCT"

TA_5prime = "TCGGTACCTCGCGAA"  # full primer: GCTCGGTACCTCGCGAAT (single-cell & bulk)
TA_3prime = "GTCTTGTCGGTGCCTTCTAGTT"  # full primer: GTCTTGTCGGTGCCTTCTAGTTGC (only for bulk DARLIN protocol)
TA_CARLIN = "TCGCCGGAGTCGAGACGCTGACGATATGGAGTCGACACGACTCGCGCATACGATGGAGTCGCGAGCGCTATGAGCGACTATGGAGTCGATAGTATGCGTACACGCGATGGAGTCGACTACAGTCGCTACGACGATGGAGTCGATACGATACGCGCACGCTATGGAGTCGCGACTGTACGCACACGCGATGGAGTCGACTGCACGACAGTCGACGATGGAGTCGATACGTAGCACGCAGACGATGGGAGCGAGAGCGCGCTCGTCGACTATGGA"

RA_5prime = (
    "GTACAAGTAAAGCGGCC"  # full primer: ATGTACAAGTAAAGCGGCCG (single-cell & bulk)
)
RA_3prime = "GTCTGCTGTGTGCCTTCTAGTT"  # full primer: GTCTGCTGTGTGCCTTCTAGTTGC (only for bulk DARLIN protocol)
RA_CARLIN = "GCGCCGGCGAGCGCTATGAGCGACTATGGAGTCGACACGACTCGCGCATACGATGGAGTCGACTACAGTCGCTACGACGATGGAGTCGATACGATACGCGCACGCTATGGAGTCGACTGCACGACAGTCGACGATGGAGTCGATACGTAGCACGCAGACGATGGGAGCGAGTCGAGACGCTGACGATATGGAGTCGATAGTATGCGTACACGCGATGGAGTCGCGACTGTACGCACACGCGATGGAGTCGAGAGCGCGCTCGTCGACTATGGA"


#########################################

## Call CARLIN alleles from raw data

#########################################


def consensus_sequence(df):
    X = np.array([np.array(bytearray(x, encoding="utf8")) for x in df])
    return bytes(np.median(X, axis=0).astype("uint8")).decode("utf8")


def CARLIN_analysis(
    df_input, cell_bc_key="cell_bc", clone_key="clone_id", read_ratio_threshold=0.6
):
    """
    This function is similar to the CARLIN pipeline, that for each tag, we find the dominant CARLIN sequence as the right candidate.
    At the moment, I believe that second part (obtain consensus sequence is not used, as there is only one sequence left after the
    dominant sequence selection.
    """

    # df_dominant_fraction=calculate_read_fraction_for_dominant_sequences(df_input,cell_bc_key=cell_bc_key,clone_key=clone_key)
    df_tmp = larry.obtain_read_dominant_sequences(
        df_input, cell_bc_key=cell_bc_key, clone_key=clone_key
    )
    df_final = df_tmp[df_tmp["max_read_ratio"] > read_ratio_threshold]

    # obtain consensus sequences
    df_final = df_final.groupby(cell_bc_key).agg(
        consensuse_CARLIN=(clone_key, consensus_sequence), read=("read", "sum")
    )
    df_final["CARLIN_length"] = df_final["consensuse_CARLIN"].apply(lambda x: len(x))
    return df_final


def CARLIN_raw_reads(data_path, sample, protocol="scCamellia"):
    """
    Load raw fastq files. This function will depend on what protocol is used.
    """
    # supported_protocol = ["scCamellia", "sc10xV3"]
    # if not (protocol in supported_protocol):
    #     raise ValueError(f"Only support protocols: {supported_protocol}")

    if protocol.startswith("sc"):
        if protocol == "scCamellia":
            bc_len = 8
            umi_len = 8
            tag_read = "R2"
            seq_read = "R1"
        elif protocol == "sc10xV3":
            bc_len = 16
            umi_len = 12
            tag_read = "R1"
            seq_read = "R2"
        else:
            raise ValueError(f"{protocol} must be among scCamellia, sc10xV3")

        seq_list = []
        seq_quality = []
        with gzip.open(
            f"{data_path}/{sample}_L001_{seq_read}_001.fastq.gz", "rt"
        ) as handle:
            for record in SeqIO.parse(handle, "fastq"):
                seq_list.append(str(record.seq))
                quality_tmp = record.letter_annotations["phred_quality"]
                seq_quality.append(quality_tmp)

        tag_list = []
        tag_quality = []
        with gzip.open(
            f"{data_path}/{sample}_L001_{tag_read}_001.fastq.gz", "rt"
        ) as handle:
            for record in SeqIO.parse(handle, "fastq"):
                tag_list.append(str(record.seq))
                quality_tmp = record.letter_annotations["phred_quality"]
                tag_quality.append(quality_tmp)

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
        seq_5prime = CA_5prime
        seq_3prime = CA_3prime
        CARLIN_seq_CA = CA_CARLIN
    elif template.startswith("Tigre"):
        seq_5prime = TA_5prime
        seq_3prime = TA_3prime
        CARLIN_seq_TA = TA_CARLIN
    elif template.startswith("Rosa"):
        seq_5prime = RA_5prime
        seq_3prime = RA_3prime
        CARLIN_seq_RA = RA_CARLIN
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
        df_output.filter(
            [
                "cell_bc",
                "library",
                "cell_id",
                "umi",
                "umi_id",
                "clone_id",
                "unique_id",
                "Valid",
            ]
        )
        .merge(df_tmp, on="unique_id")
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
    sample_name_format="LL",
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
        df_tmp["sample"] = rename_lib(sample, sample_name_format=sample_name_format)
        df_tmp["mouse"] = rename_lib(sample, sample_name_format=sample_name_format)

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
    data_path_CA,
    data_path_RA,
    data_path_TA=None,
    sample_type_CA="CA",
    sample_type_TA="RA",
    sample_type_RA="TA",
):
    """
    Merge the 3 locus (CC,TC,RC) according to the sample info and
    convert to a wide table

    data_path_CA:
        Path to CC locus root dir that contains all sample sub-folder,
        e.g. path/to/results_read_cutoff_3
    sample_type_CA
    """

    df_CA = pd.read_csv(
        f"{data_path_CA}/merge_all/refined_results.csv", index_col=0
    ).sort_values("sample")
    df_CA = df_CA[df_CA["sample"] != "merge_all"]
    idx = np.argsort(df_CA["sample"])
    df_CA = df_CA.iloc[idx]
    df_CA["sample_id"] = np.arange(len(df_CA))
    df_CA["Type"] = sample_type_CA
    df_RA = pd.read_csv(
        f"{data_path_RA}/merge_all/refined_results.csv", index_col=0
    ).sort_values("sample")
    df_RA = df_RA[df_RA["sample"] != "merge_all"]
    idx = np.argsort(df_RA["sample"])
    df_RA = df_RA.iloc[idx]
    df_RA["sample_id"] = np.arange(len(df_RA))
    df_RA["Type"] = sample_type_TA

    x = "total_alleles"
    df_CA[f"{x}_norm_fraction"] = df_CA[x] / df_CA[x].sum()
    df_RA[f"{x}_norm_fraction"] = df_RA[x] / df_RA[x].sum()
    x = "singleton"
    df_CA[f"{x}_norm_fraction"] = df_CA[x] / df_CA[x].sum()
    df_RA[f"{x}_norm_fraction"] = df_RA[x] / df_RA[x].sum()

    if data_path_TA is not None:
        df_TA = pd.read_csv(
            f"{data_path_TA}/merge_all/refined_results.csv", index_col=0
        ).sort_values("sample")
        df_TA = df_TA[df_TA["sample"] != "merge_all"]
        idx = np.argsort(df_TA["sample"])
        df_TA = df_TA.iloc[idx]
        df_TA["sample_id"] = np.arange(len(df_TA))
        df_TA["Type"] = sample_type_RA

        x = "total_alleles"
        df_TA[f"{x}_norm_fraction"] = df_TA[x] / df_TA[x].sum()
        x = "singleton"
        df_TA[f"{x}_norm_fraction"] = df_TA[x] / df_TA[x].sum()

        df_all = pd.concat([df_CA, df_RA, df_TA])
        df_sample_association = (
            df_CA.filter(["sample_id", "sample"])
            .merge(df_TA.filter(["sample_id", "sample"]), on="sample_id")
            .merge(df_RA.filter(["sample_id", "sample"]), on="sample_id")
        )
    else:
        df_all = pd.concat([df_CA, df_RA])
        df_sample_association = df_CA.filter(["sample_id", "sample"]).merge(
            df_RA.filter(["sample_id", "sample"]), on="sample_id"
        )

    df_sample_association = df_sample_association.rename(
        columns={"sample_x": "CA", "sample_y": "TA", "sample": "RA"}
    )
    return df_all, df_sample_association


def extract_lineage(x, sample_name_format="LL"):
    """
    We expect the structure like 'LL731-LF-B'
    """
    if sample_name_format == "LL":
        x = "-".join(x.split("-")[:3])  # keep at most 3 entries
        if ("MPP3" in x) and ("MPP3-4" not in x):
            return "MPP3-4".join(x.split("MPP3"))
        else:
            return x
    else:
        return x


def rename_lib(x, sample_name_format="LL"):
    if sample_name_format == "LL":
        # this is for CARLIN data
        if "_S" in x:
            x = "_".join(x.split("_")[:-1])  # remve _SX at the end of the lib name

        if (
            ("-CA" in x)
            or ("-TA" in x)
            or ("-RA" in x)
            or ("_CA" in x)
            or ("_TA" in x)
            or ("_RA" in x)
        ):
            return x[:-3]
        elif (
            ("CA-" in x)
            or ("TA-" in x)
            or ("RA-" in x)
            or ("CA_" in x)
            or ("TA_" in x)
            or ("RA_" in x)
        ):
            return x[2:]
        else:
            return x
    else:
        return x


def extract_first_sample_from_a_nesting_list(SampleList, sample_name_format="LL"):
    """
    For a nesting list like ['a',['b','c'],['d','e','f']],
    it will return the first in each element, i.e, ['a','b','d']
    """

    def custom_rename_lib(x):
        return extract_lineage(
            rename_lib(x, sample_name_format=sample_name_format),
            sample_name_format=sample_name_format,
        )

    selected_fates = []
    for x in SampleList:
        if type(x) is list:
            x = x[0]
        selected_fates.append(custom_rename_lib(x))
    return selected_fates


def extract_plate_ID(x, sample_name_format="LL"):
    # this is for single-cell Limecat protocl
    if sample_name_format == "LL":
        return x[:-3]
    else:
        return x


def add_metadata(df_sc_data, plate_map=None, sample_name_format="LL"):
    """
    Annotate single-cell CARLIN data
    """

    def custom_rename_lib(x):
        return rename_lib(x, sample_name_format=sample_name_format)

    def custom_extract_plate_ID(x):
        return extract_plate_ID(x, sample_name_format=sample_name_format)

    if "library" in df_sc_data.columns:
        # CARLIN like data, based on library
        df_sc_data["library"] = df_sc_data["library"].apply(custom_rename_lib)
        df_sc_data["sample"] = df_sc_data["library"]
        df_sc_data["plate_ID"] = df_sc_data["sample"]
    elif "sample" in df_sc_data.columns:
        # plate-based single-cell data
        df_sc_data["plate_ID"] = df_sc_data["sample"].apply(custom_extract_plate_ID)
    else:
        raise ValueError("library or sample not found")

    df_sc_data["mouse"] = df_sc_data["sample"].apply(lambda x: x.split("-")[0])

    if plate_map is not None:
        df_sc_data["plate_ID"] = df_sc_data["plate_ID"].map(plate_map)

    df_sc_data["RNA_id"] = df_sc_data["plate_ID"] + "_RNA_" + df_sc_data["cell_bc"]
    df_sc_data["clone_id"] = df_sc_data["locus"] + "_" + df_sc_data["clone_id"]
    df_sc_data["allele"] = df_sc_data["locus"] + "_" + df_sc_data["allele"]
    return df_sc_data


def generate_sc_CARLIN_from_CARLIN_output(df_all, sample_name_format="LL"):
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
    df_sc_CARLIN = add_metadata(df_merge, sample_name_format=sample_name_format)

    def custom_extract_lineage(x):
        return extract_lineage(x, sample_name_format=sample_name_format)

    df_sc_CARLIN["lineage"] = df_merge["library"].apply(custom_extract_lineage)

    return df_sc_CARLIN


def assign_clone_id_by_integrating_locus(
    df_sc_CARLIN_raw,
    prob_cutoff=0.1,
    sample_count_cutoff=2,
    joint_allele_N_cutoff=6,
    locus_list=["CA", "TA", "RA"],
    clone_key="allele",
):
    """
    Integrate alleles from different locus to assign a common clone ID.
    After running this, you will still need to remove potentially ambiguous clones, typically with large allele_num (number of alleles (either from CC,TC, or RC) within the same assigned clone_id), and remove joint_CA_TA_RA alleles with a high frequency

    Parameters
    ----------
        df_sc_CARLIN_raw:
            A long-format dataframe storing: 'RNA_id', 'locus', 'normalized_count','allele'
        prob_cutoff:
            The probability cutoff to use an allele to establish strong connection between two CA-TA-RA clone IDs
        joint_allele_N_cutoff:
            An allele needs to have less than this number co-detected alleles from other locus to be used as a strong connection in the S matrix
            we found that this filterning is usually only necessary for TC, as for CC and RC, the alleles with high joint_allele_N also has high prob

    Returns
    -------
        df_assigned_clones:
            A dataframe, each entry gives the assigned clone_id and its relation to the detected CA-TA-RA joint allele
        df_sc_CARLIN:
            Update the input df_sc_CARLIN to add columns: 'joint_clone_id', 'joint_prob', 'joint_allele_num'
        df_allele:
            CA-TA-RA joint allele table


    ## example: this code helps to connect df_allele with df_assigned_clones for debugging
    ```python
    df_display=df_assigned_clones[df_assigned_clones['allele_num']==6]
    for j in range(10):
        print(f'clone-{j}')
        BC_ids=df_display.iloc[j]['BC_id']
        display(df_allele.iloc[BC_ids])
    ```
    """

    import scanpy as sc
    from tqdm import tqdm

    df_sc_CARLIN = df_sc_CARLIN_raw[
        (df_sc_CARLIN_raw["normalized_count"] < prob_cutoff)
        & (df_sc_CARLIN_raw["sample_count"] < sample_count_cutoff)
    ]
    print(
        f"renaming fraction after initial filtering (sample_count_cutoff={sample_count_cutoff}, prob_cutoff={prob_cutoff}): {len(df_sc_CARLIN)/len(df_sc_CARLIN_raw):.2f}"
    )

    # we are not working on mutations events within allele yet. So, we can use either 'allele' or 'clone_id'
    locus_BC_names = [f"{x}_BC" for x in locus_list]
    locus_prob_names = [f"{x}_prob" for x in locus_list]

    ## extract the alleles, probabilities, and generate CA-TA-RA joint allele ID
    df_1 = df_sc_CARLIN.pivot(
        index="RNA_id", columns="locus", values=[clone_key, "normalized_count"]
    )
    dict_BC_tmp = {f"{locus}_BC": df_1[(clone_key, locus)] for locus in locus_list}
    dict_prob_tmp = {
        f"{locus}_prob": df_1[("normalized_count", locus)] for locus in locus_list
    }
    dict_BC_tmp.update(dict_prob_tmp)
    df_cells = pd.DataFrame(dict_BC_tmp, index=df_1.index)
    df_cells["joint_clone_id_tmp"] = [
        "@".join(x) for x in df_cells[locus_BC_names].fillna("nan").to_numpy()
    ]
    df_allele = df_cells.drop_duplicates().reset_index(drop=True)

    ## adjust the allele frequency to make sure that we do not have high-frequency alleles that can make connection to multiple unrelated alleles
    allele_to_norm_count = dict(
        zip(df_sc_CARLIN[clone_key], df_sc_CARLIN["normalized_count"])
    )

    def count_unique_bc(x):
        return len(set(x.dropna()))

    # ## set alleles with more than sample_count_cutoff to have a high prob (does not seem to be useful here)
    # df_allele_tmp=df_sc_CARLIN[(df_sc_CARLIN['sample_count']>=sample_count_cutoff) & (df_sc_CARLIN['normalized_count']>prob_cutoff)]
    # df_allele_tmp['normalized_count']=prob_cutoff
    # allele_to_norm_count.update(
    #     dict(zip(df_allele_tmp[clone_key], df_allele_tmp["normalized_count"]))
    # )

    ## set alleles with more than joint_allele_N_cutoff jointly detected alleles to have a high prob
    for locus in locus_list:
        df_coupling = (
            df_allele[
                (~pd.isna(df_allele[f"{locus}_BC"]))
                & (df_allele[f"{locus}_BC"] != f"{locus}_[]")
            ]
            .groupby(f"{locus}_BC")
            .agg(joint_allele=("joint_clone_id_tmp", count_unique_bc))
            .reset_index()
        )
        df_coupling["normalized_count"] = df_coupling[f"{locus}_BC"].map(
            allele_to_norm_count
        )
        df_common = df_coupling[
            (df_coupling["joint_allele"] >= joint_allele_N_cutoff)
            & (df_coupling["normalized_count"] < prob_cutoff)
        ]
        df_common["normalized_count"] = prob_cutoff
        allele_to_norm_count.update(
            dict(zip(df_common[f"{locus}_BC"], df_common["normalized_count"]))
        )

    ## update the allele frequency in df_cells (contains RNA_id)
    for locus in locus_list:
        df_cells[f"{locus}_prob"] = df_cells[f"{locus}_BC"].map(allele_to_norm_count)
    df_cells["joint_prob"] = [
        np.prod(x) for x in df_cells[locus_prob_names].fillna(1).to_numpy()
    ]

    ## establish the allele connectivity (measured by probability) matrix for CC,TC,RC separately
    prob_matrix_list = []
    for locus in locus_list:
        print(locus)
        prob_matrix_tmp = np.ones((len(df_allele), len(df_allele)))
        BC_index = (
            df_allele[f"{locus}_BC"].dropna().index
        )  # dropna before the calculate, so that if they are two alleles are different,
        # they are really different. This is critical
        BC_values = df_allele[f"{locus}_BC"].dropna().values
        for i in tqdm(range(len(BC_index))):
            same_idx = (
                BC_values == BC_values[i]
            )  # in the future, we can update this to not request exact match, but they share mutations
            prob_matrix_tmp[BC_index[i], BC_index[same_idx]] = [
                allele_to_norm_count[__] for __ in BC_values[same_idx]
            ]
            prob_matrix_tmp[BC_index[i], BC_index[~same_idx]] = np.nan

        prob_matrix_list.append(prob_matrix_tmp)

    ## calcualte the joint connectivity (probability) of the CA-TA-RA array
    prob_matrix = prob_matrix_list[0]
    for __ in range(1, len(locus_list)):
        prob_matrix = prob_matrix * prob_matrix_list[__]  # multiply the 3 probability

    ## convert the joint connectivity matrix to a binarized similarity matrix by thresholding
    similarity_matrix = np.zeros((len(df_allele), len(df_allele)))
    similarity_matrix[prob_matrix < prob_cutoff] = (
        1  # strong connection by sharing rare alleles
    )
    similarity_matrix[prob_matrix >= prob_cutoff] = 0  # weak connection
    similarity_matrix[np.isnan(prob_matrix)] = np.nan  # mismatch

    ## partition the graph from the discretized similarity matrix into different components
    from scipy.sparse.csgraph import connected_components

    S_noNAN = np.nan_to_num(similarity_matrix)
    n_components, labels = connected_components(S_noNAN, directed=False)

    ## convert the classified clones into an annotated dataframe
    df_assigned_clones = (
        pd.DataFrame({"BC_id": np.arange(len(labels)), "clone_id": labels})
        .groupby("clone_id")
        .agg(
            BC_id=("BC_id", lambda x: list(set(x))),
            BC_num=("BC_id", lambda x: len(set(x))),
        )
    )

    is_nan_list = []
    for i in range(len(df_assigned_clones)):
        BC_ids = df_assigned_clones.iloc[i]["BC_id"]
        isnan = np.isnan(similarity_matrix[BC_ids][:, BC_ids]).sum()
        is_nan_list.append(isnan)
    df_assigned_clones["mismatch_num"] = is_nan_list

    df_assigned_clones["allele_list"] = df_assigned_clones["BC_id"].apply(
        lambda x: list(
            df_allele.iloc[x].filter(locus_BC_names).melt()["value"].dropna().unique()
        )
    )
    df_assigned_clones["allele_num"] = df_assigned_clones["allele_list"].apply(
        lambda x: len(x)
    )
    df_assigned_clones["joint_clone_id_tmp_list"] = df_assigned_clones["BC_id"].apply(
        lambda x: list(df_allele.iloc[x]["joint_clone_id_tmp"].unique())
    )
    df_assigned_clones["joint_clone_id"] = df_assigned_clones["allele_list"].apply(
        lambda x: "@".join(x)
    )

    df_assigned_clones_2 = df_assigned_clones.explode("joint_clone_id_tmp_list")
    df_cells["joint_clone_id"] = df_cells["joint_clone_id_tmp"].map(
        dict(
            zip(
                df_assigned_clones_2["joint_clone_id_tmp_list"],
                df_assigned_clones_2["joint_clone_id"],
            )
        )
    )

    df_sc_CARLIN = df_sc_CARLIN.set_index("RNA_id")
    df_sc_CARLIN["joint_clone_id"] = df_cells["joint_clone_id"]
    df_sc_CARLIN["joint_clone_id_tmp"] = df_cells["joint_clone_id_tmp"]
    df_sc_CARLIN["joint_prob"] = df_cells["joint_prob"]
    df_sc_CARLIN["joint_allele_num"] = df_sc_CARLIN["joint_clone_id"].map(
        dict(
            zip(df_assigned_clones["joint_clone_id"], df_assigned_clones["allele_num"])
        )
    )
    df_sc_CARLIN = df_sc_CARLIN.reset_index()
    return (
        df_assigned_clones,
        df_sc_CARLIN,
        df_allele.filter(locus_BC_names + ["joint_clone_id_tmp"]),
    )


def assign_clone_id_by_integrating_locus_v1(
    df_sc_CARLIN_raw,
    prob_cutoff=0.1,
    sample_count_cutoff=2,
    joint_allele_N_cutoff=6,
    locus_list=["CA", "TA", "RA"],
    consider_mutation=True,
):
    """
    This version is based on additive coupling strength and leiden clustering


    Integrate alleles from different locus to assign a common clone ID.
    After running this, you will still need to remove potentially ambiguous clones, typically with large allele_num (number of alleles (either from CC,TC, or RC) within the same assigned clone_id), and remove joint_CA_TA_RA alleles with a high frequency

    Parameters
    ----------
        df_sc_CARLIN:
            A long-format dataframe storing: 'RNA_id', 'locus', 'normalized_count','allele'
        prob_cutoff:
            The probability cutoff (<) to use an allele to establish strong connection between two CA-TA-RA clone IDs
        sample_count_cutoff:
            Sample count cutoff (<) for an allele to be used to establish strong connection between two CA-TA-RA clone IDs
        joint_allele_N_cutoff:
            An allele needs to have less than this number co-detected alleles from other locus to be used as a strong connection in the S matrix
            we found that this filterning is usually only necessary for TC, as for CC and RC, the alleles with high joint_allele_N also has high prob

    Returns
    -------
        df_assigned_clones:
            A dataframe, each entry gives the assigned clone_id and its relation to the detected CA-TA-RA joint allele
        df_sc_CARLIN:
            Update the input df_sc_CARLIN to add columns: 'joint_clone_id', 'joint_prob', 'joint_allele_num'
        df_allele:
            CA-TA-RA joint allele table


    ## example: this code helps to connect df_allele with df_assigned_clones for debugging
    ```python
    df_display=df_assigned_clones[df_assigned_clones['allele_num']==6]
    for j in range(10):
        print(f'clone-{j}')
        BC_ids=df_display.iloc[j]['BC_id']
        display(df_allele.iloc[BC_ids])
    ```
    """

    import scanpy as sc
    from tqdm import tqdm

    df_sc_CARLIN = df_sc_CARLIN_raw[
        (df_sc_CARLIN_raw["normalized_count"] < prob_cutoff)
        & (df_sc_CARLIN_raw["sample_count"] < sample_count_cutoff)
    ]
    print(
        f"renaming fraction after initial filtering (sample_count_cutoff={sample_count_cutoff}, prob_cutoff={prob_cutoff}): {len(df_sc_CARLIN)/len(df_sc_CARLIN_raw):.2f}"
    )

    # we are not working on mutations events within allele yet. So, we can use either 'allele' or 'clone_id'
    clone_key = "allele"
    locus_BC_names = [f"{x}_BC" for x in locus_list]
    locus_prob_names = [f"{x}_prob" for x in locus_list]

    ## extract the alleles, probabilities, and generate CA-TA-RA joint allele ID
    df_1 = df_sc_CARLIN.pivot(
        index="RNA_id", columns="locus", values=[clone_key, "normalized_count"]
    )
    dict_BC_tmp = {f"{locus}_BC": df_1[(clone_key, locus)] for locus in locus_list}
    dict_prob_tmp = {
        f"{locus}_prob": df_1[("normalized_count", locus)] for locus in locus_list
    }
    dict_BC_tmp.update(dict_prob_tmp)
    df_cells = pd.DataFrame(dict_BC_tmp, index=df_1.index)
    df_cells["joint_clone_id_tmp"] = [
        "@".join(x) for x in df_cells[locus_BC_names].fillna("nan").to_numpy()
    ]
    df_allele = df_cells.drop_duplicates().reset_index(drop=True)

    ## adjust the allele frequency to make sure that we do not have high-frequency alleles that can make connection to multiple unrelated alleles
    print(
        f"Adjust allele frequency for alleles co-detected with {joint_allele_N_cutoff} alleles in other locus "
    )
    allele_to_norm_count = dict(
        zip(df_sc_CARLIN[clone_key], df_sc_CARLIN["normalized_count"])
    )

    def count_unique_bc(x):
        return len(set(x.dropna()))

    ## set alleles with more than sample_count_cutoff to have a high prob
    df_allele_tmp = df_sc_CARLIN[df_sc_CARLIN["sample_count"] >= sample_count_cutoff]
    print(len(df_allele_tmp) / len(df_sc_CARLIN))
    df_allele_tmp["normalized_count"] = prob_cutoff
    allele_to_norm_count.update(
        dict(zip(df_allele_tmp[clone_key], df_allele_tmp["normalized_count"]))
    )

    ## set alleles with more than joint_allele_N_cutoff jointly detected alleles to have a high prob
    for locus in locus_list:
        df_coupling = (
            df_allele[
                (~pd.isna(df_allele[f"{locus}_BC"]))
                & (df_allele[f"{locus}_BC"] != f"{locus}_[]")
            ]
            .groupby(f"{locus}_BC")
            .agg(joint_allele=("joint_clone_id_tmp", count_unique_bc))
            .reset_index()
        )
        df_coupling["normalized_count"] = df_coupling[f"{locus}_BC"].map(
            allele_to_norm_count
        )
        df_common = df_coupling[
            (df_coupling["joint_allele"] >= joint_allele_N_cutoff)
            & (df_coupling["normalized_count"] < prob_cutoff)
        ]
        df_common["normalized_count"] = prob_cutoff
        allele_to_norm_count.update(
            dict(zip(df_common[f"{locus}_BC"], df_common["normalized_count"]))
        )

    ## update the allele frequency in df_cells (contains RNA_id)
    for locus in locus_list:
        df_cells[f"{locus}_prob"] = df_cells[f"{locus}_BC"].map(allele_to_norm_count)
    df_cells["joint_prob"] = [
        np.prod(x) for x in df_cells[locus_prob_names].fillna(1).to_numpy()
    ]

    ## establish the allele connectivity (measured by probability) matrix for CC,TC,RC separately
    similarity_matrix = np.zeros((len(df_allele), len(df_allele)))
    for locus in ["CA", "TA", "RA"]:

        if consider_mutation:
            df_tmp = pd.DataFrame(df_allele[f"{locus}_BC"].dropna())
            df_tmp[f"{locus}_mutation"] = df_tmp[f"{locus}_BC"].apply(
                lambda x: x[3:].split(",")
            )
            df_tmp = df_tmp.explode(f"{locus}_mutation")
            df_tmp["prob"] = df_tmp[f"{locus}_BC"].map(allele_to_norm_count)
            df_tmp[f"{locus}_mutation"] = locus + "_" + df_tmp[f"{locus}_mutation"]
            mut_index = df_tmp[f"{locus}_mutation"].isin(allele_to_norm_count.keys())
            df_tmp.loc[mut_index, "prob"] = df_tmp.loc[
                mut_index, f"{locus}_mutation"
            ].map(allele_to_norm_count)
            column_key = f"{locus}_mutation"
        else:
            df_tmp = pd.DataFrame(df_allele[f"{locus}_BC"].dropna())
            df_tmp["prob"] = df_tmp[f"{locus}_BC"].map(allele_to_norm_count)
            column_key = f"{locus}_BC"

        """
        matrix_X: cell by allele matrix, n_cell * unique n_allele, a matrix of 1 and -1. 
                1 means that the corresponding allele are detected in this cell, -1: not detected
        mutation_value_matrix: n_allele * n_allele, the weight score for each unique allele. 
                It is a diagonal matrix
        allele_similarity_matrix: np.dot(matrix_X,mutation_value_matrix).dot(matrix_X.T),
                A n_cell by n_cell matrix. This formula gives only the match score + un-detected score - mismatch score 
                What we want is the the match score - mismatch score for each pair of cells. This is taken care of later. 
                
        A useful code to understand this process
        ```python
        matrix_X=np.array([[-1,-1,-1,1],[-1,-1,-1,1],[1,-1,-1,-1],[1,-1,-1,1]])
        mutation_value_matrix=np.diag([1,10,100,1000])

        similarity_matrix_tmp=np.dot(matrix_X,mutation_value_matrix).dot(matrix_X.T)
        print('match score + un-detected score - mismatch score ')
        print(similarity_matrix_tmp)

        matrix_X_inverse=(matrix_X<0).astype(int) # inverse the matrix, the detected alele is 0 and undetected is 1
        similarity_matrix_undetect=np.dot(matrix_X_inverse,mutation_value_matrix).dot(matrix_X_inverse.T)
        print('un-detected score')
        print(similarity_matrix_undetect)

        print('Final connectivity')
        Final_connectivity=similarity_matrix_tmp-similarity_matrix_undetect
        print(Final_connectivity)
        ```
        """

        kernel = lambda x: np.exp(-((x / 0.1) ** 2))
        # kernel=lambda x: abs(np.log(x+10**(-4)))
        df_tmp["value"] = 1
        df_tmp["transformed_connectivity"] = df_tmp["prob"].apply(kernel)
        df_tmp = df_tmp.reset_index()

        df_count = df_tmp.pivot(
            index="index", columns=column_key, values="value"
        ).fillna(
            -1
        )  # note that we use -1 to represent un-detected allele/mutations
        matrix_X = df_count.to_numpy()

        mutation_value = dict(
            zip(df_tmp[column_key], df_tmp["transformed_connectivity"])
        )
        mutation_value_matrix = np.diag([mutation_value[x] for x in df_count.columns])
        # mutation_value_matrix=np.diag(np.ones(len(df_count.columns))) ## this is for test

        vector_mask = np.zeros(len(df_allele))
        vector_mask[df_tmp["index"]] = 1
        matrix_mask = (
            np.dot(vector_mask[:, np.newaxis], vector_mask[:, np.newaxis].T) > 0
        )

        ## since we use -1 to represented un-detected allele/mutation, this gives the match score + un-detected score - mismatch score
        allele_similarity_matrix = np.dot(matrix_X, mutation_value_matrix).dot(
            matrix_X.T
        )
        similarity_matrix_tmp = np.zeros((len(df_allele), len(df_allele)))
        similarity_matrix_tmp[matrix_mask] = allele_similarity_matrix.flatten()
        similarity_matrix += similarity_matrix_tmp

        ## now, remove the un-detected score
        matrix_X_inverse = (matrix_X < 0).astype(
            int
        )  # inverse the matrix, the detected alele is 0 and undetected is 1
        allele_similarity_matrix_undetect = np.dot(
            matrix_X_inverse, mutation_value_matrix
        ).dot(matrix_X_inverse.T)
        similarity_matrix_tmp_undetect = np.zeros((len(df_allele), len(df_allele)))
        similarity_matrix_tmp_undetect[matrix_mask] = (
            allele_similarity_matrix_undetect.flatten()
        )
        similarity_matrix -= similarity_matrix_tmp_undetect

        # similarity_matrix_list.append(similarity_matrix_tmp)

    # similarity_matrix[similarity_matrix<kernel(prob_cutoff)]=0
    similarity_matrix_ps = similarity_matrix.copy()
    similarity_matrix_ps[similarity_matrix_ps < kernel(prob_cutoff)] = (
        0  # only consider positive weights for now
    )

    ####### WARN: only consider positive weights for now
    A = ssp.csr_matrix(similarity_matrix_ps)  # only consider positive weights for now
    adata = sc.AnnData(A)
    sc.tl.leiden(adata, adjacency=A, resolution=3)

    df_assigned_clones = (
        pd.DataFrame(
            {"BC_id": np.arange(adata.shape[0]), "clone_id": adata.obs["leiden"]}
        )
        .groupby("clone_id")
        .agg(
            BC_id=("BC_id", lambda x: list(set(x))),
            BC_num=("BC_id", lambda x: len(set(x))),
            BC_consistency=("BC_id", lambda x: similarity_matrix[x][:, x].mean()),
        )
    )

    is_nan_list = []
    for i in range(len(df_assigned_clones)):
        BC_ids = df_assigned_clones.iloc[i]["BC_id"]
        isnan = np.isnan(similarity_matrix[BC_ids][:, BC_ids]).sum()
        is_nan_list.append(isnan)
    df_assigned_clones["mismatch_num"] = is_nan_list

    df_assigned_clones["allele_list"] = df_assigned_clones["BC_id"].apply(
        lambda x: list(
            df_allele.iloc[x].filter(locus_BC_names).melt()["value"].dropna().unique()
        )
    )
    df_assigned_clones["allele_num"] = df_assigned_clones["allele_list"].apply(
        lambda x: len(x)
    )
    df_assigned_clones["joint_clone_id_tmp_list"] = df_assigned_clones["BC_id"].apply(
        lambda x: list(df_allele.iloc[x]["joint_clone_id_tmp"].unique())
    )
    df_assigned_clones["joint_clone_id"] = df_assigned_clones["allele_list"].apply(
        lambda x: "@".join(x)
    )

    df_assigned_clones_2 = df_assigned_clones.explode("joint_clone_id_tmp_list")
    df_cells["joint_clone_id"] = df_cells["joint_clone_id_tmp"].map(
        dict(
            zip(
                df_assigned_clones_2["joint_clone_id_tmp_list"],
                df_assigned_clones_2["joint_clone_id"],
            )
        )
    )

    df_sc_CARLIN = df_sc_CARLIN.set_index("RNA_id")
    df_sc_CARLIN["joint_clone_id"] = df_cells["joint_clone_id"]
    df_sc_CARLIN["joint_clone_id_tmp"] = df_cells["joint_clone_id_tmp"]
    df_sc_CARLIN["joint_prob"] = df_cells["joint_prob"]
    df_sc_CARLIN["joint_allele_num"] = df_sc_CARLIN["joint_clone_id"].map(
        dict(
            zip(df_assigned_clones["joint_clone_id"], df_assigned_clones["allele_num"])
        )
    )
    df_sc_CARLIN["BC_consistency"] = df_sc_CARLIN["joint_clone_id"].map(
        dict(
            zip(
                df_assigned_clones["joint_clone_id"],
                df_assigned_clones["BC_consistency"],
            )
        )
    )
    df_sc_CARLIN = df_sc_CARLIN.reset_index()
    return (
        df_assigned_clones,
        df_sc_CARLIN,
        df_allele.filter(locus_BC_names + ["joint_clone_id_tmp"]),
    )


def filter_high_quality_joint_clones(
    df_sc_CARLIN, joint_prob_cutoff=0.1, joint_allele_num_cutoff=6
):
    #  return df_sc_CARLIN.assign(
    # HQ=lambda x: (x["sample_count"] <= BC_max_sample_count)
    # & (x["normalized_count"] <= BC_max_freq)
    # ).query("HQ==True")

    return df_sc_CARLIN[
        (df_sc_CARLIN["joint_prob"] < joint_prob_cutoff)
        & (df_sc_CARLIN["joint_allele_num"] < joint_allele_num_cutoff)
    ]


def filter_high_quality_single_alleles(
    df_data, normalized_count_cutoff=0.1, sample_count_cutoff=1
):

    return df_data[
        (df_data["normalized_count"] < normalized_count_cutoff)
        & (df_data["sample_count"] <= sample_count_cutoff)
    ]
