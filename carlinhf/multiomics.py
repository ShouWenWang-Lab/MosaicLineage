import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as ssp
import scipy.stats as stats
import seaborn as sns


def preprocessing_R(df_input,minimum_N=5,mini_coverage=0.2,feature_N=5000):
    
    if len(df_input.columns)!=7:
        raise ValueError("input must has 6 columns: ['sample', 'id', 'anno', 'Nmet', 'N', 'rate','lineage']")
    os.makedirs('data/tmp',exist_ok=True)
    print('write to data/tmp')
    df_input.to_csv("data/tmp/before_preprocessing.tsv",sep='\t',index=0)
    script_dir='/n/groups/klein/shouwen/lili_project/scnmt_gastrulation/metacc'
    print('run Rscript')
    os.system(f"Rscript {script_dir}/preprocessing.R --minimum_N {minimum_N}   --mini_coverage {mini_coverage}   --feature_N {feature_N}")
    print('load from data/tmp')
    df_output=pd.read_csv('data/tmp/post_preprocessing.tsv',index_col=0)
    return df_output

def dimension_reduction(df_input):
    """
    We did not perform the log transform here, but this is advised before you run this. 
    The input should have 3 columns: ['sample','id','m']
    """
    df_temp2=df_input.filter(['sample','id','m']).pivot('sample','id','m')
    df_temp3=df_temp2.fillna(0)

    adata=sc.AnnData(ssp.csr_matrix(df_temp3.to_numpy()))
    adata.obs_names=df_temp3.index
    adata.var_names=np.array(df_temp3.columns)

    sc.tl.pca(adata, svd_solver='arpack')

    sc.pp.neighbors(adata, n_neighbors=10, n_pcs=40)

    sc.tl.umap(adata,min_dist = 0.5)

    sc.tl.tsne(adata)

    sc.tl.leiden(adata,resolution=1)
    return adata
