import scanpy as sc
from joblib import Parallel, delayed
import os
import glob
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np 
import scipy.sparse as sparse
from scipy import io
import download_pooch as dp
import pandas as pd


dir = f"./data/"
if not os.path.exists(dir):
    os.makedirs(dir)


data_raw_path = "./data/whole_taxonomy_MTG_AD.h5ad"
adata = sc.read_h5ad(data_raw_path, backed="r")
batch_size = 50000

dict_types ={
    "sc" : "10x 3' v3",
    "sn" : "10x multiome"
    }

Parallel(n_jobs=4)(
    delayed(dp.get_batch_X)(i, batch_size, data_raw_path)
    for i in range(0, adata.n_obs, batch_size)
)
chunks = sorted(glob.glob(f"./data/chunk_*.mtx"))
X_final = sparse.vstack([io.mmread(chunk) for chunk in chunks])
io.mmwrite("./data/matriz_procesada_completa.mtx", X_final)

batch_size = 50000
results = Parallel(n_jobs=4)(
    delayed(dp.get_batch_cell_id)(i, batch_size, data_raw_path)
    for i in range(0, adata.n_obs, batch_size)
)
cell_ids = pd.concat(results)

results = Parallel(n_jobs=4)(
    delayed(dp.get_batch_gene_id)(i, batch_size, data_raw_path)
    for i in range(0, adata.n_obs, batch_size)
)
gene_ids = pd.concat(results)

results = Parallel(n_jobs=4)(
    delayed(dp.get_batch_metadata)(i, batch_size, data_raw_path)
    for i in range(0, adata.n_obs, batch_size)
)
metadata_cells = pd.concat(results)


cell_ids.to_csv("./data/cell_ids.csv", index=False)
gene_ids.to_csv("./data/gene_ids.csv", index=False)
metadata_cells.to_csv("./data/metadata_cells.csv")