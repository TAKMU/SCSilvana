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

dict_types ={
    "sc" : "10x 3' v3",
    "sn" : "10x multiome"
    }

df = pd.read_csv("donor_id.csv")
donors = df["donor_id"].to_list()

results = Parallel(n_jobs=4)(
    delayed(dp.filter_id)(donor, i, data_raw_path)
    for i, donor in enumerate(donors)
)