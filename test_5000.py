import pooch
import scanpy as sc
from joblib import Parallel, delayed
import os
import glob
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

file_path = pooch.retrieve(
    # URL to one of Pooch's test files
    fname="whole_taxonomy_MTG_AD.h5ad",
    url="https://datasets.cellxgene.cziscience.com/c32964d2-3339-441f-8e56-7177234c7876.h5ad",
    path="./data/",
    known_hash="sha256:17d988d683383c707213973095400f5cbfdf8ab98b122b408904ae8ca470d791",
    progressbar=True
)

data_raw_path = "./data/whole_taxonomy_MTG_AD.h5ad"
#adata = sc.read_h5ad(data_raw_path, backed="r")

batch_size = 5000
dict_types ={
    "sc" : "10x 3' v3",
    "sn" : "10x multiome"
    }

fields = ["Genes detected", "total_counts", "pct_counts_mt"]

for key in dict_types.keys():
    dir = f"./data/{key}_test"
    if not os.path.exists(dir):
        os.makedirs(dir)

def get_batch(i, batch_size, path, dict_types):
    adata = sc.read_h5ad(path, backed="r")
    batch = adata[i:i + batch_size, :].to_memory()
    batch_dict = {}
    for key, value in dict_types.items():
        batch_dict[key] = batch[batch.obs["assay"] == value].copy()
    return batch_dict

def save_batch(batch_dict, index):
    for key, value in batch_dict.items():
        filename = f"./data/{key}_test/MTG_batch_{index}.h5ad"
        print(value.n_obs)
        value.write(filename)
        print(f"Saved {filename}")

def get_save(i, batch_size, path, dict_types):
    batch_dict = get_batch(i, batch_size, path, dict_types)
    save_batch(batch_dict, i)

def filtered_qc(path, fields):
    adata = sc.read_h5ad(path)
    # mitochondrial genes, "MT-" for human, "Mt-" for mouse
    adata.var["mt"] = adata.var_names.str.startswith("MT-")
    # ribosomal genes
    adata.var["ribo"] = adata.var_names.str.startswith(("RPS", "RPL"))
    # hemoglobin genes
    adata.var["hb"] = adata.var_names.str.contains("^HB[^(P)]")
    sc.pp.calculate_qc_metrics(
        adata, qc_vars=["mt", "ribo", "hb"], inplace=True, log1p=True
    )
    return adata.obs[fields].copy()

Parallel(n_jobs=8)(
    delayed(get_save)(i, 5000, data_raw_path, dict_types)
    for i in range(0, 5000, batch_size)
)

for key in dict_types.keys():
    all_files = glob.glob(f"./data/{key}_test/*.h5ad")
    results = Parallel(n_jobs = 8)(
        delayed(filtered_qc)(file, fields)
        for file in all_files    
    )
    melted = pd.melt(
        results[0],
        value_vars=fields,
        var_name="feature",
        value_name="value"
    )
    for feature_name in melted["feature"].unique():
        feature_data = melted[melted["feature"] == feature_name]
    
        plt.figure(figsize=(6, 4))
        sns.violinplot(data=feature_data, x="feature", y="value", inner="box", density_norm='width')
        sns.stripplot(data=feature_data, x="feature", y="value", color="k", jitter=0.4, size=0.1)
        plt.title(f"Violin plot for {feature_name}")
        plt.ylabel("Value")
        plt.xlabel("")
        plt.tight_layout()
        plt.show()


