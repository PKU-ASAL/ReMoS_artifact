import os, sys
import os.path as osp
import pandas as pd
from pdb import set_trace as st
import numpy as np
np.set_printoptions(precision = 1)

root = "RIPPLe"
attacks = ["badnet", "L0.1_noes"]
attack_names = ["Data Poison", "Weight Poison"]
dataset_names = {
    "sst2": "SST-2",
    "imdb": "IMDB"
}
models = ["bert", "roberta"]
model_names = ["BERT", "RoBERTa"]
methods = ["finetune", "magprune", "remos"]
method_names = ["Fine-tune", "Mag-prune", "ReMoS"]
pairs = [
    "sst_to_sst",
    "imdb_to_imdb",
    "sst_to_imdb",
    "imdb_to_sst"
]
pair_names = [
    "SST2-SST2",
    "IMDB-IMDB",
    "SST2-IMDB",
    "IMDB-SST2"
]
pair_to_src = {
    "sst_to_sst": "sst2",
    "imdb_to_imdb": "imdb",
    "sst_to_imdb": "sst2",
    "imdb_to_sst": "imdb"
}

def summarize_model(model):
    m_indexes = pd.MultiIndex.from_product([attack_names, method_names, ["ACC", "DIR"]], names=["Dataset", "Techniques", "Metrics"])
    result = pd.DataFrame(np.random.randn(4, 2*3*2), index=pair_names, columns=m_indexes)
    for attack, attack_name in zip(attacks, attack_names):
        for pair, pair_name in zip(pairs, pair_names):
            for method, method_name in zip(methods, method_names):
                src_dataset = pair_to_src[pair]
                dir = osp.join(root, f"{model}_weights", f"{src_dataset}_{attack}", f"{model}_weights", f"{method}_{pair}_{attack}")
                path = osp.join(dir, "eval5_summary.txt")
                if not osp.exists(path):
                    for dirpath, dirnames, filenames in os.walk(dir):
                        if len(dirnames) > 0:
                            path = osp.join(dir, dirnames[0], "eval5_summary.txt")
                            break

                if not osp.exists(path):
                    print(path)
                    st()
                with open(path) as f:
                    lines = f.readlines()
                f1 = float(lines[2].split()[-1])
                lfr  = float(lines[-2].split()[-1])
                result[(attack_name, method_name, "ACC")][pair_name] = round(f1*100,2)
                result[(attack_name, method_name, "DIR")][pair_name] = round(lfr*100,2)
    return result
            
bert_result = summarize_model("bert")
roberta_result = summarize_model("roberta")
            
print(f"Results for BERT")
print(bert_result)
print(f"\nResults for RoBERTa")
print(roberta_result)