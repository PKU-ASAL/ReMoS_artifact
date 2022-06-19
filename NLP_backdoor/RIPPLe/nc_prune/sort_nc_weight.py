import os
import os.path as osp
import argparse
import numpy as np
from pdb import set_trace as st
import pickle
import time

parser = argparse.ArgumentParser()
parser.add_argument("--output_dir")
args = parser.parse_args()


path = osp.join(args.output_dir, "accumulate_coverage.pkl")
with open(path, "rb") as f:
    accumulate_coverage = pickle.load(f)


all_weight_coverage = {}
for layer_name, (input_coverage, output_coverage) in accumulate_coverage.items():
    input_dim, output_dim = len(input_coverage), len(output_coverage)
    weight_coverage = np.zeros((output_dim, input_dim))
    weight_coverage = weight_coverage + input_coverage
    weight_coverage = weight_coverage + output_coverage[:,np.newaxis]
    all_weight_coverage[layer_name] = weight_coverage
    
    # for input_idx in range(input_dim):
    #     for output_idx in range(output_dim):
    #         coverage_score = input_coverage[input_idx] + output_coverage[output_idx]
    #         all_weight_coverage.append((coverage_score,  (layer_name, input_idx, output_idx)))

path = osp.join(args.output_dir, "all_weight_coverage.pkl")
with open(path, "wb") as f:
    pickle.dump(all_weight_coverage, f)