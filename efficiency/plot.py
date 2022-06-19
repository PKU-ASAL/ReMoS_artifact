import os
import os.path as osp
import numpy as np
import csv
from datetime import datetime
from pdb import set_trace as st
from matplotlib import pyplot as plt
plt.switch_backend('agg')
datasets = ["mit67", "cub200"]
finetune_dataset_to_name = {
    "mit67": "resnet18_mit67_lr5e-3_iter30000_feat0_wd1e-4_mmt0_1",
    "cub200": "resnet18_cub200_lr5e-3_iter30000_feat0_wd1e-4_mmt0_1",
}
remos_dataset_to_name = {
    "mit67": "resnet18_mit67_do_total0.05_trainall_lr5e-3_iter30000_feat0_wd1e-4_mmt0_1",
    "cub200": "resnet18_cub200_do_total0.05_trainall_lr5e-3_iter30000_feat0_wd1e-4_mmt0_1",
}
retrain_dataset_to_name = {
    "mit67": "resnet18_mit67_reinit_lr1e-2_iter30000_feat0_wd5e-3_mmt0.9_1",
    "cub200": "resnet18_cub200_reinit_lr1e-2_iter30000_feat0_wd5e-3_mmt0.9_1",
}
dataset_abbr_to_name = {
    "mit67": "Scenes",
    "cub200": "Birds",
}

def load_tsv(path, dataset):
    print(path)
    with open(path, ) as fd:
        rd = csv.reader(fd, delimiter="\t", quotechar='"')
        log_times, accs = [], []
        for idx, row in enumerate(rd):
            # print(row)
            if idx == 0:
                continue
            log_time = row[0]
            acc = row[2]
            time = datetime.strptime(log_time[:-2], "%b %d %H:%M")
            if idx == 1:
                start = time
            relative_time = (time - start).total_seconds() / 60
            if relative_time in log_times:
                continue
            log_times.append(relative_time )
            accs.append(float(acc))

    return log_times, accs

FONTSIZE = 20
LINEWIDTH = 3
def plot(dataset, ax, draw_ylabel=False):
    finetune_path = osp.join("speed", "finetune", finetune_dataset_to_name[dataset], "test.tsv")
    finetune_logs = load_tsv(finetune_path, dataset)
    finetune_max_time = finetune_logs[0][-1]
    
    ncprune_path = osp.join("speed", "remos", remos_dataset_to_name[dataset], "test.tsv")
    ncprune_logs = load_tsv(ncprune_path, dataset)
    ncprune_max_time = ncprune_logs[0][-1]
    ncprune_time_scale = finetune_max_time / ncprune_max_time
    ncprune_logs = ( [t * ncprune_time_scale for t in ncprune_logs[0]], ncprune_logs[1])
    
    retrain_path = osp.join("speed", "retrain", retrain_dataset_to_name[dataset], "test.tsv")
    retrain_logs = load_tsv(retrain_path, dataset)
    ncprune_max_time = retrain_logs[0][-1]
    retrain_time_scale = finetune_max_time / ncprune_max_time
    retrain_logs = ( [t * retrain_time_scale for t in retrain_logs[0]], retrain_logs[1])
    
    ax.plot(
        finetune_logs[0], finetune_logs[1], 
        linewidth=LINEWIDTH,
        label="Fine-tuning", 
        linestyle='solid', alpha=0.5, color="blue",
    )
    
    ax.plot(
        retrain_logs[0], retrain_logs[1], 
        linewidth=LINEWIDTH,
        label="Retraining", 
        linestyle='dotted', alpha=0.5, color="red", 
    )
    
    ax.plot(
        ncprune_logs[0], ncprune_logs[1], 
        linewidth=LINEWIDTH,
        label="ReMoS", 
        linestyle='dashed', alpha=0.5, color="green",
    )
    
    ax.set_ylim(0, 80)
    # x_labels = labels = [item.get_text() for item in ax.get_xticklabels()]
    # ax.set_xticklabels(x_labels, fontsize=FONTSIZE)
    ax.set_xticks([50, 100, 150, 200])
    ax.set_xticklabels([50, 100, 150, 200], fontsize=FONTSIZE)
    ax.set_yticks([20, 40, 60, 80])
    ax.set_yticklabels([20, 40, 60, 80], fontsize=FONTSIZE)
    # for tick in ax.xaxis.get_major_ticks():
    #     tick.label.set_fontsize(FONTSIZE) 
    # for tick in ax.yaxis.get_major_ticks():
    #     tick.label.set_fontsize(FONTSIZE) 
    ax.set_title(dataset_abbr_to_name[dataset], fontsize=FONTSIZE)
    ax.set_xlabel("Minutes", fontsize=FONTSIZE)
    if draw_ylabel:
        ax.set_ylabel("Accuracy", fontsize=FONTSIZE)


fig, axs = plt.subplots(1, 2, figsize=(8, 4), )
plot("mit67", axs[0], draw_ylabel=True)
plot("cub200", axs[1])

axLine, axLabel = axs[0].get_legend_handles_labels()
fig.legend(axLine, axLabel,           
        #    bbox_to_anchor= (0.5, 0.0),
        loc = (0.67, 0.2), ncol=1, fontsize=FONTSIZE)

plt.tight_layout()
plt.savefig(f"speed.pdf")
plt.clf()