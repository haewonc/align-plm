import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt 

path = 'proteingym'
out_dir = 'figure'
train_ratio = 0.7
reference = pd.read_csv(f'{path}/reference.csv')

if not os.path.exists(out_dir):
    os.makedirs(out_dir)

for dms_csv in reference['DMS_filename']:
    dms = pd.read_csv(f'{path}/{dms_csv}')
    dms = dms['DMS_score'].to_numpy()
    fig, ax = plt.subplots(figsize=(10, 6))
    counts, bins, patches = ax.hist(dms, bins=12, facecolor='skyblue', edgecolor='gray')
    ax.set_xticks(bins)
    fig.tight_layout()
    plt.savefig(f"{out_dir}/{dms_csv.replace('csv','png')}", dpi=150)
    plt.clf()
    plt.cla()
    plt.close()