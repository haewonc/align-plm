import pandas as pd 
import random 
from tqdm import tqdm 
import os 
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--preference', default='preference_500', help='Root directory to pair data')
parser.add_argument('--ablation', action='store_true', default=False)
args = parser.parse_args()

if not args.ablation:
    reference = pd.read_csv(f'results/reference_main_sft.csv')
    distances = reference['distance']
else:
    reference = pd.read_csv(f'reference/reference_ablation.csv')

dms_csvs = reference['DMS_filename']
print('Start processing...')

for dms_index in tqdm(range(len(dms_csvs))):
    dms_csv = dms_csvs[dms_index]
    out_csv = f"{args.preference}/{dms_csv.replace('.csv','')}"
    df = pd.read_csv(f'{out_csv}/dms_train.csv')
    sorted_df = df.sort_values(by='DMS_score', ascending=False, ignore_index=True)
    train = [s[0] for s in sorted_df[['mutated_sequence']].values.tolist()]
    train_size = len(train)
    
    if not args.ablation:
        d_ref = distances[dms_index]
        pairs = []
        for i in range(train_size-d_ref):
            for j in range(i+d_ref, min(i+2*d_ref, train_size)):
                pairs.append([train[i], train[j]])
        random.shuffle(pairs)
        if len(pairs) > 5000:
            pairs = pairs[:5000]
        pairs = pd.DataFrame(pairs, columns=['preferred', 'dispreferred'])
        pairs.to_csv(os.path.join(os.path.join(out_csv, f'd_ref.csv')), index=False)
    else:
        thresholds = [1, 32, 64, 128, 256]
        for threshold in thresholds:
            pairs = []
            for i in range(train_size-threshold):
                for j in range(i+threshold, min(i+2*threshold, train_size)):
                    pairs.append([train[i], train[j]])
            random.shuffle(pairs)
            if len(pairs) > 5000:
                pairs = pairs[:5000]
            pairs = pd.DataFrame(pairs, columns=['preferred', 'dispreferred'])
            pairs.to_csv(os.path.join(os.path.join(out_csv, f'd_{threshold}.csv')), index=False)