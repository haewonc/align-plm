import argparse
import pandas as pd
from core.sft import sft
import os

parser = argparse.ArgumentParser()
parser.add_argument('--name', default='sft_all')
parser.add_argument('--preference', default='preference_500', help='Root directory to pair data')
parser.add_argument('--size', default='Large', choices=['Large', 'Small'])
parser.add_argument('--device', default='cuda:0')
parser.add_argument('--logger', default=None, choices=['wandb'])
args = parser.parse_args()

if not os.path.exists('saved'):
    os.mkdir('saved')

reference = pd.read_csv('reference/reference_main.csv')
list_DMS = reference["DMS_id"]
tran_size = f'Tranception_{args.size}'

results = []
sft_ckpts = []

for DMS_index in range(len(list_DMS)):
    DMS_id = list_DMS[DMS_index]
    best_ckpt, base, best_sft = sft(tran_size=tran_size, 
                                            DMS_id=DMS_id, 
                                            preference=args.preference, 
                                            exp_name=args.name,
                                            total_epochs=6,
                                            learning_rate=2e-5,
                                            device=args.device,
                                            log=args.logger,
                                            quiet=False,
                                            save_model=True)
    results.append([DMS_id, base, best_sft, best_ckpt])
    sft_ckpts.append(best_ckpt)

if not os.path.exists('results'):
    os.mkdir('results')
pd.DataFrame(results, columns=['DMS_id', tran_size +" no retrieval (val)", 'After SFT (val)', 'SFT ckpt']).to_csv(
    f'results/{tran_size}_{args.preference}_sft_results.csv', index=False)

reference['SFT ckpt'] = sft_ckpts
reference.to_csv('results/reference_main_sft.csv', index=False)