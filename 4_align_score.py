import os
import argparse
import pandas as pd
from core.align import align
import os
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = 'true'

parser = argparse.ArgumentParser()
parser.add_argument('--name', default='main_align')
parser.add_argument('--preference', default='preference_500')
parser.add_argument('--size', default='Large', choices=['Large', 'Small'])
parser.add_argument('--device', default='cuda:0')
parser.add_argument('--beta', type=float, default=0.4)
parser.add_argument('--tokenizer', default='Basic_tokenizer')
parser.add_argument('--split', default='d_ref')
parser.add_argument('--logger', default=None, choices=['wandb'])
args = parser.parse_args()

exp_name = f'{args.name}_{args.split}'
reference = pd.read_csv(f'results/reference_main_sft.csv')
results = []
align_ckpts = []

for DMS_index in range(reference.shape[0]):
    DMS_id = reference["DMS_id"][DMS_index]
    best_ckpt, test_score = align(tran_size=f'Tranception_{args.size}', 
                                                    DMS_id=DMS_id, 
                                                    tran_name=reference["SFT ckpt"][DMS_index],
                                                    exp_name=exp_name,
                                                    preference=args.preference, 
                                                    total_epochs=1,
                                                    device=args.device,
                                                    rdevice=args.device,
                                                    split=args.split,
                                                    beta=args.beta,
                                                    learning_rate=2e-5,
                                                    log=args.logger,
                                                    quiet=False,
                                                    save_model=True, 
                                                    save_score=True)
    results.append([DMS_id, test_score, best_ckpt]) 
    align_ckpts.append(best_ckpt)

pd.DataFrame(results, columns=['DMS_id', 'After Align (val)', 'Align ckpt']).to_csv( 
    f'results/{exp_name}_{args.preference}_align_results.csv', index=False)

reference['Align ckpt'] = align_ckpts
reference.to_csv(f'results/reference_main_aligned.csv')