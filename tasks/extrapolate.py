import torch
import pandas as pd
from transformers import PreTrainedTokenizerFast
import tranception
from tqdm import tqdm  
import warnings
import numpy as np 
import argparse
warnings.filterwarnings('ignore')
import os
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = 'true'

parser = argparse.ArgumentParser()
parser.add_argument('--task', choices=['ace2', 'aav'])
parser.add_argument('--ckpt', default=None)
args = parser.parse_args()

if args.task == 'ace2':
    wt = 'STIEEQAKTFLDKFNHEAEDLFYQSSLASWNYNwTNITEENVQNMNNAGDKWSAFLKEQSTLAQMYPLQEIQNLTVKLQLQALQ'
    mutation_range = list(set(range(len(wt))) - set(range(wt.find('NTNITEEN'), wt.find('NTNITEEN')+len('NTNITEEN'))))
    # Following ICE paper, 8 amino acids (NTNITEEN) is kept fixed
elif args.task == 'aav':
    wt = 'NSEYSWTGATKYHLNGRDSLVNPGPAMASHKDDEEKFFPQSGVLIFGKQGSEKTNVDIEKVMITDEEEIRTTNPVATEQYGSVSTNLQRGNR'
    # See AAV README
    mutation_range = list(range(len(wt)))

num_inits = 100
total_mutants = 10_000
init_points = pd.read_csv(f'tasks/{args.task}/sft.csv')
init_points = [s[0] for s in init_points[['mutated_sequence']].values.tolist()[:num_inits]]

AA_vocab = "ACDEFGHIKLMNPQRSTVWY"
tokenizer = PreTrainedTokenizerFast(tokenizer_file="tranception/utils/tokenizers/Basic_tokenizer",
    unk_token="[UNK]", sep_token="[SEP]", pad_token="[PAD]", cls_token="[CLS]", mask_token="[MASK]"
)

def create_all_single_mutants(sequence,AA_vocab,mutation_range):
    all_single_mutants={}
    sequence_list=list(sequence)
    for position in mutation_range:
        current_AA = sequence[position]
        for mutated_AA in AA_vocab:
            if current_AA!=mutated_AA:
                mutated_sequence = sequence_list.copy()
                mutated_sequence[position] = mutated_AA
                all_single_mutants[current_AA+str(position+1)+mutated_AA]="".join(mutated_sequence)
    all_single_mutants = pd.DataFrame.from_dict(all_single_mutants,columns=['mutated_sequence'],orient='index')
    all_single_mutants.reset_index(inplace=True)
    all_single_mutants.columns = ['mutant','mutated_sequence']
    return all_single_mutants

def suggest_mutations(scores,n_mutants):
  top_mutants=list(scores.sort_values(by=['avg_score'],ascending=False).head(n_mutants).mutant)
  top_mutants_fitness=list(scores.sort_values(by=['avg_score'],ascending=False).head(n_mutants).avg_score)
  return top_mutants, top_mutants_fitness

def get_mutated_protein(sequence,mutant):
  mutated_sequence = list(sequence)
  mutated_sequence[int(mutant[1:-1])-1]=mutant[-1]
  return ''.join(mutated_sequence)

model = tranception.model_base.TranceptionLMHeadModel.from_pretrained(pretrained_model_name_or_path="ckpt/Tranception_Small")
if args.ckpt is not None:
    model.load_state_dict(torch.load(args.ckpt, map_location='cpu'))
model.config.tokenizer = tokenizer
model = model.to(args.device)
  
result_mutants = []

with tqdm(total=total_mutants) as pbar:
    for iteration in range(10):
        points_pool = init_points if len(result_mutants) == 0 else \
            [a[0] for a in sorted(result_mutants, key=lambda a:a[1], reverse=True)[:100]]
        for point in points_pool:
            all_single_mutants = create_all_single_mutants(point,AA_vocab,mutation_range)
            scores = model.score_mutants(DMS_data=all_single_mutants, 
                                                target_seq=point, 
                                                scoring_mirror=True, 
                                                batch_size_inference=128,  
                                                num_workers=4, 
                                                indel_mode=False
                                                )
            scores = pd.merge(scores,all_single_mutants,on="mutated_sequence",how="left")
            scores["position"]=scores["mutant"].map(lambda x: int(x[1:-1]))
            scores["target_AA"] = scores["mutant"].map(lambda x: x[-1])
            num_mutants = 10
            top_mutants, top_mutants_fitness = suggest_mutations(scores, num_mutants)
            for top_mutant, pred_fitness in zip(top_mutants, top_mutants_fitness):
                result_mutants.append([get_mutated_protein(point, top_mutant), pred_fitness])
            pbar.update(num_mutants)
        tqdm.write(f'{iteration}/10 | Top100 {np.mean(sorted([a[1] for a in result_mutants], reverse=True)[:100])}')

result_mutants = pd.DataFrame(result_mutants, columns=['sequence', 'pred_fitness'])
result_mutants = result_mutants.sort_values(by='pred_fitness', ascending=False, ignore_index=True)
result_mutants[:1000].to_csv(f'results_{args.task}.csv', index=False)
