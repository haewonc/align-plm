from core.align import align
from core.sft import sft
from core.get_log_ps import get_log_ps
import argparse 
import pandas as pd
from tqdm import tqdm 
import tranception
import torch 
import json
from core.utils import get_inference_batch_size
from transformers import PreTrainedTokenizerFast
import os
import random
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = 'true'

parser = argparse.ArgumentParser()
parser.add_argument('--task', choices=['ace2', 'aav'])
parser.add_argument('--device')
parser.add_argument('--rdevice', help='device to load reference policy. can be same as device')
args = parser.parse_args()

tran_size = 'Tranception_Small'
if args.task == 'ace2':
    target_seq = 'STIEEQAKTFLDKFNHEAEDLFYQSSLASWNYNTNITEENVQNMNNAGDKWSAFLKEQSTLAQMYPLQEIQNLTVKLQLQALQ'
elif args.task == 'aav':
    target_seq = 'NSEYSWTGATKYHLNGRDSLVNPGPAMASHKDDEEKFFPQSGVLIFGKQGSEKTNVDIEKVMITDEEEIRTTNPVATEQYGSVSTNLQRGNR'

print('SFT')
best_model, base, best_sft = sft(tran_size=tran_size, 
                                            DMS_id=args.task,
                                            preference='tasks', 
                                            exp_name=f'{args.task}_sft',
                                            total_epochs=8,
                                            target_seq=target_seq,
                                            learning_rate=2e-5,
                                            device=args.device,
                                            log='None',
                                            quiet=False,
                                            save_model=True)

print('Finding d_ref')
checkpoint = f'ckpt/{tran_size}'
tokenizer = PreTrainedTokenizerFast(tokenizer_file=f"tranception/utils/tokenizers/Basic_tokenizer",
    unk_token="[UNK]", sep_token="[SEP]", pad_token="[PAD]", cls_token="[CLS]", mask_token="[MASK]"
)
config = json.load(open(checkpoint+os.sep+'config.json'))
config = tranception.config.TranceptionConfig(**config)
config.tokenizer = tokenizer
config.vocab_size = tokenizer.vocab_size
config.retrieval_aggregation_mode = None

model = tranception.model_base.TranceptionLMHeadModel.from_pretrained(pretrained_model_name_or_path=checkpoint,config=config,ignore_mismatched_sizes=True)
model.load_state_dict(torch.load(sft, map_location='cpu'))
model = model.to(args.device)   
model.config.context_s, model.config.context_e, truncated = 0, len(target_seq), False
    
df = pd.read_csv(f'tasks/{args.task}/dms_train.csv')
sorted_df = df.sort_values(by='DMS_score', ascending=False, ignore_index=True)
train = [s[0] for s in sorted_df[['mutated_sequence']].values.tolist()]
train_size = len(train)
pairs = []
thresholds = range(16, 272, 16)
d_ref = 16
for threshold in tqdm(thresholds, desc=args.task):
    for i in range(train_size-threshold):
        pairs.append([train[i], train[i+threshold]])
    pairs = pd.DataFrame(pairs, columns=['preferred', 'dispreferred'])
    chosen, rejected = get_log_ps(model=model,
                                  device=args.device,
                                  batch_size=get_inference_batch_size(len(target_seq), tran_size), 
                                  split=pairs)
    if sum(chosen > rejected)/chosen.shape[0] > 0.65:
        d_ref = threshold
        break

print('Generate pairs')
pairs = []
for i in range(train_size-d_ref):
    for j in range(i+d_ref, min(i+2*d_ref, train_size)):
        pairs.append([train[i], train[j]])
random.shuffle(pairs)
if len(pairs) > 5000:
    pairs = pairs[:5000]
pairs = pd.DataFrame(pairs, columns=['preferred', 'dispreferred'])
pairs.to_csv(os.path.join(os.path.join(f'tasks/{args.task}/d_ref.csv')), index=False)

print('Align')
best, _, test_score, _ = align(tran_size=tran_size,
                                            DMS_id=args.task,
                                            tran_name=best_model,
                                            exp_name=f'{args.task}_align',
                                            preference='tasks', 
                                            total_epochs=8,
                                            target_seq=target_seq,
                                            device=args.device,
                                            rdevice=args.rdevice,
                                            split='d_ref',
                                            beta=0.4,
                                            learning_rate=2e-5,
                                            log='None',
                                            quiet=False,
                                            save_model=True, 
                                            save_score=False)
print(base, best_sft, test_score, best)