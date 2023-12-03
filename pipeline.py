from core.align import align
from core.sft import sft
from core.get_log_ps import get_log_ps
import argparse 
import pandas as pd
from tqdm import tqdm 
import tranception
import torch 
import json
from core.utils import get_inference_batch_size, get_context
from transformers import PreTrainedTokenizerFast
import os
import random
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = 'true'

parser = argparse.ArgumentParser()
parser.add_argument('--preference', default='preference_500', help='Root directory to pair data')
parser.add_argument('--DMS_id')
parser.add_argument('--size', default='Large', choices=['Large', 'Small'])
parser.add_argument('--device', default='cuda:0')
parser.add_argument('--rdevice', help='device to load reference policy. can be same as device', default='cuda:0')
args = parser.parse_args()

tran_size = f'Tranception_{args.size}'
DMS_id = args.DMS_id
reference = pd.read_csv('reference/reference_main.csv')

print('SFT')
best_model, base, best_sft = sft(tran_size=tran_size, 
                                            DMS_id=DMS_id,
                                            preference=args.preference, 
                                            exp_name=f'{DMS_id}_sft',
                                            total_epochs=8,
                                            learning_rate=2e-5,
                                            device=args.device,
                                            log='None',
                                            quiet = False,
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
model.load_state_dict(torch.load(best_model, map_location='cpu'))
model = model.to(args.device)   
target_seq = reference["target_seq"][reference["DMS_id"]==DMS_id].values[0].upper()
mutated_region = reference["region_mutated"][reference["DMS_id"]==DMS_id].values[0]
mutated_s, mutated_e = mutated_region.split('-')   
model.config.context_s, model.config.context_e, truncated = get_context(int(mutated_s)-1, int(mutated_e), len(target_seq))
        
df = pd.read_csv(f'{args.preference}/{DMS_id}/dms_train.csv')
sorted_df = df.sort_values(by='DMS_score', ascending=False, ignore_index=True)
train = [s[0] for s in sorted_df[['mutated_sequence']].values.tolist()]
train_size = len(train)
thresholds = range(16, 272, 16)
d_ref = 16
for threshold in tqdm(thresholds):
    pairs = []
    for i in range(train_size-threshold):
        pairs.append([train[i], train[i+threshold]])
    pairs = pd.DataFrame(pairs, columns=['preferred', 'dispreferred'])
    chosen, rejected = get_log_ps(model=model,
                                batch_size=get_inference_batch_size(len(target_seq), tran_size),
                                device=args.device, 
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
pairs.to_csv(os.path.join(os.path.join(f'{args.preference}/{DMS_id}/d_ref.csv')), index=False)

print('Align')
best, test_score = align(tran_size=tran_size,
                                            DMS_id=DMS_id,
                                            tran_name=best_model,
                                            exp_name=f'{DMS_id}_align',
                                            preference=args.preference, 
                                            total_epochs=1,
                                            device=args.device,
                                            rdevice=args.rdevice,
                                            split='d_ref',
                                            beta=0.4,
                                            learning_rate=2e-5,
                                            log='None',
                                            quiet=False,
                                            save_model=True, 
                                            save_score=True)
print(base, best_sft, test_score, best)