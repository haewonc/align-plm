import torch 
import tranception
import json 
import os
import pandas as pd 
import argparse
from transformers import PreTrainedTokenizerFast
from core.get_log_ps import get_log_ps
from core.utils.log_utils import *
from tqdm import tqdm 
import json
from core.utils import get_inference_batch_size, get_context
import os
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = 'true'

parser = argparse.ArgumentParser()
parser.add_argument('--preference', default='preference_500', help='Root directory to pair data')
parser.add_argument('--size', default='Large', choices=['Large', 'Small'])
parser.add_argument('--device', default='cuda:0')
args = parser.parse_args()

device = args.device
reference = pd.read_csv(f'results/reference_main_sft.csv')
log = None

all_accs = []
dms_ids = []
thresholds = range(16, 272, 16)

tran_size = f'ckpt/Tranception_{args.size}'
tokenizer = PreTrainedTokenizerFast(tokenizer_file=f"tranception/utils/tokenizers/Basic_tokenizer",
    unk_token="[UNK]", sep_token="[SEP]", pad_token="[PAD]", cls_token="[CLS]", mask_token="[MASK]"
)

config = json.load(open(tran_size+os.sep+'config.json'))
config = tranception.config.TranceptionConfig(**config)
config.tokenizer = tokenizer
config.vocab_size = tokenizer.vocab_size
config.retrieval_aggregation_mode = None

model = tranception.model_base.TranceptionLMHeadModel.from_pretrained(pretrained_model_name_or_path=tran_size,config=config,ignore_mismatched_sizes=True)
dists = []

for DMS_index in range(reference.shape[0]):
    DMS_id = reference["DMS_id"][DMS_index]
    dms_ids.append(DMS_id)
    sft = reference["SFT ckpt"][DMS_index]
    model.load_state_dict(torch.load(sft, map_location='cpu'))
    model = model.to(device)   

    length = reference["seq_len"][DMS_index]
    target_seq = reference["target_seq"][reference["DMS_id"]==DMS_id].values[0].upper()
    mutated_region = reference["region_mutated"][reference["DMS_id"]==DMS_id].values[0]
    mutated_s, mutated_e = mutated_region.split('-')   
    model.config.context_s, model.config.context_e, truncated = get_context(int(mutated_s)-1, int(mutated_e), len(target_seq))
      
    for threshold in tqdm(thresholds, desc=DMS_id):
        df = pd.read_csv(f'{args.preference}/{DMS_id}/dms_train.csv')
        sorted_df = df.sort_values(by='DMS_score', ascending=False, ignore_index=True)
        train = [s[0] for s in sorted_df[['mutated_sequence']].values.tolist()]
        train_size = len(train)
        pairs = []
        if train_size < threshold + 50:
            dists.append(threshold)
            break
        for i in range(train_size-threshold):
            pairs.append([train[i], train[i+threshold]])
        pairs = pd.DataFrame(pairs, columns=['preferred', 'dispreferred'])
        chosen, rejected = get_log_ps(model=model,
                                    batch_size=get_inference_batch_size(length, tran_size), 
                                    device=device,
                                    split=pairs)
        if sum(chosen > rejected)/chosen.shape[0] > 0.65:
            dists.append(threshold)
            break

reference['distance'] = dists
reference.to_csv('results/reference_main_sft.csv', index=False)