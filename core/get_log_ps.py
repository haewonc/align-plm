import torch 
import os
import numpy as np
from transformers import DataCollatorForLanguageModeling
from datasets import Dataset
from core.evaluation import _get_batch_logps
from core.utils.log_utils import *
from tqdm import tqdm 
import json
import os
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = 'true'

def get_log_ps(device, model, split=None, batch_size=4):
    sequence_collator = DataCollatorForLanguageModeling(tokenizer=model.config.tokenizer, mlm=False)

    def encoding_to_dict(encoding):
        return {
            'input_ids': encoding.ids,
            'token_type_ids': encoding.type_ids,
            'attention_mask': encoding.attention_mask,
        }
    
    def preference_collator(batch):
        p, d = [encoding_to_dict(b['preferred']) for b in batch], [encoding_to_dict(b['dispreferred']) for b in batch]
        return sequence_collator(p), sequence_collator(d)
    
    train_dataset = Dataset.from_pandas(split)
    train_dataset.set_transform(model.encode_batch_pair)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, collate_fn=preference_collator, num_workers=2, pin_memory=True, drop_last=False)

    model.eval()
    policy_chosen_logps = []
    policy_rejected_logps = []
    for i, data in enumerate(tqdm(train_loader)):
        p, d = data

        p = {key: tensor.to(device) for key, tensor in p.items()}
        d = {key: tensor.to(device) for key, tensor in d.items()}
        with torch.no_grad():
            pouts = model(**p,return_dict=True)
            douts = model(**d,return_dict=True)

            policy_chosen_logps.extend(list(_get_batch_logps(pouts.logits, p['input_ids']).cpu().numpy()))
            policy_rejected_logps.extend(list(_get_batch_logps(douts.logits, d['input_ids']).cpu().numpy()))
    
    return np.array(policy_chosen_logps), np.array(policy_rejected_logps)
