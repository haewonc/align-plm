import torch 
import tranception
import json 
import os
import pandas as pd 
from transformers import PreTrainedTokenizerFast, DataCollatorForLanguageModeling
import numpy as np
import matplotlib.pyplot as plt
from core.utils.log_utils import *
import json
from matplotlib.colors import ListedColormap

colors1 = plt.cm.tab20b(np.linspace(0., 1, 20))
colors2 = plt.cm.tab20c(np.linspace(0., 1, 20))
colors = np.vstack((colors1, colors2))
mymap = ListedColormap(colors, name='myColormap', N=colors.shape[0])

checkpoint = 'saved/test/A0A2Z5U3Z0_9INFA_Wu_2014/sft_epoch0.pt'
device = 'cuda:1'
vocab = {"[UNK]":0,"[CLS]":1,"[SEP]":2,"[PAD]":3,"[MASK]":4,"A":5,"C":6,"D":7,"E":8,"F":9,"G":10,"H":11,"I":12,"K":13,"L":14,"M":15,"N":16,"P":17,"Q":18,"R":19,"S":20,"T":21,"V":22,"W":23,"Y":24}
idx_to_aa = {vocab[k]:k for k in vocab}

mapping_protein_seq_DMS = pd.read_csv("proteingym/reference.csv")
DMS_list = mapping_protein_seq_DMS["DMS_id"]
MAX_LEN = 64

tokenizer = PreTrainedTokenizerFast(tokenizer_file=f"tranception/utils/tokenizers/Basic_tokenizer",
    unk_token="[UNK]", sep_token="[SEP]", pad_token="[PAD]", cls_token="[CLS]", mask_token="[MASK]"
)

config = json.load(open(checkpoint+os.sep+'config.json'))
config = tranception.config.TranceptionC책onfig(**config)
config.tokenizer = tokenizer
config.vocab_size = tokenizer.vocab_size
config.retrieval_aggregation_mode = None

model = tranception.model_base.TranceptionLMHeadModel.from_pretrained(pretrained_model_name_or_path='ckpt/Tranception_Small',config=config)
model.load_state_dict(torch.load(checkpoint, map_location='cpu'))
model = model.to(device)

sequence_collator = DataCollatorForLanguageModeling(tokenizer=model.config.tokenizer, mlm=False)

DMS_id = 'A0A2Z5U3Z0_9INFA_Wu_2014'
target_seq = mapping_protein_seq_DMS["target_seq"][mapping_protein_seq_DMS["DMS_id"]==DMS_id].values[0].upper()
encoded_target = model.encode_batch(pd.DataFrame([[target_seq]], columns=["sliced_mutated_sequence"]))
encoded_target = {key: tensor.to(device) for key, tensor in sequence_collator([encoded_target]).items()}
with torch.no_grad():
    lm_logits = model(**encoded_target,return_dict=True).logits
    lm_logits = lm_logits.detach().cpu()[0,0]
lm_logits = lm_logits.softmax(dim=-1).numpy()

lm_logits = lm_logits[:min(lm_logits.shape[0], MAX_LEN),:]
L = lm_logits.shape[0]

fig = plt.figure(figsize=(8,4))
ax = plt.subplot(111)

ind = np.arange(L)
class_data = list(lm_logits.T)
bottom = np.zeros(L)

for i, class_datum in enumerate(class_data):
    ax.bar(ind, class_datum.flatten(), bottom=bottom, color=mymap(i))
    bottom += class_datum.flatten()

plt.xlabel('Position')
plt.ylabel('AA Type')

plt.xticks(ind)
plt.yticks(np.arange(0, np.max(bottom), 10))

ax.set_position([0.1, 0.3, 0.8, 0.6]) 
ax.get_xaxis().set_visible(False)
ax.legend([f"{idx_to_aa[i]}" for i in range(25)], fontsize="9", loc='upper center', ncol=9, bbox_to_anchor=(0.5,-0.1))

plt.savefig(f'figure/logits_sft/{DMS_id}.png',dpi=150)
plt.clf()
plt.close()