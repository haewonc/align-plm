import sys
import scipy.stats as st
import pandas as pd
import pickle5 as pickle
import random

WT = 'STIEEQAKTFLDKFNHEAEDLFYQSSLASWNYNTNITEENVQNMNNAGDKWSAFLKEQSTLAQMYPLQEIQNLTVKLQLQALQ'

filename = sys.argv[1]
output_dir = sys.argv[2]
with open('train_ddG.pkl', 'rb') as file:
    df = pickle.load(file)
ip = df.to_dict(orient='records')

c_inc, c_dec = 0, 0
random.shuffle(ip)

op = []
all_scores = []

for src_idx in range(len(ip)):
    c = 0
    src = ip[src_idx]
    if src['ddG'] < -4 or src['ddG'] > 3:
        continue

    op_item = {}
    op_item['label'] = src['ddG']
    op_item['text'] =  ' '.join(src['MT_seq'])
    # op_item['original'] = src
    op.append(op_item)
    all_scores.append(op_item['label'])

print(len(op))         
# print(len(list(set([item['translation']['src'][5:] for item in op]))))
random.shuffle(op)
train = op[:int(0.9*len(op))]
val = op[int(0.9*len(op)):]
print(len(train), len(val))
statistic, bins, binnumber = st.binned_statistic(all_scores, all_scores, statistic = 'count', bins = 50)
print("Score stats")
print([int(i) for i in statistic])
print([round(i, 3) for i in bins])
print(binnumber)

train_df = []
for item in train:
    mutant = item['text'].replace(' ','') 
    pos = [i for i in range(len(mutant)) if mutant[i]!=WT[i]]
    if len(pos) == 0:
        continue    
    train_df.append([':'.join([f'{WT[p]}{p}{mutant[p]}' for p in pos]), item['text'].replace(' ',''), -item['label']])
df = pd.DataFrame(train_df, columns=['mutant', 'mutated_sequence', 'DMS_score'])

val_df = []
for item in val:
    mutant = item['text'].replace(' ','') 
    pos = [i for i in range(len(mutant)) if mutant[i]!=WT[i]]
    if len(pos) == 0:
        continue    
    val_df.append([':'.join([f'{WT[p]}{p}{mutant[p]}' for p in pos]), item['text'].replace(' ',''), -item['label']])
val_df = pd.DataFrame(val_df, columns=['mutant', 'mutated_sequence', 'DMS_score'])

sorted_df = df.sort_values(by='DMS_score', ascending=False, ignore_index=True)
sfts = sorted_df.iloc[:int(0.2 * len(df))]

df.to_csv('dms_train.csv',index=False)
sfts.to_csv('sft.csv', index=False)
val_df.to_csv('dms_val.csv',index=False)
