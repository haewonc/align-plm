import pandas as pd
import os
from tqdm import tqdm
import random
from sklearn.model_selection import train_test_split

def random_indices(n):
    return random.choices(range(n), k=2)

out_dir = 'preference_500'
sft_top = 0.2
max_size = 500
reference = pd.read_csv(f'reference/reference_main.csv')
normalize = True

if not os.path.exists(out_dir):
    os.makedirs(out_dir)

print('Start processing...')

for dms_csv in tqdm(reference['DMS_filename'], total=85):
    dms = pd.read_csv(f'proteingym/{dms_csv}')
    df, dms_test = train_test_split(dms, train_size=min(int(dms.shape[0]*0.2), max_size))
    dms_val, dms_test =  train_test_split(dms_test, train_size=min(int(dms.shape[0]*0.2), 500))

    out_csv = f"{out_dir}/{dms_csv.replace('.csv','')}"
    os.makedirs(out_csv)
    df.to_csv(f'{out_csv}/dms_train.csv', index=False)

    if normalize:
        df['DMS_score'] -= df['DMS_score'].min()
        df['DMS_score'] /= df['DMS_score'].max()

    sorted_df = df.sort_values(by='DMS_score', ascending=False, ignore_index=True)
    sfts = sorted_df.iloc[:int(sft_top * len(df))]

    dms_val.to_csv(f'{out_csv}/dms_val.csv', index=False)
    dms_test.to_csv(f'{out_csv}/dms_test.csv', index=False)
    sfts.to_csv(f'{out_csv}/sft.csv', index=False)
    

    '''
        Ablation for pair construction
    '''

    # train = [s[0] for s in sorted_df[['mutated_sequence']].values.tolist()]
    # scores = [s[0] for s in sorted_df[['DMS_score']].values.tolist()]
    # train_size = len(train)
    # threshold = len(train) // 10

    ## adjacent
    # adjacent = []
    # for i in range(train_size-1):
    #     adjacent.append([train[i], train[i+1]])
    # adjacent = pd.DataFrame(adjacent, columns=['preferred', 'dispreferred'])

    ## N-distance
    # ndist = []
    # for i in range(train_size-threshold):
    #     ndist.append([train[i], train[i+threshold]])
    # ndist = pd.DataFrame(ndist, columns=['preferred', 'dispreferred'])

    ## sliding
    # sliding = []
    # for i in range(train_size-2*threshold):
    #     for j in range(0, threshold, 2):
    #         sliding.append([train[i], train[i+threshold+j], scores[i]-scores[i+threshold+j]])
    # random.shuffle(sliding)
    # sliding = pd.DataFrame(sliding, columns=['preferred', 'dispreferred', 'difference'])

    ## random
    # num_pairs = len(sliding)
    # randoms = []
    # for _ in range(num_pairs):
    #     i,j = sorted(random_indices(train_size))
    #     randoms.append([train[i], train[j]])
    # randoms = pd.DataFrame(randoms, columns=['preferred', 'dispreferred'])

    ## binning
    # def divide_into_bins(lst, bins):
    #     bin_size = len(lst) // bins
    #     divided = [lst[i:i + bin_size] for i in range(0, len(lst), bin_size)]
    #     if len(divided) > bins:
    #         divided[-2] = divided[-2] + divided[-1]
    #         divided.pop()
    #     return divided
    # bins = divide_into_bins(train, 20)
    # pairs = []
    # for i in range(len(bins) - 1):
    #     for elem in bins[i]:
    #         for next_elem in bins[i+1]:
    #             pairs.append([elem, next_elem])
    # random.shuffle(pairs)
    # bins = pd.DataFrame(pairs, columns=['preferred', 'dispreferred'])  

    ## all
    # alls = []
    # for i in range(len(train)):
    #     for j in range(i+1, len(train)):
    #         alls.append([train[i], train[j]])
    # alls = pd.DataFrame(alls, columns=['preferred', 'dispreferred'])  

    # bins.to_csv(f'{out_csv}/binning.csv', index=False)
    # sliding.to_csv(f'{out_csv}/sliding_diff.csv', index=False)
    # adjacent.to_csv(f'{out_csv}/adjacent.csv', index=False)
    # ndist.to_csv(f'{out_csv}/ndist.csv', index=False)
    # randoms.to_csv(f'{out_csv}/random.csv', index=False)
    # alls.to_csv(f'{out_csv}/all.csv', index=False)