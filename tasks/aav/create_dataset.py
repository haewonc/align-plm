from Bio import SeqIO
import pandas as pd 
import random 

def aggregate_sequences(fasta_file):
    sets_data = {}
    
    for record in SeqIO.parse(fasta_file, "fasta"):
        description_parts = record.description.split()
        target_value = None
        set_value = None
        
        for part in description_parts:
            if "TARGET=" in part:
                target_value = float(part.split("=")[1])
            elif "SET=" in part:
                set_value = part.split("=")[1]
        
        if target_value is not None and set_value is not None:
            if set_value == 'nan':
                continue
            if set_value not in sets_data:
                sets_data[set_value] = []
            sets_data[set_value].append((str(record.seq)[496:588], target_value))
    
    return sets_data

fasta_file_path = "low_vs_high.fasta"
aggregated_data = aggregate_sequences(fasta_file_path)

df = pd.DataFrame(aggregated_data['train'], columns=['mutated_sequence', 'DMS_score'])
val_df = pd.DataFrame(aggregated_data['test'], columns=['mutated_sequence', 'DMS_score'])

sorted_df = df.sort_values(by='DMS_score', ascending=False, ignore_index=True)
sfts = sorted_df.iloc[:int(0.2 * len(df))] # top 20% for SFT

df.to_csv('dms_train.csv',index=False)
sfts.to_csv('sft.csv', index=False)
val_df.to_csv('dms_val.csv',index=False)