import matplotlib.pyplot as plt
from Bio import AlignIO
from tranception.utils.msa_utils import process_msa_data

msa = 'MSA_files/F7YBW8_MESOW_full_01-07-2022_b02.a2m'

def visualize_msa_with_ids(alignments):
    max_length = max(len(seq) for seq in alignments.values())
    for i, (id, seq) in enumerate(alignments.items()):
        if i > 100:
            print(seq)
        if i> 150:
            break

def plot_msa_heatmap(alignment):
    alignment_length = alignment.get_alignment_length()
    num_sequences = len(alignment)
    
    heatmap = [[0 for _ in range(alignment_length)] for _ in range(num_sequences)]
    
    for i, record in enumerate(alignment):
        for j, letter in enumerate(record.seq):
            if letter != '-':
                heatmap[i][j] = 1
    
    plt.imshow(heatmap, cmap='hot', interpolation='none', aspect='auto')
    plt.xlabel('Position in Alignment')
    plt.ylabel('Sequences')
    plt.title('MSA Heatmap')
    plt.savefig('F7YBW8_MESOW.png')

alignment = AlignIO.read(open(msa), "fasta")
visualize_msa_with_ids(process_msa_data(msa))
# plot_msa_heatmap(alignment)