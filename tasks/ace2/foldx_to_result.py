import numpy as np 
from scipy.stats import spearmanr
import pandas as pd
import sys

flag = False
lines = []
with open(f'foldx5/{sys.argv[1]}/Dif_s.fxout', 'r') as file:
    for line in file:
        if 'Pdb	total energy' in line:
            flag = True 
            continue
        if flag: 
            lines.append(float(line.split()[1]))
total = len(lines)
print(total)
print(len([l for l in lines if l < -2.5])/total)
print(len([l for l in lines if l < -5])/total)
print(len([l for l in lines if l < -6])/total)
print(len([l for l in lines if l < -7])/total)
