# AAV 
## Data 
Download `low_vs_high.fasta` from [FLIP Benchmark](https://github.com/J-SNACKKB/FLIP).

## Sequence truncation
Wild-type sequence is length of 735 but mutations are made on indices between 561 and 588 in DMS dataset. 
Fine-tuning and extrapolation is only conducted in indices between 496 and 588 (includes context length 64)