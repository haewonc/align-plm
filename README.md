# align-plm

## Get Started
### Environment 
```
conda env create -f env.yml
conda activate align
```

### Data
Download from [Tranception](https://github.com/OATML-Markslab/Tranception)
- `MSA_weights` (download & unzip MSA_weights.zip)
- `MSA_files` (download & unzip MSA_ProteinGym.zip)
- Tranception checkpoints (Small and Large) then place under `ckpt/`

Download DMS data from [ProteinGym](https://github.com/OATML-Markslab/ProteinGym) then place under `proteingym/`

Split the dataset.
```
python split_dataset.py
```

### Resource constraints
The allocated memory depends on the length of the protein, which is very diverse. Therefore, batch size is handled by the functions in `core/utils/get_batch.py`. It is hard-coded for GPU with 24GB memory, so you may change the value  considering your resource constraints.

## Fitness Prediction in ProteinGym
### Pipeline for Single DMS assay
```
python pipeline.py --DMS_id IF1_ECOLI_Kelsic_2016 
```

### Run for all ProteinGym benchmarks
```
python 1_sft_all.py
python 2_ref_dist.py
python 3_generate_pairs.py
python 4_align_score.py
```
The scoring files will be saved under `dms-results`.
```
python performance.py --input_scoring_files_folder dms-results/$exp_name$ --performance_by_depth
```
This will evaluate the experiment in terms of correlation metrics.

## Fitness Optimization
See `tasks/`

## Reproduce the Figures
See `draw_figure.ipynb`. Note that `total.csv` contains Spearman's rho correlation of Ours, Alpha-Missense,Tranception Large, Tranception Large (no retrieval), EVE, and ESM1v in ProteinGym benchmark.

### TODO
- [ ] Support LoRA