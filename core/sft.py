import torch 
import tranception
import json 
import os
import pandas as pd 
from transformers import DataCollatorForLanguageModeling, PreTrainedTokenizerFast
from datasets import Dataset
from torch.optim import AdamW
from core.evaluation import _get_batch_logps
from core.utils import *
from scipy.stats import spearmanr
from core.utils import get_context, get_batch_size
import time
import json

def sft(tran_size, DMS_id, preference, exp_name, wandb_name=None, run_name=None, device='cuda:0',
              log='wandb', quiet=True, total_epochs=4, target_seq=None,
              learning_rate=1e-5, indel_mode=False, save_model=False):
    checkpoint = f'ckpt/{tran_size}'
    if run_name == None:
        run_name = DMS_id
    best_name = None
    os.makedirs(f'saved/{exp_name}/{run_name}/', exist_ok=True)
    if wandb_name == None:
        wandb_name = exp_name

    if target_seq == None:
        mapping_protein_seq_DMS = pd.read_csv("reference/reference_main.csv")
        target_seq = mapping_protein_seq_DMS["target_seq"][mapping_protein_seq_DMS["DMS_id"]==DMS_id].values[0].upper()
        mutated_region = mapping_protein_seq_DMS["region_mutated"][mapping_protein_seq_DMS["DMS_id"]==DMS_id].values[0]
        mutated_s, mutated_e = mutated_region.split('-')
        mutated_s, mutated_e = int(mutated_s)-1, int(mutated_e)
        mutated_s, mutated_e, truncated = get_context(mutated_s, mutated_e, len(target_seq))
    else:
        mutated_s, mutated_e = 0, len(target_seq)
    DMS_data = pd.read_csv(f'{preference}/{DMS_id}/dms_val.csv')

    if truncated:
        print(toRed(f"Truncated: {len(target_seq)} -> {mutated_s},{mutated_e}"))

    length = mutated_e - mutated_s
    batch_size = get_batch_size(length, tran_size)
    inference_batch_size = get_inference_batch_size(length, tran_size)

    tokenizer = PreTrainedTokenizerFast(tokenizer_file=f"tranception/utils/tokenizers/Basic_tokenizer",
        unk_token="[UNK]", sep_token="[SEP]", pad_token="[PAD]", cls_token="[CLS]", mask_token="[MASK]"
    )

    config = json.load(open(checkpoint+os.sep+'config.json'))
    config = tranception.config.TranceptionConfig(**config)
    config.tokenizer = tokenizer
    config.vocab_size = tokenizer.vocab_size
    config.retrieval_aggregation_mode = None
    config.context_s, config.context_e = mutated_s, mutated_e

    model = tranception.model_base.TranceptionLMHeadModel.from_pretrained(pretrained_model_name_or_path=checkpoint,config=config,ignore_mismatched_sizes=True)
    model = model.to(device)
    optimizer = AdamW(model.parameters(), lr=learning_rate)

    sequence_collator = DataCollatorForLanguageModeling(tokenizer=model.config.tokenizer, mlm=False)
    
    sft_data = pd.read_csv(f'{preference}/{DMS_id}/sft.csv')
    sft_data = sft_data.rename(columns={'mutated_sequence': 'msa_sequence'})
    dataset = Dataset.from_pandas(sft_data)
    dataset = dataset.train_test_split(test_size=0.2)
    train_dataset, test_dataset = dataset['train'], dataset['test']

    train_dataset.set_transform(model.encode_batch_sft)
    test_dataset.set_transform(model.encode_batch_sft)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, collate_fn=sequence_collator, num_workers=2, pin_memory=True, drop_last=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, collate_fn=sequence_collator, num_workers=2, pin_memory=True, drop_last=False)

    if log == 'wandb':
        import wandb
        wandb.init(project=f"DPO_{wandb_name}", name="[SFT] " + run_name)
    if not quiet:
        print(toYellow("Scoring model"))
    all_scores = model.score_mutants(DMS_data=DMS_data, target_seq=target_seq, batch_size_inference=inference_batch_size, scoring_mirror=True, num_workers=4, indel_mode=indel_mode)
    base_all_score = spearmanr(DMS_data['DMS_score'].to_numpy(), all_scores['avg_score'].to_numpy()).correlation
    best_all_score = -1
    if not quiet:
        print(toGreen(f'Base score: all {base_all_score}'))
        print(toYellow(f"Training start: {DMS_id}\n"))
    if log == 'wandb':
        wandb.log({
            "base/all_rho": base_all_score
        })

    def run_epoch(phase, loader):
        assert phase in ['train', 'test']
        start_time = time.time()
        total_loss = 0.0
        if phase == 'test':
            model.eval()
        else:
            model.train()
        for i, data in enumerate(loader):
            if phase == 'train':
                optimizer.zero_grad()

            data = {key: tensor.to(device) for key, tensor in data.items()}
            if phase == 'train':
                outs = model(**data,return_dict=True)
            else:
                with torch.no_grad():
                    outs = model(**data,return_dict=True)

            losses = -_get_batch_logps(outs.logits, data['input_ids'])
            loss = losses.mean()

            if phase == 'train':
                loss.backward()
                optimizer.step()

            elapsed_time = time.time() - start_time
            total_loss += loss.item()

            if log == 'wandb':
                wandb.log({
                    f"{phase}/sft_loss": loss.item()
                })
            if not quiet:
                print_progress(phase.upper(), epoch, total_epochs, i, len(loader), elapsed_time, ["SFT"], [loss.item()])
        if not quiet:
            print_total(phase.upper(), epoch, total_epochs, ["SFT"], [total_loss / len(loader)])

    for epoch in range(total_epochs):
        run_epoch("train", train_loader)
        run_epoch("test", test_loader)
        if not quiet:
            print(toYellow("Scoring reference model"))
        all_scores = model.score_mutants(DMS_data=DMS_data, target_seq=target_seq, batch_size_inference=inference_batch_size, scoring_mirror=True, num_workers=4, indel_mode=indel_mode)
        new_all_score = spearmanr(DMS_data['DMS_score'].to_numpy(), all_scores['avg_score'].to_numpy()).correlation
        if log == 'wandb':
            wandb.log({
                "test/all_rho": new_all_score
            })
        if not quiet:
            print(toGreen(f'Trained score: all {new_all_score}'),end='\n\n')
        if new_all_score > best_all_score:
            best_name = f'saved/{exp_name}/{run_name}/sft_best.pt'
            if save_model:
                torch.save(model.state_dict(), f'saved/{exp_name}/{run_name}/sft_best.pt')
            best_all_score = new_all_score
    if log == 'wandb':
        wandb.finish()
    return best_name, base_all_score, best_all_score
