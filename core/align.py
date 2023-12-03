import torch 
import tranception
import json 
import os
import pandas as pd 
from transformers import DataCollatorForLanguageModeling, PreTrainedTokenizerFast
from datasets import Dataset
from torch.optim import AdamW
from core.evaluation import dpo_loss, _get_batch_logps
from core.utils import *
from scipy.stats import spearmanr
import time
import json

def align(tran_size, DMS_id, exp_name, run_name=None, tran_name=None, device='cuda:0', preference='preference_30_20', target_seq=None,
              split='train', rdevice='cuda:1', beta=0.4, log='wandb', quiet=True, total_epochs=4, 
              learning_rate=1e-5, indel_mode=False, save_model=False, save_score=True):
    checkpoint = f'ckpt/{tran_size}'
    if run_name == None:
        run_name = DMS_id
    os.makedirs(f'saved/{exp_name}/{run_name}/', exist_ok=True)

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
    batch_size = get_batch_size_align(length, tran_size)
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
    reference = tranception.model_base.TranceptionLMHeadModel.from_pretrained(pretrained_model_name_or_path=checkpoint,config=config,ignore_mismatched_sizes=True)

    if tran_name is not None and os.path.isfile(tran_name):
        model.load_state_dict(torch.load(tran_name, map_location='cpu'))
        reference.load_state_dict(torch.load(tran_name, map_location='cpu'))
    else:
        print(toRed(f"\n\n{tran_name} not found, starting align from {checkpoint}"), end='\n\n')
    model = model.to(device)        
    reference = reference.to(rdevice)
    optimizer = AdamW(model.parameters(), lr=learning_rate)

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
    
    train_dataset = Dataset.from_pandas(pd.read_csv(f'{preference}/{DMS_id}/{split}.csv'))
    train_dataset.set_transform(model.encode_batch_pair)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, collate_fn=preference_collator, num_workers=2, pin_memory=True, drop_last=False)

    if log == 'wandb':
        import wandb
        wandb.init(project="Align_{exp_name}", name="[Align] "+run_name)
    
    best_score = -1
    best_name = None

    if not quiet:
        print(toYellow(f"Training start {DMS_id}"))
        print('\n')
    
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
            p, d = data
            
            pref = {key: tensor.to(rdevice) for key, tensor in p.items()}
            dref = {key: tensor.to(rdevice) for key, tensor in d.items()}
            p = {key: tensor.to(device) for key, tensor in p.items()}
            d = {key: tensor.to(device) for key, tensor in d.items()}

            if phase == 'train':
                pouts = model(**p,return_dict=True)
                douts = model(**d,return_dict=True)
            else:
                with torch.no_grad():
                    pouts = model(**p,return_dict=True)
                    douts = model(**d,return_dict=True)
            
            with torch.no_grad():
                pouts_ref = reference(**pref,return_dict=True)
                douts_ref = reference(**dref,return_dict=True)
            
            policy_chosen_logps = _get_batch_logps(pouts.logits, p['input_ids'])
            policy_rejected_logps = _get_batch_logps(douts.logits, d['input_ids'])
            reference_chosen_logps = _get_batch_logps(pouts_ref.logits, pref['input_ids']).to(device)
            reference_rejected_logps = _get_batch_logps(douts_ref.logits, dref['input_ids']).to(device)
            
            losses, chosen_rewards, rejected_rewards, bt_prob = dpo_loss(
                policy_chosen_logps, policy_rejected_logps, 
                reference_chosen_logps, reference_rejected_logps,
                beta=beta, return_bt=True)
            loss = losses.mean()

            if phase == 'train':
                loss.backward()
                optimizer.step()

            elapsed_time = time.time() - start_time
            total_loss += loss.item()

            payload = {
                f"{phase}/total_loss": loss.item(),
                f"{phase}/preferred": chosen_rewards.detach().mean().item(),
                f"{phase}/dispreferred": rejected_rewards.detach().mean().item(),
                f"{phase}/accuracy": (chosen_rewards.cpu() > rejected_rewards.cpu()).float().mean().item(),
                f"{phase}/margin": (chosen_rewards.cpu() - rejected_rewards.cpu()).mean().item(),
                f"{phase}/preferred_acc": pouts.accuracy.item(),
                f"{phase}/dispreferred_acc": douts.accuracy.item(),
                f"{phase}/bt_prob": bt_prob.mean().item(),
                f"{phase}/ref_preferred_acc": pouts_ref.accuracy.item(),
                f"{phase}/ref_dispreferred_acc": douts_ref.accuracy.item(),
            }

            if log == 'wandb':
                wandb.log(payload)
            if not quiet:
                print_progress(phase.upper(), epoch, total_epochs, i, len(loader), elapsed_time, ["Align"], [loss.item()])
        if not quiet:
            print_total(phase.upper(), epoch, total_epochs, ["Align"], [total_loss / len(loader)])

    for epoch in range(total_epochs):
        run_epoch("train", train_loader)
        if not quiet:
            print(toYellow("Scoring model"))
        
        all_scores = model.score_mutants(DMS_data=DMS_data, batch_size_inference=inference_batch_size, target_seq=target_seq, scoring_mirror=True, num_workers=4, indel_mode=indel_mode)
        new_score = spearmanr(DMS_data['DMS_score'].to_numpy(), all_scores['avg_score'].to_numpy()).correlation

        if log == 'wandb':
            wandb.log({
                "test/epoch": epoch+1,
                "test/test_rho": new_score,
            })

        if not quiet:
            print(toGreen(f'Trained score: test {new_score}'), end='\n\n')
        if new_score > best_score:
            if save_model:
                torch.save(model.state_dict(), f'saved/{exp_name}/{run_name}/model_epoch{epoch}.pt')
            best_score = new_score
            best_name = f'saved/{exp_name}/{run_name}/model_epoch{epoch}.pt'
        
        if save_score:
            DMS_test_data = pd.read_csv(f'{preference}/{DMS_id}/dms_test.csv', low_memory=False)
            all_scores = model.score_mutants(
                                            DMS_data=DMS_test_data, 
                                            target_seq=target_seq, 
                                            scoring_mirror=True,
                                            batch_size_inference=get_inference_batch_size(length, tran_size),
                                            num_workers=4,
                                            indel_mode=False
                                            )
            scoring_filename = 'dms-results' + os.sep + exp_name
            if not os.path.isdir(scoring_filename):
                os.mkdir(scoring_filename)
            all_scores.to_csv(scoring_filename + os.sep + DMS_id + '.csv', index=False)
    if log == 'wandb':
        wandb.finish()
    return best_name, best_score