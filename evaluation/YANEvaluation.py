# evaluation/yan/YANEvaluation.py 

# ------------------------------------------------------------------------------------------------ 
# This is YAN evaluation scripts.
# Running the following in terminal in 'evaluation/' will save 3 CSV fils in 'RESULTS/' 
# corresponding to three tasks: text infilling, question answering, and last-word completion. 
# > python -m yan.YANEvaluation
# ------------------------------------------------------------------------------------------------ 


import os
os.chdir('/home/evaluation')
import warnings
warnings.filterwarnings("ignore", message="IProgress not found.*")
warnings.filterwarnings("ignore", message=".*ipywidgets.*")
warnings.filterwarnings("ignore", message="`clean_up_tokenization_spaces` was not set.*")

import time
import torch
import pandas as pd
from tqdm import tqdm
from functions import *
from yan.YANModel import YAN
from torch.nn.utils.rnn import pad_sequence

set_seed(30)
yan_tok = load_yan_tokenizer()
dp_config = get_dp_config()
evaluator = Evaluator()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = 'yan'


def evaluate_yan(
    task, dataset, 
    device: torch.device, 
    dp_config: Dict[str, Any], 
    yan_tokenizer, loader,
    n_timesteps: int = 4, 
    use_oracle_tgt_len: bool = True, 
    max_tgt_length: int | None = None,
    ckpt_dir: str = '../final_ckpts'
):
    assert task in ['last', 'qa', 'infill']
    # ----- Load checkpoint ----- 
    ckpt_path = os.path.join(ckpt_dir, f'{model_name}_{dataset}.pt')
    ckpt = torch.load(ckpt_path, map_location=device)
    model = YAN(ckpt['model_config'])
    model.load_state_dict(ckpt['model'])
    model.eval()
    model.to(device)

    # ----- Get task-specific data processor ----- 
    dp_config.update({'split': task, 'max_length': 2048})
    dp = DataProcessor(yan_tokenizer, dp_config, device, yan_tokenizer)
    if dataset in ['simplestories', 'rocstories']:
        info = 'Last-Word Completion'
        data_process_fcn = dp.split_enc_tgt_last
    elif dataset in ['squad', 'babiqa']:
        info = 'Question Answering'
        dp_config = {'mask_ratio': 0., 'remove_answer_prob': 1.0, 'corrupt_span_prob': 0.0,
                     'max_span_len': 1, 'other_mask_token': None, 'max_length': 2048, 'split': 'infill'}
        dp = DataProcessor(yan_tokenizer, dp_config, device, yan_tokenizer)
        data_process_fcn = dp.generate_span_corruption
    elif dataset in ['narrativeqa']:
        info = 'Text Infilling'
        data_process_fcn = dp.generate_span_corruption
    elif dataset in ['sst2', 'dbpedia', 'agnews']:
        info = 'Classification'
        data_process_fcn = dp.split_enc_tgt_qa
    else:
        info, data_process_fcn = '', None
        print("dataset input wrong!")
    assert data_process_fcn is not None
    
    predictions, references = [], []
    n_gen_tokens, n_gen_times = [], []
    total_gen_tokens, total_gen_time = 0, 0.0
    
    for loader_batch in tqdm(loader, desc=f"Evaluating YAN on [{info}][{dataset}]"):
        # ----- Get batch data ----- 
        batch = data_process_fcn(loader_batch)
        batch = {key: tensor.to(device, non_blocking=True) for key, tensor in batch.items()}
        if dataset in ['squad', 'babiqa']:
            enc_ids, enc_att_mask, tgt_ids, tgt_att_mask = open_batch_by_task('infill', batch)
        else:
            enc_ids, enc_att_mask, tgt_ids, tgt_att_mask = open_batch_by_task(task, batch)

        # ----- Decide maximum generation length ----- 
        if use_oracle_tgt_len:
            max_tgt_length = tgt_ids.shape[1]
        else:
            assert max_tgt_length is not None
        
        # ----- Start generation and timing ----- 
        with torch.no_grad():
            if device.type == "cuda": torch.cuda.synchronize()
            t0 = time.perf_counter()

            latent, logits = model.generate(enc_ids, max_tgt_length, enc_att_mask, 
                                            n_timesteps=n_timesteps, trajectory=False)

            if device.type == "cuda": torch.cuda.synchronize()
            dt = time.perf_counter() - t0
            total_gen_time += dt
        
        # ----- Generated token ids ----- 
        gen_ids = logits.argmax(-1)

        # --- special format ---
        if dataset in ['squad', 'babiqa']:
            mask_idx = (enc_ids == yan_tokenizer.mask_token_id).long().argmax(dim=1) - 5
            tgt_suffix = [tgt_ids[b, mask_idx[b]:] for b in range(tgt_ids.shape[0])]
            tgt_ids_cleaned = pad_sequence(tgt_suffix, batch_first=True, 
                                           padding_value=yan_tokenizer.eos_token_id)
            gen_suffix = [gen_ids[b, mask_idx[b]:] for b in range(gen_ids.shape[0])]
            gen_ids_cleaned = pad_sequence(gen_suffix, batch_first=True, 
                                           padding_value=yan_tokenizer.eos_token_id)
            # ----- Number of generated tokens (before and including EOS) -----
            gen_tokens = count_tokens_until_eos(gen_ids_cleaned, yan_tokenizer.eos_token_id)    # (B,)
            total_gen_tokens += gen_tokens.sum().item()
            n_gen_tokens.extend(gen_tokens.tolist())
            n_gen_times.extend([dt]*len(gen_tokens))
            # ----- Decoded texts -----
            batch_pred = yan_tokenizer.batch_decode(gen_ids_cleaned, skip_special_tokens=False)
            batch_pred = clean_texts(batch_pred, yan_tokenizer.bos_token, yan_tokenizer.eos_token)
            batch_ref = yan_tokenizer.batch_decode(tgt_ids_cleaned, skip_special_tokens=False)
            batch_ref = clean_texts(batch_ref, yan_tokenizer.bos_token, yan_tokenizer.eos_token)
        else:
            # ----- Number of generated tokens (before and including EOS) -----
            gen_tokens = count_tokens_until_eos(gen_ids, yan_tokenizer.eos_token_id)    # (B,)
            total_gen_tokens += gen_tokens.sum().item()
            n_gen_tokens.extend(gen_tokens.tolist())
            n_gen_times.extend([dt]*len(gen_tokens))
            # ----- Decoded texts -----
            batch_pred = yan_tokenizer.batch_decode(gen_ids, skip_special_tokens=False)
            batch_pred = clean_texts(batch_pred, yan_tokenizer.bos_token, yan_tokenizer.eos_token)
            batch_ref = yan_tokenizer.batch_decode(tgt_ids, skip_special_tokens=False)
            batch_ref = clean_texts(batch_ref, yan_tokenizer.bos_token, yan_tokenizer.eos_token)
        
        predictions.extend(batch_pred)
        references.extend(batch_ref)
    
    # ----- Evaluate generated texts -----
    evaluation_metrics_dict = evaluator.evaluate_all_metrics(predictions, references)

    evaluation_metrics_dict["TPS"] = total_gen_tokens / max(total_gen_time, 1e-8)
    evaluation_metrics_dict["total_gen_tokens"] = total_gen_tokens
    evaluation_metrics_dict["total_gen_time"] = float(total_gen_time)

    # ----- Combine into one table -----
    N = len(n_gen_tokens)
    evaluation_metrics_df = pd.DataFrame([evaluation_metrics_dict] * N)
    evaluation_metrics_df["n_gen_tokens"] = n_gen_tokens
    evaluation_metrics_df["n_gen_times"] = n_gen_times

    return evaluation_metrics_df, evaluation_metrics_dict, predictions, references, n_gen_tokens, n_gen_times





if __name__ == '__main__':
    results_path = '../RESULTS'
    os.makedirs(results_path, exist_ok=True)

    # --------- SQuAD --------- #
    loader = YANDataLoader(batch_size=1, max_batches={'test':1000}).create_loader('../data/squad', yan_tok)[2]
    steps_all = [1, 2, 3, 4, 6, 8]
    metrics_list = []
    task = 'qa'
    dataset = 'squad'
    for step in steps_all:
        print(f"------ step:{step} ------")
        metrics = evaluate_yan(task, dataset, device, dp_config, yan_tok, loader, n_timesteps=step)[0]
        metrics['n_timesteps'] = step
        metrics['model'] = 'YAN'
        metrics['task'] = task
        metrics['dataset'] = dataset
        metrics_list.append(metrics)
    df = pd.concat(metrics_list, ignore_index=True)
    df.to_csv(os.path.join(results_path, f"{model_name}_{dataset}.csv"), index=False)

    # --------- bAbI --------- #
    loader = YANDataLoader(batch_size=1, max_batches={'test':1000}).create_loader('../data/babiqa', yan_tok)[2]
    steps_all = [1, 2, 3, 4, 6, 8]
    metrics_list = []
    task = 'qa'
    dataset = 'babiqa'
    for step in steps_all:
        print(f"------ step:{step} ------")
        metrics = evaluate_yan(task, dataset, device, dp_config, yan_tok, loader, n_timesteps=step)[0]
        metrics['n_timesteps'] = step
        metrics['model'] = 'YAN'
        metrics['task'] = task
        metrics['dataset'] = dataset
        metrics_list.append(metrics)
    df = pd.concat(metrics_list, ignore_index=True)
    df.to_csv(os.path.join(results_path, f"{model_name}_{dataset}.csv"), index=False)


    # --------- SimpleStories --------- #
    loader = YANDataLoader(batch_size=1, max_batches={'test':1000}).create_loader('../data/simplestories', yan_tok)[2]
    steps_all = [1, 2, 3, 4, 6, 8]
    metrics_list = []
    task = 'last'
    dataset = 'simplestories'
    for step in steps_all:
        print(f"------ step:{step} ------")
        metrics = evaluate_yan(task, dataset, device, dp_config, yan_tok, loader, n_timesteps=step)[0]
        metrics['n_timesteps'] = step
        metrics['model'] = 'YAN'
        metrics['task'] = task
        metrics['dataset'] = dataset
        metrics_list.append(metrics)
    df = pd.concat(metrics_list, ignore_index=True)
    df.to_csv(os.path.join(results_path, f"{model_name}_{dataset}.csv"), index=False)

    # --------- ROCStories --------- #
    loader = YANDataLoader(batch_size=1, max_batches={'test':1000}).create_loader('../data/rocstories', yan_tok)[2]
    steps_all = [1, 2, 3, 4, 6, 8]
    metrics_list = []
    task = 'last'
    dataset = 'rocstories'
    for step in steps_all:
        print(f"------ step:{step} ------")
        metrics = evaluate_yan(task, dataset, device, dp_config, yan_tok, loader, n_timesteps=step)[0]
        metrics['n_timesteps'] = step
        metrics['model'] = 'YAN'
        metrics['task'] = task
        metrics['dataset'] = dataset
        metrics_list.append(metrics)
    df = pd.concat(metrics_list, ignore_index=True)
    df.to_csv(os.path.join(results_path, f"{model_name}_{dataset}.csv"), index=False)



    # --------- AGNews --------- #
    loader = YANDataLoader(batch_size=1, max_batches={'test':1000}).create_loader('../data/agnews', yan_tok)[2]
    steps_all = [1, 2, 3, 4, 6, 8]
    metrics_list = []
    task = 'qa'
    dataset = 'agnews'
    for step in steps_all:
        print(f"------ step:{step} ------")
        metrics = evaluate_yan(task, dataset, device, dp_config, yan_tok, loader, n_timesteps=step)[0]
        metrics['n_timesteps'] = step
        metrics['model'] = 'YAN'
        metrics['task'] = task
        metrics['dataset'] = dataset
        metrics_list.append(metrics)
    df = pd.concat(metrics_list, ignore_index=True)
    df.to_csv(os.path.join(results_path, f"{model_name}_{dataset}.csv"), index=False)



    # --------- DBpedia --------- #
    loader = YANDataLoader(batch_size=1, max_batches={'test':1000}).create_loader('../data/dbpedia', yan_tok)[2]
    steps_all = [1, 2, 3, 4, 6, 8]
    metrics_list = []
    task = 'qa'
    dataset = 'dbpedia'
    for step in steps_all:
        print(f"------ step:{step} ------")
        metrics = evaluate_yan(task, dataset, device, dp_config, yan_tok, loader, n_timesteps=step)[0]
        metrics['n_timesteps'] = step
        metrics['model'] = 'YAN'
        metrics['task'] = task
        metrics['dataset'] = dataset
        metrics_list.append(metrics)
    df = pd.concat(metrics_list, ignore_index=True)
    df.to_csv(os.path.join(results_path, f"{model_name}_{dataset}.csv"), index=False)



    # --------- SST-2 --------- #
    loader = YANDataLoader(batch_size=1, max_batches={'test':1000}).create_loader('../data/sst2', yan_tok)[2]
    steps_all = [1, 2, 3, 4, 6, 8]
    metrics_list = []
    task = 'qa'
    dataset = 'sst2'
    for step in steps_all:
        print(f"------ step:{step} ------")
        metrics = evaluate_yan(task, dataset, device, dp_config, yan_tok, loader, n_timesteps=step)[0]
        metrics['n_timesteps'] = step
        metrics['model'] = 'YAN'
        metrics['task'] = task
        metrics['dataset'] = dataset
        metrics_list.append(metrics)
    df = pd.concat(metrics_list, ignore_index=True)
    df.to_csv(os.path.join(results_path, f"{model_name}_{dataset}.csv"), index=False)



    # --------- NarrativeQA --------- #
    loader = YANDataLoader(batch_size=1, max_batches={'test':500}).create_loader('../data/narrativeqa', yan_tok)[2]
    steps_all = [1, 2, 3, 4, 6, 10, 15, 20, 30, 100, 500, 1000]
    metrics_list = []
    task = 'infill'
    dataset = 'narrativeqa'
    for step in steps_all:
        print(f"------ step:{step} ------")
        metrics = evaluate_yan(task, dataset, device, dp_config, yan_tok, loader, n_timesteps=step)[0]
        metrics['n_timesteps'] = step
        metrics['model'] = 'YAN'
        metrics['task'] = task
        metrics['dataset'] = dataset
        metrics_list.append(metrics)
    df = pd.concat(metrics_list, ignore_index=True)
    df.to_csv(os.path.join(results_path, f"{model_name}_{dataset}.csv"), index=False)




