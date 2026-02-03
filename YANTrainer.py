# trainer/YANTrainer.py

# =================================================== #
#   This script contains the Trainer class            #
# =================================================== #

import os
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from logging import Logger
from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple, List
from torch.utils.data.dataloader import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP

### ---- Trainer class ---- ###
class Trainer(ABC):
    """Trainer class
    Args:
        model: model to train
        model_cfg: model configuration
        loss_cfg: loss configuration
        trainer_cfg: trainer configuration
        device: device the train the model
        logger (optional): logger to output training process in .log file
        train_sampler (optional): DistributedSamplers when using DDP
        is_main_process (optional): whether it is the main process when using DDP
    """
    def __init__(self, model: nn.Module | DDP, 
                 model_cfg: Dict[str, Any], loss_cfg: Dict[str, Any], trainer_cfg: Dict[str, Any], 
                 device: torch.device, teacher_model: nn.Module | None = None, tokenizer = None,
                 logger: Logger | None = None, train_sampler = None, is_main_process: bool = True):
        super().__init__()
        self.model = model
        
        self.model_cfg = model_cfg
        self.loss_cfg = loss_cfg
        self.trainer_cfg = trainer_cfg
        self.device = device
        
        self.teacher_model = teacher_model
        self.tokenizer = tokenizer

        self.logger = logger
        self.train_sampler = train_sampler
        self.is_main_process = is_main_process

        self.base_lr = trainer_cfg['lr']
        self.warmup_steps = trainer_cfg.get('warmup_steps', 0)
        self.weight_decay = trainer_cfg.get('weight_decay', 0.01)
        self.optimizer = self.get_optimizer(self.base_lr, self.weight_decay)
        self.scheduler = self.get_scheduler(self.warmup_steps)

    def _unwrap_ddp(self, model_to_unwrap: nn.Module | DDP) -> nn.Module:
        """If model_to_unwrap is wrapped by DistributedDataParallel (or has .module), unwrap it; 
        otherwise return model_to_unwrap itself."""
        if isinstance(model_to_unwrap, DDP):
            return model_to_unwrap.module
        return model_to_unwrap


    def _ddp_mean_scalar(self, x: torch.Tensor) -> float:
        """Compute mean of scalar across DDP GPUs, return float"""
        if not torch.is_tensor(x):
            x = torch.tensor(float(x), device=self.device)
        if dist.is_initialized():
            dist.all_reduce(x, op=dist.ReduceOp.SUM)
            x = x / dist.get_world_size()
        return x.item()
    
        
    def _log(self, message: str):
        """If logger exists, then records to log, otherwise print."""
        if self.logger:
            self.logger.info(message)
        else:
            print(message)
         
    def get_optimizer(self, base_lr: float, weight_decay: float = 0.01):
        return torch.optim.AdamW(self.model.parameters(), lr=base_lr, 
                                 betas=(0.9, 0.999), eps=1e-8, 
                                 weight_decay=weight_decay)
    
    def get_scheduler(self, warmup_steps: int):
        """Create a linear warmup learning rate scheduler"""
        if warmup_steps > 0:
            return torch.optim.lr_scheduler.LinearLR(
                self.optimizer,
                start_factor=1e-6,
                end_factor=1.0,
                total_iters=warmup_steps
            )
        else:
            return None

    def get_batch_dict(self, loader_batch: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """Get and move the dictionary for the given batch"""
        batch_dict = {key: tensor.to(self.device, non_blocking=True) 
                      for key, tensor in loader_batch.items()}
        return batch_dict
 
    @abstractmethod
    def get_batch_loss(self, batch_dict: Dict[str, Any]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        pass

    def preview_validation_text(self, val_loader, tokenizer, ckpt_dir) -> None:
        pass

    def generate_span_corruption(self, loader_batch: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """Generate span corruption.
        Args:
            loader_batch: {'input_ids': (B, L), 'att_mask': (B, L)}.
        Returns:
            { 'input_ids': (B, L),      # masked collapsed-span ids
              'tgt_ids':   (B, L),      # deep copy of original input_ids
              'att_mask':  (B, L) }     # original att_mask
        """
        device = self.device
        mask_token_id = self.model_cfg["mask_token_id"]
        pad_token_id = self.model_cfg["pad_token_id"]
        bos_token_id = self.model_cfg["bos_token_id"]
        eos_token_id = self.model_cfg["eos_token_id"]

        input_ids = loader_batch["input_ids"].to(device)
        att_mask = loader_batch["att_mask"].to(device)
        tgt_ids = input_ids.clone()

        mask_ratio = self.trainer_cfg['mask_ratio']
        remove_answer_prob = self.trainer_cfg['remove_answer_prob']
        corrupt_span_prob = self.trainer_cfg['corrupt_span_prob']
        max_span_len = self.trainer_cfg['max_span_len']

        B, L = input_ids.shape
        out_ids = torch.full((B, L), pad_token_id, dtype=input_ids.dtype, device=device)

        # Pattern that marks the start of the ANSWER section ("####### ANSWER #######\n\n")
        answer_start = torch.tensor([98964, 97804, 643, 27370, 51624],
                                    dtype=input_ids.dtype, device=device)
        pat_len = answer_start.numel()

        # Helper: ensure sequence starts with <bos> and ends with <eos>
        def _ensure_bos_eos(seq_1d: torch.Tensor) -> torch.Tensor:
            if seq_1d.numel() == 0:
                return torch.tensor([bos_token_id, eos_token_id], dtype=input_ids.dtype, device=device)

            if seq_1d[0].item() != bos_token_id:
                seq_1d = torch.cat(
                    [torch.tensor([bos_token_id], dtype=seq_1d.dtype, device=device), seq_1d],
                    dim=0
                )

            if seq_1d[-1].item() != eos_token_id:
                seq_1d = torch.cat(
                    [seq_1d, torch.tensor([eos_token_id], dtype=seq_1d.dtype, device=device)],
                    dim=0
                )

            return seq_1d

        # Process each sample independently (answer removal / span selection / collapse is per sequence).
        for b in range(B):
            # "Real" tokens are determined by ORIGINAL att_mask (includes <bos> ... <eos>).
            orig_real_len = int(att_mask[b].sum().item())
            if orig_real_len <= 0:
                continue

            # 1D, valid tokens only (original)
            seq = input_ids[b, :orig_real_len]

            # ---- Step 1: optionally remove answer part before any infilling decision ----
            if remove_answer_prob > 0.0 and (torch.rand((), device=device).item() < remove_answer_prob):
                split_idx = None
                if orig_real_len >= pat_len:
                    # Find first occurrence of answer_start
                    for i in range(0, orig_real_len - pat_len + 1):
                        if torch.equal(seq[i:i + pat_len], answer_start):
                            split_idx = i
                            break

                if split_idx is not None:
                    # Keep up to and including the ANSWER header, then insert a single <mask>
                    # to represent the removed answer.
                    kept = seq[: split_idx + pat_len]
                    seq = torch.cat([kept, torch.tensor([mask_token_id], dtype=seq.dtype, device=device)], dim=0)

            # Always enforce <bos> ... <eos> on the (possibly shortened) sequence
            seq = _ensure_bos_eos(seq)

            # ---- Step 2: decide whether to do text-infilling at all (independent of remove_answer_prob) ----
            do_infill = (corrupt_span_prob > 0.0) and (torch.rand((), device=device).item() < corrupt_span_prob)
            if not do_infill:
                # No masking/collapsing; just pad-right to (B, L)
                seq_list = seq.tolist()[:L]
                out_ids[b, :len(seq_list)] = torch.tensor(seq_list, dtype=input_ids.dtype, device=device)
                continue

            # ---- Step 3: span corruption (text-infilling) on the shortened sequence ----
            real_len = int(seq.numel())
            if real_len <= 2:
                # Degenerate: only <bos>, <eos> (or less). Just copy and pad.
                seq_list = seq.tolist()[:L]
                out_ids[b, :len(seq_list)] = torch.tensor(seq_list, dtype=input_ids.dtype, device=device)
                continue

            eligible_len = real_len - 2  # exclude <bos>, <eos>
            if eligible_len <= 0 or mask_ratio <= 0.0:
                seq_list = seq.tolist()[:L]
                out_ids[b, :len(seq_list)] = torch.tensor(seq_list, dtype=input_ids.dtype, device=device)
                continue

            num_to_mask = int(round(mask_ratio * eligible_len))
            num_to_mask = max(0, min(num_to_mask, eligible_len))
            if num_to_mask == 0:
                seq_list = seq.tolist()[:L]
                out_ids[b, :len(seq_list)] = torch.tensor(seq_list, dtype=input_ids.dtype, device=device)
                continue

            # Boolean mask over real_len positions; we will only ever set positions in [1, real_len-2].
            masked = torch.zeros(real_len, dtype=torch.bool, device=device)

            # Greedy randomized span placement without overlap.
            masked_count = 0
            max_tries = 10 * eligible_len  # safety to avoid pathological loops
            tries = 0

            def available_starts() -> torch.Tensor:
                # positions 1..real_len-2 inclusive
                avail = (~masked[1:real_len-1]).nonzero(as_tuple=False).view(-1)
                return avail + 1

            while masked_count < num_to_mask and tries < max_tries:
                tries += 1
                starts = available_starts()
                if starts.numel() == 0:
                    break

                # Sample a start uniformly from remaining available positions.
                start_idx = starts[torch.randint(0, starts.numel(), (1,), device=device)].item()

                # Sample a span length uniformly in [1, max_span_len], then clip to boundaries.
                span_len = int(torch.randint(1, max_span_len + 1, (1,), device=device).item())

                # Clip span to not include <eos> at (real_len-1) and not exceed boundaries.
                end_exclusive = min(start_idx + span_len, real_len - 1)  # do not include eos
                if end_exclusive <= start_idx:
                    continue

                # Further trim to avoid overlap: only extend while unmasked.
                end = start_idx
                assert isinstance(end, int)
                while end < end_exclusive and (not masked[end].item()):
                    end += 1

                # Now span is [start_idx, end) with no overlap.
                if end <= start_idx:
                    continue

                # If this span would mask too many tokens beyond quota, trim it.
                remaining = num_to_mask - masked_count
                span_actual = min(end - start_idx, remaining)
                end = start_idx + span_actual
                if end <= start_idx:
                    continue

                masked[start_idx:end] = True
                masked_count += (end - start_idx)

            # Build collapsed sequence: replace each masked span by a single mask token id.
            src = seq.tolist()
            new_seq = []
            i = 0
            while i < real_len:
                if masked[i].item():
                    new_seq.append(mask_token_id)
                    j = i + 1
                    while j < real_len and masked[j].item():
                        j += 1
                    i = j
                else:
                    new_seq.append(src[i])
                    i += 1

            # Ensure <bos> ... <eos> after corruption too (defensive)
            new_seq_t = torch.tensor(new_seq, dtype=input_ids.dtype, device=device)
            new_seq_t = _ensure_bos_eos(new_seq_t)

            new_seq_list = new_seq_t.tolist()[:L]
            out_ids[b, :len(new_seq_list)] = torch.tensor(new_seq_list, dtype=input_ids.dtype, device=device)

        return {
            'input_ids': out_ids,   # (B, L) corrupted token ids
            'tgt_ids': tgt_ids,     # (B, L) clean token ids
            'att_mask': att_mask    # (B, L) attention mask of clean token ids
        }


    def get_epoch_loss(self, loader: DataLoader, is_train: bool = True) -> Tuple[float, Dict[str, float]]:
        """Compute loss over one epoch of the given data loader
        Args:
            loader: data loader
            is_train: True if this is a training epoch, False otherwise
        Returns:
            epoch_tot: float
            epoch_metrics: Dict[str, float]  (per-batch average over the epoch)
        """
        total_loss_sum = torch.tensor(0.0, device=self.device)
        metric_sums: Dict[str, torch.Tensor] = {}
        num_batches = 0

        if is_train:
            self.model.train()
            for loader_batch in loader:
                batch_dict = self.get_batch_dict(loader_batch)

                self.optimizer.zero_grad(set_to_none=True)
                loss, metrics = self.get_batch_loss(batch_dict)
                loss.backward()
                self.optimizer.step()
                if self.scheduler is not None:
                    self.scheduler.step()

                total_loss_sum += loss.detach()
                for k, v in metrics.items():
                    if k not in metric_sums:
                        metric_sums[k] = torch.zeros((), device=self.device)
                    metric_sums[k] += v.detach()
                num_batches += 1
        else:
            self.model.eval()
            with torch.no_grad():
                for loader_batch in loader:
                    batch_dict = self.get_batch_dict(loader_batch)
                    loss, metrics = self.get_batch_loss(batch_dict)

                    total_loss_sum += loss.detach()
                    for k, v in metrics.items():
                        if k not in metric_sums:
                            metric_sums[k] = torch.zeros((), device=self.device)
                        metric_sums[k] += v.detach()
                    num_batches += 1

        if dist.is_initialized():
            dist.all_reduce(total_loss_sum, op=dist.ReduceOp.SUM)
            for k in metric_sums:
                dist.all_reduce(metric_sums[k], op=dist.ReduceOp.SUM)

            nb = torch.tensor(num_batches, device=self.device, dtype=torch.float32)
            dist.all_reduce(nb, op=dist.ReduceOp.SUM)
            total_batches = float(nb.item())
        else:
            total_batches = float(num_batches)

        if total_batches <= 0:
            return 0.0, {}

        epoch_tot = float((total_loss_sum / total_batches).item())
        epoch_metrics = {k: float((v / total_batches).item()) for k, v in metric_sums.items()}
        return epoch_tot, epoch_metrics


    def train(self, train_loader: DataLoader, val_loader: DataLoader | None = None,
              ckpt_dir: str | None = None, start_step: int = 0) -> None:
        """Step-based training loop. Each step = one optimizer update.
        Args:
            train_loader: Training data loader.
            val_loader (optional): Validation data loader.
            ckpt_dir (optional): Directory to save the checkpoint. Must be provided if save_every is not None.
        """
        max_steps = self.trainer_cfg['max_steps']
        log_every = self.trainer_cfg['log_every']
        val_every = self.trainer_cfg.get('val_every', 1)
        save_every = self.trainer_cfg.get('save_every', None)

        if save_every is not None:
            assert ckpt_dir is not None, "Missing ckpt_dir."
            os.makedirs(ckpt_dir, exist_ok=True)
            
        self.global_step = start_step
        self.model.train()
        running: Dict[str, float] = {} # running loss of current `log_every` steps
        epoch = 0

        last_log_metrics = None
        last_val_metrics = None

        while self.global_step < max_steps:
            if self.train_sampler is not None:
                self.train_sampler.set_epoch(epoch)
            
            for batch_idx, loader_batch in enumerate(train_loader):
                # ---- forward ---- #
                batch_dict = self.get_batch_dict(loader_batch)
                loss, metrics = self.get_batch_loss(batch_dict) # per-GPU, per-batch / per-step loss

                # ---- backward & step ---- #
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward() # If on multi-GPUs, this triggers the synchronization of gradients
                self.optimizer.step()
                if self.scheduler is not None:
                    self.scheduler.step()
                
                self.global_step += 1
                # ----- update running stats ----- #
                running["Tot"] = running.get("Tot", 0.0) + float(loss.detach().item())
                for k, v in metrics.items():
                    running[k] = running.get(k, 0.0) + float(v.detach().item())

                # ----- log ----- #
                # (average training loss, averaged across current `log_every` steps)
                if self.global_step % log_every == 0:
                    avg = {k: v / log_every for k, v in running.items()}

                    if dist.is_initialized(): # average across GPUs if on DDP
                        for k in list(avg.keys()):
                            avg[k] = self._ddp_mean_scalar(torch.tensor(avg[k], device=self.device))


                    current_lr = self.optimizer.param_groups[0]['lr']
                    if self.is_main_process:
                        msg = f"🚏 [Train @ Step {self.global_step}/{max_steps}] "
                        msg += " | ".join([f"{k} {avg[k]:<8.6f}" for k in avg.keys()])
                        msg += f" | LR {current_lr:.2e}"
                        self._log(msg)
                    last_log_metrics = {"step": self.global_step,
                                        **{f"(train) {k}": float(avg[k]) for k in avg.keys()},
                                        "lr": float(current_lr)}
                    running = {}
                
                # ----- validation ----- #
                # (average validation loss, averaged across all batches in the entire validation epoch)
                if val_loader is not None and self.global_step % val_every == 0:
                    self.model.eval()
                    with torch.no_grad():
                        val_tot, val_metrics = self.get_epoch_loss(val_loader, is_train=False)
                    if self.is_main_process:
                        msg = f"🧪 [Val @ Step {self.global_step}] Tot {val_tot:<8.6f}"
                        if len(val_metrics) > 0:
                            msg += " | " + " | ".join([f"{k} {val_metrics[k]:<8.6f}" for k in sorted(val_metrics.keys())])
                        self._log(msg)
                    last_val_metrics = {"step": self.global_step, 
                                        "(val) Tot": float(val_tot), 
                                        **{f"(val) {k}": float(v) for k, v in val_metrics.items()}}
                    if self.tokenizer is not None and save_every is not None:
                        self.preview_validation_text(val_loader, self.tokenizer, ckpt_dir)
                    self.model.train()
                
                # ----- save ----- #
                if save_every is not None and self.global_step % save_every == 0:
                    assert ckpt_dir is not None
                    if self.is_main_process:
                        path = os.path.join(ckpt_dir, f"step{self.global_step:08d}.pt")
                        torch.save({
                            "model": self.model.module.state_dict() if hasattr(self.model, "module") else self.model.state_dict(), # type:ignore
                            "step": self.global_step,
                            "last_log_metrics": last_log_metrics,
                            "last_val_metrics": last_val_metrics,
                            "model_config": self.model_cfg,
                            "loss_config": self.loss_cfg,
                            "trainer_config": self.trainer_cfg
                        }, path)
                        self._log(f"🗃️ Checkpoint saved to {path}")
                
                if self.global_step >= max_steps:
                    break
            
            epoch += 1


class YANTrainer(Trainer):
    def split_enc_tgt_qa(self, loader_batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Split each sample in a batch into encoder ids (summary+question) and target ids (answer),
        using a fixed ANSWER header token pattern as the split marker.
        Args:
            loader_batch: {'input_ids':  (B, L), 'att_mask': (B, L)}
        Returns:
            { 'enc_ids':      (B, Lx), 
              'enc_att_mask': (B, Lx), 
              'tgt_ids':      (B, Ly), 
              'att_mask':     (B, Ly) }
        """
        bos_id = self.model_cfg['bos_token_id']                 # 128000
        eos_id = self.model_cfg['eos_token_id']                 # 128001
        pad_id = self.model_cfg['pad_token_id']                 # 128256

        input_ids = loader_batch['input_ids']                          # (B, L)
        input_att_mask = loader_batch['att_mask']                      # (B, L)

        device = input_ids.device
        B, L = input_ids.shape

        # Pattern that marks the start of the ANSWER section ("####### ANSWER #######\n\n")
        answer_start = torch.tensor([98964, 97804, 643, 27370, 51624],
                                    dtype=input_ids.dtype, device=device)
        pat_len = answer_start.numel()

        enc_seqs: List[torch.Tensor] = []
        tgt_seqs: List[torch.Tensor] = []


        # Helper: ensure sequence starts with <bos> and ends with <eos>
        def _ensure_bos_eos(seq: torch.Tensor) -> torch.Tensor:
            # seq is 1D tensor, assumed to be within valid range (no right padding)
            if seq.numel() == 0:
                return torch.tensor([bos_id, eos_id], dtype=input_ids.dtype, device=device)

            # Add BOS if missing
            if seq[0].item() != bos_id:
                seq = torch.cat([torch.tensor([bos_id], dtype=seq.dtype, device=device), seq], dim=0)

            # Add EOS if missing
            if seq[-1].item() != eos_id:
                seq = torch.cat([seq, torch.tensor([eos_id], dtype=seq.dtype, device=device)], dim=0)

            return seq

        for b in range(B):
            # valid length from att mask
            valid_len = int(input_att_mask[b].sum().item())
            seq = input_ids[b, :valid_len]  # 1D, valid tokens only

            # Find first occurrence of answer_start within valid tokens
            # If not found, we fall back to putting everything into enc, and tgt = <bos><eos>.
            split_idx = None
            if valid_len >= pat_len:
                # naive scan; for typical L it's fine; easy to reason about and robust.
                for i in range(0, valid_len - pat_len + 1):
                    if torch.equal(seq[i : i + pat_len], answer_start):
                        split_idx = i
                        break

            if split_idx is None:
                # Fallback: no marker -> all goes to encoder; empty target
                enc_raw = seq
                tgt_raw = torch.empty((0,), dtype=seq.dtype, device=device)
            else:
                enc_raw = seq[:split_idx]          # up to before ANSWER header
                tgt_raw = seq[split_idx:]          # from ANSWER header to end (includes eos if present)

            enc_seq = _ensure_bos_eos(enc_raw)
            tgt_seq = _ensure_bos_eos(tgt_raw)

            enc_seqs.append(enc_seq)
            tgt_seqs.append(tgt_seq)

        # Pad to batch max lengths
        enc_max = max(s.numel() for s in enc_seqs)
        tgt_max = max(s.numel() for s in tgt_seqs)

        enc_ids = torch.full((B, enc_max), pad_id, dtype=input_ids.dtype, device=device)
        tgt_ids = torch.full((B, tgt_max), pad_id, dtype=input_ids.dtype, device=device)

        enc_att_mask = torch.zeros((B, enc_max), dtype=input_att_mask.dtype, device=device)
        att_mask = torch.zeros((B, tgt_max), dtype=input_att_mask.dtype, device=device)

        for b in range(B):
            e = enc_seqs[b]
            t = tgt_seqs[b]

            enc_ids[b, : e.numel()] = e
            tgt_ids[b, : t.numel()] = t

            enc_att_mask[b, : e.numel()] = 1
            att_mask[b, : t.numel()] = 1


        return {
            "enc_ids":      enc_ids,            # (B, Lx)
            "enc_att_mask": enc_att_mask,       # (B, Lx)
            "tgt_ids":      tgt_ids,            # (B, Ly)
            "att_mask":     att_mask            # (B, Ly)
        }

    def split_enc_tgt_summary(self, loader_batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Split each sample in a batch into encoder ids (summary+question) and target ids (answer),
        using a fixed ANSWER header token pattern as the split marker.
        Args:
            loader_batch: {'input_ids':  (B, L), 'att_mask': (B, L)}
        Returns:
            { 'enc_ids':      (B, Lx), 
              'enc_att_mask': (B, Lx), 
              'tgt_ids':      (B, Ly), 
              'att_mask':     (B, Ly) }
        """
        bos_id = self.model_cfg['bos_token_id']                 # 128000
        eos_id = self.model_cfg['eos_token_id']                 # 128001
        pad_id = self.model_cfg['pad_token_id']                 # 128256

        input_ids = loader_batch['input_ids']                          # (B, L)
        input_att_mask = loader_batch['att_mask']                      # (B, L)

        device = input_ids.device
        B, L = input_ids.shape

        # Pattern that marks the start of the SUMMARY section ("####### SUMMARY #######\n\n")
        answer_start = torch.tensor([98964, 96885, 27370, 51624],
                                    dtype=input_ids.dtype, device=device)
        pat_len = answer_start.numel()

        enc_seqs: List[torch.Tensor] = []
        tgt_seqs: List[torch.Tensor] = []


        # Helper: ensure sequence starts with <bos> and ends with <eos>
        def _ensure_bos_eos(seq: torch.Tensor) -> torch.Tensor:
            # seq is 1D tensor, assumed to be within valid range (no right padding)
            if seq.numel() == 0:
                return torch.tensor([bos_id, eos_id], dtype=input_ids.dtype, device=device)

            # Add BOS if missing
            if seq[0].item() != bos_id:
                seq = torch.cat([torch.tensor([bos_id], dtype=seq.dtype, device=device), seq], dim=0)

            # Add EOS if missing
            if seq[-1].item() != eos_id:
                seq = torch.cat([seq, torch.tensor([eos_id], dtype=seq.dtype, device=device)], dim=0)

            return seq

        for b in range(B):
            # valid length from att mask
            valid_len = int(input_att_mask[b].sum().item())
            seq = input_ids[b, :valid_len]  # 1D, valid tokens only

            # Find first occurrence of answer_start within valid tokens
            # If not found, we fall back to putting everything into enc, and tgt = <bos><eos>.
            split_idx = None
            if valid_len >= pat_len:
                # naive scan; for typical L it's fine; easy to reason about and robust.
                for i in range(0, valid_len - pat_len + 1):
                    if torch.equal(seq[i : i + pat_len], answer_start):
                        split_idx = i
                        break

            if split_idx is None:
                # Fallback: no marker -> all goes to encoder; empty target
                enc_raw = seq
                tgt_raw = torch.empty((0,), dtype=seq.dtype, device=device)
            else:
                enc_raw = seq[:split_idx]          # up to before ANSWER header
                tgt_raw = seq[split_idx:]          # from ANSWER header to end (includes eos if present)

            enc_seq = _ensure_bos_eos(enc_raw)
            tgt_seq = _ensure_bos_eos(tgt_raw)

            enc_seqs.append(enc_seq)
            tgt_seqs.append(tgt_seq)

        # Pad to batch max lengths
        enc_max = max(s.numel() for s in enc_seqs)
        tgt_max = max(s.numel() for s in tgt_seqs)

        enc_ids = torch.full((B, enc_max), pad_id, dtype=input_ids.dtype, device=device)
        tgt_ids = torch.full((B, tgt_max), pad_id, dtype=input_ids.dtype, device=device)

        enc_att_mask = torch.zeros((B, enc_max), dtype=input_att_mask.dtype, device=device)
        att_mask = torch.zeros((B, tgt_max), dtype=input_att_mask.dtype, device=device)

        for b in range(B):
            e = enc_seqs[b]
            t = tgt_seqs[b]

            enc_ids[b, : e.numel()] = e
            tgt_ids[b, : t.numel()] = t

            enc_att_mask[b, : e.numel()] = 1
            att_mask[b, : t.numel()] = 1


        return {
            "enc_ids":      enc_ids,            # (B, Lx)
            "enc_att_mask": enc_att_mask,       # (B, Lx)
            "tgt_ids":      tgt_ids,            # (B, Ly)
            "att_mask":     att_mask            # (B, Ly)
        }


    def split_enc_tgt_last(self, loader_batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Split each sample in a batch into encoder ids (all bust last token) and target ids (last token).
        Args:
            loader_batch: {'input_ids': (B, L), 'att_mask': (B, L)}.
        Returns:
            { 'enc_ids':      (B, Lx), 
              'enc_att_mask': (B, Lx), 
              'tgt_ids':      (B, Ly), 
              'att_mask':     (B, Ly) }
        """
        assert self.tokenizer is not None
        input_ids = loader_batch['input_ids']       # (B, L)
        input_att_mask = loader_batch['att_mask']   # (B, L)
        B, L = input_ids.shape
        
        # ---- 1) Decode input tokens into texts ----
        texts = []
        for b in range(B):
            valid_len = int(input_att_mask[b].sum().item())
            texts.append(input_ids[b, :valid_len].tolist())
        texts = self.tokenizer.batch_decode(texts, skip_special_tokens=True)
        
        # ---- 2) Split last word and replace with mask token ----
        # E.g., "This is an example." -> prefix="This is an ", word="example", suffix="."
        #       enc="This is an <mask>.", tgt="example"
        enc_texts, tgt_texts = [], []
        pattern = re.compile(r"([A-Za-z]+)([^A-Za-z]*)$", flags=re.S)
        for text in texts:
            words = text.strip()
            m = pattern.search(text)

            if m is None:
                enc_texts.append(self.tokenizer.mask_token)
                tgt_texts.append(self.tokenizer.pad_token)
                continue

            last_word = m.group(1)
            suffix = m.group(2)
            prefix = text[:m.start(1)]

            enc_texts.append(f"{prefix}{self.tokenizer.mask_token}{suffix}")
            tgt_texts.append(last_word)
        
        # ---- 3) Add <EOS> ----
        enc_texts = [t + self.tokenizer.eos_token for t in enc_texts]
        tgt_texts = [t + self.tokenizer.eos_token for t in tgt_texts]

        # ---- 4) Retokenize ----
        enc = self.tokenizer(
            enc_texts, padding=True, truncation=True, return_tensors="pt", add_special_tokens=True
        )
        tgt = self.tokenizer(
            tgt_texts, padding=True, truncation=True, return_tensors="pt", add_special_tokens=True
        )

        return {
            'enc_ids': enc['input_ids'].to(self.device),
            'enc_att_mask': enc['attention_mask'].to(self.device),
            'tgt_ids': tgt['input_ids'].to(self.device),
            'att_mask': tgt['attention_mask'].to(self.device)
        }


    def split_enc_tgt_middle(self, loader_batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Remove the middle sentence of the rocstories dataset.
        Args:
            loader_batch: {'input_ids': (B, L), 'att_mask': (B, L)}.
        Returns:
            { 'enc_ids':      (B, Lx),  # containing the first two and last two sentences, with the middle replaced with one <|mask|>
              'enc_att_mask': (B, Lx),  # attention mask for enc_ids, 1=valid tokens, 0=paddings
              'tgt_ids':      (B, Ly),  # the original input_ids, i.e., ground truth complete tokens
              'att_mask':     (B, Ly) } # the original att_mask, i.e., ground truth attention mask
        """
        assert self.tokenizer is not None
        input_ids = loader_batch['input_ids']       # (B, L)
        input_att_mask = loader_batch['att_mask']   # (B, L)
        B, L = input_ids.shape

        # ---- 1) Decode input tokens into texts ----
        texts = []
        for b in range(B):
            valid_len = int(input_att_mask[b].sum().item())
            texts.append(input_ids[b, :valid_len].tolist())
        texts = self.tokenizer.batch_decode(texts, skip_special_tokens=True)

        # ---- 2) Replace the middle sentence with <|mask|> ----
        masked_texts = []
        for t in texts:
            sentences = [s.strip() for s in t.split('.') if s.strip()]
            if len(sentences) >= 4:
                sentences[2] = self.tokenizer.mask_token
                masked_text = '. '.join(sentences) + '.'
            else:
                t2 = t.strip()
                mid = len(t2) // 2
                masked_text = (
                    t2[:mid].rstrip()
                    + " " + self.tokenizer.mask_token + " "
                    + t2[mid:].lstrip()
                )
                if not masked_text.endswith('.'):
                    masked_text += '.'
            masked_texts.append(masked_text + self.tokenizer.eos_token) # add <EOS>

        # ---- 3) Retokenize ----
        enc = self.tokenizer(
            masked_texts, padding=True, truncation=True, return_tensors="pt", 
            add_special_tokens=True, max_length=self.model_cfg['max_length']
        )
        return {
            'enc_ids': enc['input_ids'].to(self.device),
            'enc_att_mask': enc['attention_mask'].to(self.device),
            'tgt_ids': input_ids.to(self.device),
            'att_mask': input_att_mask.to(self.device)
        }


    def get_batch_dict(self, loader_batch: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        assert self.trainer_cfg['flow_split'] in ['qa', 'infill', 'last', 'middle', 'summary']
        if self.trainer_cfg['flow_split'] == 'qa':
            batch = self.split_enc_tgt_qa(loader_batch)
        elif self.trainer_cfg['flow_split'] == 'last':
            batch = self.split_enc_tgt_last(loader_batch)
        elif self.trainer_cfg['flow_split'] == 'middle':
            batch = self.split_enc_tgt_middle(loader_batch)
        elif self.trainer_cfg['flow_split'] == 'summary':
            batch = self.split_enc_tgt_summary(loader_batch)
        else:
            batch = self.generate_span_corruption(loader_batch)
            batch['enc_ids'] = batch['input_ids'].clone()
            batch['enc_att_mask'] = batch['att_mask'].clone()
            del batch['input_ids']
        return {key: tensor.to(self.device, non_blocking=True) for key, tensor in batch.items()}

        
    def _compute_masked_mse(self, pred: torch.Tensor, target: torch.Tensor,
                            att_mask: torch.Tensor):
        """
        Args:
            pred, target: (B, L, d)
            att_mask: (B, L)
        """
        mask = att_mask.unsqueeze(-1)                   # (B, L, 1)

        diff = (pred - target) ** 2
        diff = diff * mask                              # (B, L, d)
        
        n_valid = mask.sum(dtype=diff.dtype).clamp(min=1.0)
        return diff.sum() / n_valid

    def _compute_moe_nll(
        self, vec_field_target: torch.Tensor, vec_field_experts: torch.Tensor,
        gate_pi: torch.Tensor, att_mask: torch.Tensor
    ) -> torch.Tensor:
        """Negative log-likelihood of Gaussian mixture (token-wise gates)
        Args:
            vec_field_target: (B, Ly, d)
            vec_field_experts: (B, K, Ly, d)
            gate_pi: (B, Ly, K)
            att_mask: (B, Ly)
        Returns:
            Average NLL over valid tokens
        """
        B, Ly, d = vec_field_target.shape
        K = vec_field_experts.shape[1]
        assert gate_pi.shape == (B, Ly, K), f"gate_pi must be (B,Ly,K), got {gate_pi.shape}"
        moe_sigma = self.model_cfg.get('moe_sigma', 1.0)

        log_pi = torch.log(gate_pi.clamp_min(1e-8)).permute(0, 2, 1)    # (B, K, Ly)

        diff = vec_field_target.unsqueeze(1) - vec_field_experts        # (B, K, Ly, d)
        diff_sq_sum = (diff ** 2).sum(-1)                               # (B, K, Ly)

        sigma2 = float(max(moe_sigma ** 2, 1e-8))
        log_norm_const = -0.5 * d * torch.log(torch.tensor(
            2.0 * torch.pi * sigma2, device=vec_field_target.device
        ))
        log_gauss = log_norm_const - 0.5 * diff_sq_sum / sigma2         # (B, K, Ly)

        log_mix = torch.logsumexp(log_pi + log_gauss, dim=1)            # (B, Ly)
        nll = -log_mix                                                  # (B, Ly)

        mask = att_mask.to(dtype=nll.dtype)                             # (B, Ly)
        n_valid = mask.sum().clamp(min=1.0)
        return (nll * mask).sum() / n_valid


    @staticmethod
    def _rbf_kernel(x: torch.Tensor, y: torch.Tensor, sigma: float) -> torch.Tensor:
        """
        x: (N, D), y: (M, D)
        return: (N, M) kernel matrix
        """
        # ||x-y||^2 = x^2 + y^2 - 2xy
        x2 = (x * x).sum(dim=1, keepdim=True)          # (N, 1)
        y2 = (y * y).sum(dim=1, keepdim=True).t()      # (1, M)
        xy = x @ y.t()                                 # (N, M)
        dist2 = (x2 + y2 - 2.0 * xy).clamp_min(0.0)
        k = torch.exp(-dist2 / (2.0 * (sigma ** 2)))
        return k

    @classmethod
    def _mmd_rbf_multi_kernel(cls, x: torch.Tensor, y: torch.Tensor, sigmas: list[float]) -> torch.Tensor:
        """
        MMD^2 with multi-RBF kernels.
        x: (N, D), y: (M, D)
        """
        assert x.dim() == 2 and y.dim() == 2
        n = x.size(0)
        m = y.size(0)

        # Guard: if too small, return 0 (no meaningful MMD)
        if n < 2 or m < 2:
            return torch.zeros((), device=x.device, dtype=x.dtype)

        k_xx = torch.zeros((n, n), device=x.device, dtype=x.dtype)
        k_yy = torch.zeros((m, m), device=x.device, dtype=x.dtype)
        k_xy = torch.zeros((n, m), device=x.device, dtype=x.dtype)

        for s in sigmas:
            s = float(s)
            k_xx = k_xx + cls._rbf_kernel(x, x, s)
            k_yy = k_yy + cls._rbf_kernel(y, y, s)
            k_xy = k_xy + cls._rbf_kernel(x, y, s)


        # Unbiased: exclude diagonals for xx/yy
        k_xx = (k_xx.sum() - k_xx.diag().sum()) / (n * (n - 1))
        k_yy = (k_yy.sum() - k_yy.diag().sum()) / (m * (m - 1))
        k_xy = k_xy.mean()

        mmd2 = k_xx + k_yy - 2.0 * k_xy
        return mmd2


    def get_batch_loss(self, batch_dict: Dict[str, Any]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        enc_ids = batch_dict['enc_ids']
        tgt_ids = batch_dict['tgt_ids']
        att_mask = batch_dict['att_mask']
        enc_att_mask = batch_dict['enc_att_mask']

        vec_field_target, vec_field_experts, gate_pi, _, x1, x1_hat, logits, _ = self.model(
            enc_ids, tgt_ids, att_mask, enc_att_mask
        )

        # ---- (A) MoE Flow Matching loss: GMM NLL ----
        vf = self._compute_moe_nll(vec_field_target, vec_field_experts, gate_pi, att_mask)

        # ---- (B) MSE in latent space ----
        distill = self._compute_masked_mse(x1_hat, x1, att_mask)

        # ---- (C) CE ----
        ce = F.cross_entropy(
            logits.transpose(1, 2), tgt_ids,
            ignore_index=self.model_cfg['pad_token_id'],
            reduction='mean',
            label_smoothing=self.loss_cfg.get('label_smoothing', 0.0)
        )

        # ---- (D) Gate regularization ----
        # 1) entropy regularization
        eps = 1e-8
        logp = torch.log(gate_pi.clamp_min(eps))
        ent_tok = -(gate_pi * logp).sum(dim=-1)                                 # (B, Ly)

        mask = att_mask.to(ent_tok.dtype)                                       # (B, Ly)
        n_valid = mask.sum().clamp(min=1.0)
        entropy = (ent_tok * mask).sum() / n_valid                              # scalar
        # 2) load-balance regularization
        K = gate_pi.shape[-1]
        mask3 = att_mask.unsqueeze(-1).to(gate_pi.dtype)                        # (B, Ly, 1)
        p_avg = (gate_pi * mask3).sum(dim=(0,1)) / mask3.sum().clamp(min=1.0)   # (K,)

        load_balance = (p_avg * torch.log(p_avg.clamp_min(1e-8) * K)).sum()



        # ---- (E) Latent regularization ----
        z_valid = x1[att_mask == 1]                                             # (N_valid, d)
        # 1) MMD
        max_mmd_points = 2000
        if z_valid.numel() > 0:
            if z_valid.size(0) > max_mmd_points:
                idx = torch.randperm(z_valid.size(0), device=z_valid.device)[:max_mmd_points]
                z_valid_mmd = z_valid[idx]
            else:
                z_valid_mmd = z_valid
            z_prior = torch.randn_like(z_valid_mmd)                             # (N_valid, d)
            mmd2 = self._mmd_rbf_multi_kernel(z_valid_mmd, z_prior, 
                                              sigmas=[0.2, 0.5, 1.0, 2.0, 5.0])
        else:
            mmd2 = torch.zeros((), device=x1.device, dtype=x1.dtype)
        
        # 2) L2
        if z_valid.numel() > 0:
            z_l2 = (z_valid ** 2).mean()
        else:
            z_l2 = torch.zeros((), device=x1.device, dtype=x1.dtype)
        

        # ---- (F) Total ----
        lambda_vf = self.loss_cfg.get('lambda_vf', 1.0)
        lambda_distill = self.loss_cfg.get('lambda_distill', 0.0)
        lambda_ce = self.loss_cfg.get('lambda_ce', 0.0)
        lambda_gate_entropy = self.loss_cfg.get('lambda_gate_entropy', 0.0)
        lambda_load_balance = self.loss_cfg.get('lambda_load_balance', 0.0)
        lambda_mmd = self.loss_cfg.get('lambda_mmd', 0.0)
        lambda_l2 = self.loss_cfg.get('lambda_l2', 0.0)

        loss = (
            lambda_vf * vf +
            lambda_distill * distill + 
            lambda_ce * ce +
            lambda_gate_entropy * (-entropy) +
            lambda_load_balance * load_balance +
            lambda_mmd * mmd2 +
            lambda_l2 * z_l2
        )
        metrics = {}
        metrics["NLL"] = vf
        metrics["MSE"] = distill
        metrics["CE"] = ce
        metrics["GateEntr"] = entropy
        metrics["LoadBal"] = load_balance
        metrics["MMD"] = mmd2
        metrics["L2"] = z_l2

        return loss, metrics

    def preview_validation_text(self, val_loader, tokenizer, ckpt_dir) -> None:
        """Generate a qualitative text preview during validation."""
        if not self.is_main_process:
            return None

        # ---- Sample one batch ----
        preview_batch = next(iter(val_loader))
        batch_dict = self.get_batch_dict({
            'input_ids': preview_batch["input_ids"][:1], 'att_mask': preview_batch["att_mask"][:1]
        })
        enc_ids = batch_dict["enc_ids"]
        tgt_ids = batch_dict["tgt_ids"]

        # ---- YAN Forward ----
        with torch.no_grad():
            _, _, _, _, _, _, logits, _ = self.model(
                enc_ids, tgt_ids, batch_dict['att_mask'], batch_dict['enc_att_mask']
            )

        # ---- Decode ----
        prefix_text = tokenizer.decode(enc_ids[0].tolist(), skip_special_tokens=False)
        true_text = tokenizer.decode(tgt_ids[0].tolist(), skip_special_tokens=False)
        pred_ids = logits.argmax(dim=-1)[0] # (L,)
        pred_text = tokenizer.decode(pred_ids.tolist(), skip_special_tokens=False)

        # ---- Write preview to a separate file ----
        preview_path = os.path.join(ckpt_dir, "preview_flow.txt")

        with open(preview_path, "a", encoding="utf-8") as f:
            f.write(f"\n\n==================== Step {self.global_step} ====================\n\n")
            f.write("▶︎ Source ◀\n\n")
            f.write(prefix_text[:5000] + ("..." if len(prefix_text) > 5000 else "") + "\n\n")
            f.write("▶︎ Target ◀\n\n")
            f.write(true_text[:5000] + ("..." if len(true_text) > 5000 else "") + "\n\n")
            f.write("▶︎ YAN ◀\n\n")
            f.write(pred_text[:5000] + ("..." if len(pred_text) > 5000 else "") + "\n")







class YANAutoEncoderTrainer(Trainer):
    def get_batch_dict(self, loader_batch: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        batch = self.generate_span_corruption(loader_batch)
        return {key: tensor.to(self.device, non_blocking=True) for key, tensor in batch.items()}

    @staticmethod
    def _rbf_kernel(x: torch.Tensor, y: torch.Tensor, sigma: float) -> torch.Tensor:
        """
        x: (N, D), y: (M, D)
        return: (N, M) kernel matrix
        """
        # ||x-y||^2 = x^2 + y^2 - 2xy
        x2 = (x * x).sum(dim=1, keepdim=True)          # (N, 1)
        y2 = (y * y).sum(dim=1, keepdim=True).t()      # (1, M)
        xy = x @ y.t()                                 # (N, M)
        dist2 = (x2 + y2 - 2.0 * xy).clamp_min(0.0)
        k = torch.exp(-dist2 / (2.0 * (sigma ** 2)))
        return k

    @classmethod
    def _mmd_rbf_multi_kernel(cls, x: torch.Tensor, y: torch.Tensor, sigmas: list[float]) -> torch.Tensor:
        """
        MMD^2 with multi-RBF kernels.
        x: (N, D), y: (M, D)
        """
        assert x.dim() == 2 and y.dim() == 2
        n = x.size(0)
        m = y.size(0)

        # Guard: if too small, return 0 (no meaningful MMD)
        if n < 2 or m < 2:
            return torch.zeros((), device=x.device, dtype=x.dtype)

        k_xx = torch.zeros((n, n), device=x.device, dtype=x.dtype)
        k_yy = torch.zeros((m, m), device=x.device, dtype=x.dtype)
        k_xy = torch.zeros((n, m), device=x.device, dtype=x.dtype)

        for s in sigmas:
            s = float(s)
            k_xx = k_xx + cls._rbf_kernel(x, x, s)
            k_yy = k_yy + cls._rbf_kernel(y, y, s)
            k_xy = k_xy + cls._rbf_kernel(x, y, s)


        # Unbiased: exclude diagonals for xx/yy
        k_xx = (k_xx.sum() - k_xx.diag().sum()) / (n * (n - 1))
        k_yy = (k_yy.sum() - k_yy.diag().sum()) / (m * (m - 1))
        k_xy = k_xy.mean()

        mmd2 = k_xx + k_yy - 2.0 * k_xy
        return mmd2


    def get_batch_loss(self, batch_dict: Dict[str, Any]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        att_mask = batch_dict['att_mask']

        # ---- AE forward ----
        z, logits = self.model(batch_dict['input_ids'], att_mask)

        # ---- Reconstruction loss (token-level CE) ----
        ce = F.cross_entropy(
            logits.transpose(1, 2), # F.cross_entropy requires (B, V, L)
            batch_dict['tgt_ids'],
            ignore_index=self.model_cfg['pad_token_id'],
            reduction='mean', 
            label_smoothing=self.loss_cfg.get('label_smoothing', 0.0)
        )

        z_valid = z[att_mask == 1]                                          # (N_valid, d)
        # ----- MMD -----
        max_mmd_points = 2000
        if z_valid.numel() > 0:
            if z_valid.size(0) > max_mmd_points:
                idx = torch.randperm(z_valid.size(0), device=z_valid.device)[:max_mmd_points]
                z_valid_mmd = z_valid[idx]
            else:
                z_valid_mmd = z_valid
            z_prior = torch.randn_like(z_valid_mmd)                         # (N_valid, d)
            mmd2 = self._mmd_rbf_multi_kernel(z_valid_mmd, z_prior, 
                                              sigmas=[0.2, 0.5, 1.0, 2.0, 5.0])
        else:
            mmd2 = torch.zeros((), device=z.device, dtype=z.dtype)
        
        # ---- L2 ----
        if z_valid.numel() > 0:
            z_l2 = (z_valid ** 2).mean()
        else:
            z_l2 = torch.zeros((), device=z.device, dtype=z.dtype)

        # ---- Total ----
        lambda_mmd = self.loss_cfg.get('lambda_mmd', 0.0)
        lambda_l2 = self.loss_cfg.get('lambda_l2', 0.0)
        loss = ce + lambda_l2 * z_l2 + lambda_mmd * mmd2

        metrics = {"CE": ce, "MMD": mmd2, "L2": z_l2}
        return loss, metrics
    
    def preview_validation_text(self, val_loader, tokenizer, ckpt_dir) -> None:
        """Generate a qualitative text preview during validation."""
        if not self.is_main_process:
            return None

        # ---- Sample one batch ----
        preview_batch = next(iter(val_loader))
        batch_dict = self.get_batch_dict({
            'input_ids': preview_batch["input_ids"][:1], 'att_mask': preview_batch["att_mask"][:1]
        })
        input_ids = batch_dict["input_ids"]
        att_mask = batch_dict["att_mask"]
        tgt_ids = batch_dict['tgt_ids']

        # ---- AE Forward ----
        with torch.no_grad():
            _, logits = self.model(input_ids, att_mask)

        # ---- Decode ----
        true_text = tokenizer.decode(tgt_ids[0].tolist(), skip_special_tokens=False)
        masked_text = tokenizer.decode(input_ids[0].tolist(), skip_special_tokens=False)
        pred_ids = logits.argmax(dim=-1)[0] # (L,)
        pred_text = tokenizer.decode(pred_ids.tolist(), skip_special_tokens=False)

        # ---- Write preview to a separate file ----
        preview_path = os.path.join(ckpt_dir, "preview_ae.txt")

        with open(preview_path, "a", encoding="utf-8") as f:
            f.write(f"\n\n==================== Step {self.global_step} ====================\n\n")
            f.write("▶︎ Ground Truth ◀\n\n")
            f.write(true_text[:10000] + ("..." if len(true_text) > 10000 else "") + "\n\n")
            f.write("▶︎ Corrupted Text ◀\n\n")
            f.write(masked_text[:10000] + ("..." if len(masked_text) > 10000 else "") + "\n\n")
            f.write("▶︎ YAN Reconstruction ◀\n\n")
            f.write(pred_text[:10000] + ("..." if len(pred_text) > 10000 else "") + "\n")
