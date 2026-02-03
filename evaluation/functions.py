from __future__ import annotations
import os
import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import random
import numpy as np
from typing import Any, Dict, Tuple, List, cast
from datasets import load_from_disk, DatasetDict
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import Dataset, DataLoader, Subset
from dataclasses import dataclass
from abc import ABC, abstractmethod
import logging
from logging import Logger
from torch.nn.parallel import DistributedDataParallel as DDP
import re
import evaluate
logging.getLogger("transformers").setLevel(logging.ERROR)
from collections import Counter
from transformers import AutoTokenizer
import matplotlib.pyplot as plt

def set_seed(seed: int = 42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def matplotlib_style():
    from matplotlib import font_manager
    font_manager.fontManager.addfont("fonts/texgyrepagella-regular.otf")
    font_manager.fontManager.addfont("fonts/texgyrepagella-bold.otf")
    font_manager.fontManager.addfont("fonts/texgyrepagella-italic.otf")
    font_manager.fontManager.addfont("fonts/texgyrepagella-bolditalic.otf")

    plt.style.use("tableau-colorblind10")
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Palatino", "TeX Gyre Pagella"],
        "mathtext.fontset": "stix",
    })



class DataProcessor:
    def __init__(self, yan_tokenizer, dp_config: Dict[str, Any], device: torch.device, 
                 other_tokenizer, seed: int = 42):
        self.yan_tokenizer = yan_tokenizer
        self.dp_config = dp_config
        self.device = device
        
        self.other_tokenizer = other_tokenizer
        self.other_mask_token = dp_config['other_mask_token']

        self.rng = torch.Generator(device=device)
        self.rng.manual_seed(int(seed))

    def generate_span_corruption(self, loader_batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Generate span corruption.
        Args:
            loader_batch: {'input_ids': (B, L), 'att_mask': (B, L)}.
        Returns:
            { 'input_ids': (B, L),      # masked collapsed-span ids
              'tgt_ids':   (B, L),      # deep copy of original input_ids
              'att_mask':  (B, L) }     # original att_mask
        """
        device = self.device
        mask_token_id = self.yan_tokenizer.mask_token_id
        pad_token_id = self.yan_tokenizer.pad_token_id
        bos_token_id = self.yan_tokenizer.bos_token_id
        eos_token_id = self.yan_tokenizer.eos_token_id

        input_ids = loader_batch["input_ids"].to(device)
        att_mask = loader_batch["att_mask"].to(device)
        tgt_ids = input_ids.clone()

        mask_ratio = self.dp_config['mask_ratio']
        remove_answer_prob = self.dp_config['remove_answer_prob']
        corrupt_span_prob = self.dp_config['corrupt_span_prob']
        max_span_len = self.dp_config['max_span_len']

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

            # ---- Step 1: optionally remove answer part BEFORE any infilling decision ----
            if remove_answer_prob > 0.0 and (torch.rand((), device=device, generator=self.rng).item() < remove_answer_prob):
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
            do_infill = (corrupt_span_prob > 0.0) and (torch.rand((), device=device, generator=self.rng).item() < corrupt_span_prob)
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
                start_idx = starts[torch.randint(0, starts.numel(), (1,), device=device, generator=self.rng)].item()

                # Sample a span length uniformly in [1, max_span_len], then clip to boundaries.
                span_len = int(torch.randint(1, max_span_len + 1, (1,), device=device, generator=self.rng).item())

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

            # Ensure <bos> ... <eos> after corruption too
            new_seq_t = torch.tensor(new_seq, dtype=input_ids.dtype, device=device)
            new_seq_t = _ensure_bos_eos(new_seq_t)

            new_seq_list = new_seq_t.tolist()[:L]
            out_ids[b, :len(new_seq_list)] = torch.tensor(new_seq_list, dtype=input_ids.dtype, device=device)

        return {
            'input_ids': out_ids,   # (B, L) corrupted token ids
            'tgt_ids': tgt_ids,     # (B, L) original clean token ids deep copy
            'att_mask': att_mask    # (B, L) original attention mask
        }

    def split_enc_tgt_qa(self, loader_batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Split each sample in a batch into encoder ids (summary+question) and target ids (answer),
        using a fixed ANSWER header token pattern as the split marker.
        Args:
            loader_batch: {'input_ids': (B, L), 'att_mask': (B, L)}.
        Returns:
            { 'enc_ids':      (B, Lx), 
              'enc_att_mask': (B, Lx), 
              'tgt_ids':      (B, Ly), 
              'att_mask':     (B, Ly) }
        """
        bos_id = self.yan_tokenizer.bos_token_id
        eos_id = self.yan_tokenizer.eos_token_id
        pad_id = self.yan_tokenizer.pad_token_id

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
        assert self.yan_tokenizer is not None
        input_ids = loader_batch['input_ids']       # (B, L)
        input_att_mask = loader_batch['att_mask']   # (B, L)
        B, L = input_ids.shape
        
        # ---- 1) Decode input tokens into texts ----
        texts = []
        for b in range(B):
            valid_len = int(input_att_mask[b].sum().item())
            texts.append(input_ids[b, :valid_len].tolist())
        texts = self.yan_tokenizer.batch_decode(texts, skip_special_tokens=True)
        
        # ---- 2) Split last word and replace with mask token ----
        # E.g., "This is an example." -> prefix="This is an ", word="example", suffix="."
        #       enc="This is an <mask>.", tgt="example"
        enc_texts, tgt_texts = [], []
        pattern = re.compile(r"([A-Za-z]+)([^A-Za-z]*)$", flags=re.S)
        for text in texts:
            words = text.strip()
            m = pattern.search(text)

            if m is None:
                enc_texts.append(self.yan_tokenizer.mask_token)
                tgt_texts.append(self.yan_tokenizer.pad_token)
                continue

            last_word = m.group(1)
            suffix = m.group(2)
            prefix = text[:m.start(1)]

            enc_texts.append(f"{prefix}{self.yan_tokenizer.mask_token}{suffix}")
            tgt_texts.append(last_word)
        
        # ---- 3) Add <EOS> ----
        enc_texts = [t + self.yan_tokenizer.eos_token for t in enc_texts]
        tgt_texts = [t + self.yan_tokenizer.eos_token for t in tgt_texts]

        # ---- 4) Retokenize ----
        enc = self.yan_tokenizer(
            enc_texts, padding=True, truncation=True, return_tensors="pt", add_special_tokens=True
        )
        tgt = self.yan_tokenizer(
            tgt_texts, padding=True, truncation=True, return_tensors="pt", add_special_tokens=True
        )
        return {
            'enc_ids': enc['input_ids'].to(self.device),
            'enc_att_mask': enc['attention_mask'].to(self.device),
            'tgt_ids': tgt['input_ids'].to(self.device),
            'att_mask': tgt['attention_mask'].to(self.device)
        }


    def _cleanup_text(self, text: str) -> str:
        """Remove padding, bos, eos in text (but keep mask)"""
        if text is None:
            return ""
        text = text.replace(self.yan_tokenizer.bos_token, "")
        text = text.replace(self.yan_tokenizer.eos_token, "")
        text = text.replace(self.yan_tokenizer.pad_token, "")
        return text

    def _convert_pair(self, token_ids: torch.Tensor, att_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Retokenze a pair of (token_ids, att_mask) from yan-tokenizer to other-tokenizer: 
            1) Decode using yan-tokenizer.
            2) Replace yan <|mask|> with other mask token.
            3) Encode using other-tokenizer.
        Args:
            token_ids: (B, L)
            att_mask: (B, L)
        Returns:
            token_ids_converted: (B, L')
            att_mask_converted: (B, L')
        """
        B = token_ids.shape[0]

        # ---- 1) Decode valid parts using yan tokenizer ----
        texts = []
        for b in range(B):
            real_len = int(att_mask[b].sum().item())
            if real_len <= 0:
                texts.append("")
                continue
            text = self.yan_tokenizer.decode(
                token_ids[b, :real_len].tolist(), skip_special_tokens=False
            )
            texts.append(self._cleanup_text(text))

        # ---- 2) Replace yan mask string with other mask string ----
        yan_mask = self.yan_tokenizer.mask_token
        if yan_mask != self.other_mask_token:
            texts = [t.replace(yan_mask, self.other_mask_token) for t in texts]

        # ---- 3) Tokenize with other tokenizer ----
        re_tokenized = self.other_tokenizer(
            texts, padding=True, truncation=True, max_length=self.dp_config['max_length'],
            return_tensors="pt", add_special_tokens=True
        )

        # ---- Move to device ----
        token_ids_converted = re_tokenized["input_ids"].to(self.device)
        att_mask_converted = re_tokenized["attention_mask"].to(self.device) 

        return token_ids_converted, att_mask_converted


    def convert_to_other(self, batch_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Convert yan-tokenizer tokenized batch_dict to other-tokenizer tokenized batch_dict. The 
        key-valud structrue of the batch_dict will keep the same."""
        assert self.other_tokenizer is not None and self.other_mask_token is not None
        assert self.dp_config['split'] in ['infill', 'qa', 'last']

        device = self.device

        if self.dp_config['split'] == 'infill':
            # ---- Get tensors ----
            yan_enc_ids  = batch_dict["input_ids"].to(device)
            yan_att_mask = batch_dict["att_mask"].to(device)
            yan_tgt_ids  = batch_dict["tgt_ids"].to(device)
            B = yan_enc_ids.shape[0]

            # ---- Convert ----
            enc_ids, _ = self._convert_pair(yan_enc_ids, yan_att_mask)
            tgt_ids, tgt_att_mask = self._convert_pair(yan_tgt_ids, yan_att_mask)

            # ---- Align enc_ids length to tgt_ids length ----
            enc_len = enc_ids.shape[1]
            tgt_len = tgt_ids.shape[1]

            if enc_len < tgt_len:
                pad_len = tgt_len - enc_len
                pad_id = self.other_tokenizer.pad_token_id
                pad = torch.full((B, pad_len), pad_id, dtype=enc_ids.dtype, device=device)

                if self.other_tokenizer.padding_side == "left":
                    enc_ids = torch.cat([pad, enc_ids], dim=1)
                else:  # right padding
                    enc_ids = torch.cat([enc_ids, pad], dim=1)
            elif enc_len > tgt_len:
                # Truncate enc_ids to match tgt_len, respecting padding_side
                if self.other_tokenizer.padding_side == "left":
                    # keep the rightmost tgt_len tokens (drop extra left tokens)
                    enc_ids = enc_ids[:, -tgt_len:]
                else:  # right padding
                    # keep the leftmost tgt_len tokens (drop extra right tokens)
                    enc_ids = enc_ids[:, :tgt_len]
            
            return {
                "input_ids": enc_ids,           # (B, L) corrupted token ids
                "att_mask": tgt_att_mask,       # (B, L) attention mask of clean tokens
                "tgt_ids": tgt_ids              # (B, L) clean token ids
            }
        
        else:
            # ---- Get tensors ----
            yan_enc_ids      = batch_dict['enc_ids'].to(device)
            yan_enc_att_mask = batch_dict['enc_att_mask'].to(device)
            yan_tgt_ids      = batch_dict['tgt_ids'].to(device)
            yan_tgt_att_mask = batch_dict['att_mask'].to(device)

            # ---- Convert ----
            enc_ids, enc_att_mask = self._convert_pair(yan_enc_ids, yan_enc_att_mask)
            tgt_ids, tgt_att_mask = self._convert_pair(yan_tgt_ids, yan_tgt_att_mask)

            return {
                "enc_ids":      enc_ids,            # (B, Lx)
                "enc_att_mask": enc_att_mask,       # (B, Lx)
                "tgt_ids":      tgt_ids,            # (B, Ly)
                "att_mask":     tgt_att_mask        # (B, Ly)
            }



### ---- Data Loader ---- ###
@dataclass
class YANDataCollator:
    pad_token_id: int
    
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        batch_size = len(features)
        lengths = [len(f["input_ids"]) for f in features]
        max_length = max(lengths)
        
        input_ids = torch.full(
            (batch_size, max_length), fill_value = self.pad_token_id, dtype = torch.long
        )
        att_mask = torch.zeros(
            (batch_size, max_length), dtype=torch.long
        )
        
        for i, feat in enumerate(features):
            L = len(feat["input_ids"])
            ids_i = torch.tensor(feat["input_ids"], dtype=torch.long)
            mask_i = torch.tensor(feat.get("att_mask", [1] * L), dtype=torch.long)

            input_ids[i, :L] = ids_i
            att_mask[i, :L] = mask_i
    
        return {'input_ids': input_ids, 'att_mask': att_mask}


class YANDataLoader:
    def __init__(self, loader_cfg: Dict[str, Any] | None = None, 
                 batch_size: int = 2, shuffle: bool = True, drop_last: bool = True, 
                 max_batches: Dict | None = None, num_workers: int = 0, 
                 pin_memory: bool = False, subset_seed: int | None = None):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.max_batches = max_batches
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.subset_seed = subset_seed

        # Override by loader_cfg
        if loader_cfg is not None:
            self.batch_size = loader_cfg.get("batch_size", self.batch_size)
            self.shuffle = loader_cfg.get("shuffle", self.shuffle)
            self.drop_last = loader_cfg.get("drop_last", self.drop_last)
            self.max_batches = loader_cfg.get("max_batches", self.max_batches)
            self.num_workers = loader_cfg.get("num_workers", self.num_workers)
            self.pin_memory = loader_cfg.get("pin_memory", self.pin_memory)
            self.subset_seed = loader_cfg.get("subset_seed", self.subset_seed)
    
    def create_loader(self, data_dir: str, tokenizer) -> Tuple[DataLoader | None, ...]:
        """
        Args:
            data_dir: The directory of the tokenized datasets.
            tokenizer: The tokenizer used to tokenize the dataset.
        Returns:
            (train_loader, val_loader, test_loader)
        """
        tokenized_datasets = cast(DatasetDict, load_from_disk(data_dir))
        data_collator = YANDataCollator(pad_token_id=tokenizer.pad_token_id)
        self.data_collator = data_collator
        
        loaders = {}
        
        for split in ['train', 'validation', 'test']:
            if split not in tokenized_datasets:
                continue

            full_dataset = tokenized_datasets[split]
            dataset_to_load = cast(Dataset, full_dataset)

            n_batches = None
            if self.max_batches is not None:
                n_batches = self.max_batches.get(split, None)

            if n_batches is not None:
                max_samples = min(n_batches * self.batch_size, len(full_dataset))

                if split == 'train' and self.subset_seed is not None:
                    g = torch.Generator()
                    g.manual_seed(int(self.subset_seed))
                    perm = torch.randperm(len(full_dataset), generator=g)[:max_samples]
                    indices = perm.tolist()
                else:
                    indices = list(range(max_samples))

                full_dataset = cast(Dataset, full_dataset)
                dataset_to_load = Subset(full_dataset, indices)

            loaders[split] = DataLoader(
                dataset_to_load,
                collate_fn=data_collator,
                batch_size=self.batch_size,
                shuffle=(split == "train" and self.shuffle),
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                drop_last=self.drop_last
            )

        return loaders.get('train'), loaders.get('validation'), loaders.get('test')






### ---- Trainer ---- ###
class Trainer(ABC):
    """Trainer class
    Args:
        model: model to train
        device: device the train the model
        dp_config: data processor configuration
        trainer_cfg: trainer configuration
        yan_tokenizer: yan tokenizer
        other_tokenizer: other tokenizer
        model_cfg (optional): model configuration
        loss_cfg (optional): loss configuration
        logger (optional): logger to output training process in .log file
        train_sampler (optional): DistributedSamplers when using DDP
        is_main_process (optional): whether it is the main process when using DDP
    """
    def __init__(self, model: nn.Module | DDP, device: torch.device, 
                 dp_config: Dict[str, Any], trainer_cfg: Dict[str, Any], 
                 yan_tokenizer, other_tokenizer,
                 model_cfg: Dict[str, Any] | None = None, 
                 loss_cfg: Dict[str, Any] | None = None, 
                 logger: Logger | None = None, train_sampler = None, is_main_process: bool = True):
        super().__init__()
        self.model = model
        self.device = device
        self.dp = DataProcessor(yan_tokenizer, dp_config, device, other_tokenizer)

        self.model_cfg = model_cfg
        self.loss_cfg = loss_cfg
        self.trainer_cfg = trainer_cfg
        self.dp_config = dp_config
        
        self.yan_tokenizer = yan_tokenizer
        self.other_tokenizer = other_tokenizer

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
        assert self.dp_config['split'] in ['qa', 'infill', 'last']
        if self.dp_config['split'] == 'qa':                         # question answering
            batch = self.split_enc_tgt_qa(loader_batch)
        if self.dp_config['split'] == 'infill':                     # infill
            batch = self.generate_span_corruption(loader_batch)
        if self.dp_config['split'] == 'last':                       # last word
            batch = self.split_enc_tgt_last(loader_batch)
        batch = self.convert_to_other(batch)                        # convert to current tokenizer # type: ignore
        return {key: tensor.to(self.device, non_blocking=True) for key, tensor in batch.items()}
 
    @abstractmethod
    def get_batch_loss(self, batch_dict: Dict[str, Any]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        pass

    def preview_validation_text(self, val_loader, tokenizer, ckpt_dir) -> None:
        pass

    def generate_span_corruption(self, loader_batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return self.dp.generate_span_corruption(loader_batch)
    
    def split_enc_tgt_qa(self, loader_batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return self.dp.split_enc_tgt_qa(loader_batch)

    def split_enc_tgt_last(self, loader_batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return self.dp.split_enc_tgt_last(loader_batch)
        
    def convert_to_other(self, batch_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return self.dp.convert_to_other(batch_dict)


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
                    if self.other_tokenizer is not None and save_every is not None:
                        self.preview_validation_text(val_loader, self.other_tokenizer, ckpt_dir)
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




### ---- Logging ---- ###
def setup_logging(log_dir: str) -> Logger:
    """Set up a logger that prints to the console and saves to a file
    Args:
        log_dir: directory to save the log file in
    """
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, 'training.log')
    logger = logging.getLogger('Trainer')
    logger.setLevel(logging.INFO)
    if logger.hasHandlers():
        logger.handlers.clear()
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger





### --- Evaluator ---- ###
class Evaluator:
    def __init__(self):
        self.rouge = evaluate.load('rouge')
        self.bertscore = evaluate.load("bertscore")

    def compute_rouge(self, predictions: List[str], references: List[str]) -> Dict[str, float]:
        """Compute ROUGE-1, ROUGE-2, ROUGE-L"""
        assert len(predictions) == len(references)
        scores = self.rouge.compute(predictions=predictions, references=references)
        assert isinstance(scores, dict)
        return {'ROUGE-1': scores['rouge1'], 'ROUGE-2': scores['rouge2'], 'ROUGE-L': scores['rougeL']}

    def compute_exact_match(self, predictions: List[str], references: List[str]) -> Dict[str, float]:
        """Compute Exact Match (EM) score. 
        EM = percentge of predictions that exactly match the references after normalization.
        """
        assert len(predictions) == len(references)
        def normalize(text: str) -> str:
            return " ".join(text.strip().lower().split())
        exact_matches = 0
        for pred, ref in zip(predictions, references):
            exact_matches += int(normalize(pred) == normalize(ref))
        return {'EM': exact_matches / max(len(predictions), 1)}

    def compute_token_f1(self, predictions: List[str], references: List[str]) -> Dict[str, float]:
        """Compute token-level F1 score"""
        assert len(predictions) == len(references)

        def tokenize(text: str) -> List[str]:
            return text.strip().lower().split()

        def f1_score(pred_tokens: List[str], ref_tokens: List[str]) -> float:
            if len(pred_tokens) == 0 and len(ref_tokens) == 0: return 1.0
            if len(pred_tokens) == 0 or len(ref_tokens) == 0: return 0.0

            common = Counter(pred_tokens) & Counter(ref_tokens)
            num_same = sum(common.values())
            if num_same == 0: return 0.0

            precision = num_same / len(pred_tokens)
            recall = num_same / len(ref_tokens)
            return 2 * precision * recall / (precision + recall)

        f1_scores = []
        for pred, ref in zip(predictions, references):
            f1_scores.append(f1_score(tokenize(pred), tokenize(ref)))

        return {"F1": sum(f1_scores) / len(f1_scores) if predictions else 0.0}

    def compute_bertscore(self, predictions: List[str], references: List[str]) -> Dict[str, float]:
        """Compute BERTScore-F1 using roberta-large with baseline rescaling"""
        assert len(predictions) == len(references)
        scores = self.bertscore.compute(predictions=predictions, references=references, lang="en",
                                        model_type="roberta-large", rescale_with_baseline=True)
        assert isinstance(scores, dict)
        return {"BERTScore-F1": sum(scores["f1"]) / len(scores["f1"])}

    def evaluate_all_metrics(self, predictions: List[str], references: List[str]) -> Dict[str, float]:
        assert len(predictions) == len(references)
        rouge = self.compute_rouge(predictions, references)
        em = self.compute_exact_match(predictions, references)
        f1 = self.compute_token_f1(predictions, references)
        bertscore = self.compute_bertscore(predictions, references)
        return rouge | em | f1 | bertscore



class EvaluatorDiversity:
    """
    Diversity metrics for a *set* of K candidates generated from the same source input.

    Metrics:
      - Self-BLEU: lower => more diverse
      - distinct-1: higher => more diverse
      - distinct-2: higher => more diverse
      - SemanticDist (avg pairwise cosine distance): higher => more diverse

    """

    def __init__(
        self,
        embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        embedding_device: str | None = None,
    ):
        # --- Self-BLEU (sacrebleu) ---
        try:
            import sacrebleu  # type: ignore
            self.sacrebleu = sacrebleu
        except Exception as e:
            raise ImportError(
                "EvaluatorDiversity requires `sacrebleu` for Self-BLEU. "
                "Install it via: pip install sacrebleu"
            ) from e

        # --- Semantic distance (transformers) ---
        try:
            from transformers import AutoTokenizer, AutoModel  # type: ignore
        except Exception as e:
            raise ImportError(
                "EvaluatorDiversity requires `transformers` for semantic distance. "
                "Install it via: pip install transformers"
            ) from e

        self.embedding_model_name = embedding_model_name
        if embedding_device is None:
            embedding_device = "cuda" if torch.cuda.is_available() else "cpu"
        self.embedding_device = embedding_device

        self.emb_tokenizer = AutoTokenizer.from_pretrained(embedding_model_name)
        self.emb_model = AutoModel.from_pretrained(embedding_model_name)
        self.emb_model.eval()
        self.emb_model.to(self.embedding_device)

    # -------------------------
    # Tokenization helper
    # -------------------------
    def _simple_tokenize(self, text: str) -> List[str]:
        # basic, deterministic tokenizer (keeps punctuation as tokens)
        return re.findall(r"\w+|[^\w\s]", text.lower(), flags=re.UNICODE)

    # -------------------------
    # distinct-1 (corpus-level)
    # -------------------------
    def compute_distinct_1(self, candidates: List[str]) -> Dict[str, float]:
        """
        Corpus-level distinct-1 over the candidate set.
        distinct-1 = (# unique unigrams) / (# total unigrams)
        """
        total = 0
        uniq = set()

        for s in candidates:
            toks = self._simple_tokenize(s)
            for tok in toks:
                total += 1
                uniq.add(tok)

        score = (len(uniq) / total) if total > 0 else 0.0
        return {"distinct-1": score}

    # -------------------------
    # distinct-2 (corpus-level)
    # -------------------------
    def compute_distinct_2(self, candidates: List[str]) -> Dict[str, float]:
        """
        Corpus-level distinct-2 over the candidate set.
        distinct-2 = (# unique bigrams) / (# total bigrams)
        """
        total = 0
        uniq = set()

        for s in candidates:
            toks = self._simple_tokenize(s)
            if len(toks) < 2:
                continue
            for bg in zip(toks[:-1], toks[1:]):
                total += 1
                uniq.add(bg)

        score = (len(uniq) / total) if total > 0 else 0.0
        return {"distinct-2": score}

    # -------------------------
    # Self-BLEU
    # -------------------------
    def compute_self_bleu(self, candidates: List[str]) -> Dict[str, float]:
        """
        Self-BLEU: For each candidate i, compute BLEU(candidate_i, refs=others), then average.
        Lower => more diverse.
        Returns score in [0, 100] (sacrebleu convention).
        """
        K = len(candidates)
        if K <= 1:
            return {"Self-BLEU": float("nan")}

        scores = []
        for i in range(K):
            hyp = candidates[i]
            refs = [candidates[j] for j in range(K) if j != i]
            s = self.sacrebleu.sentence_bleu(hypothesis=hyp, references=refs).score
            scores.append(s)

        return {"Self-BLEU": sum(scores) / len(scores)}

    # -------------------------
    # Semantic distance
    # -------------------------
    @torch.no_grad()
    def _embed_texts(self, texts: List[str]) -> torch.Tensor:
        """
        Mean-pool token embeddings (masked), then L2-normalize.
        Returns: (K, D)
        """
        tok = self.emb_tokenizer(
            texts, padding=True, truncation=True, return_tensors="pt"
        ).to(self.embedding_device)

        out = self.emb_model(**tok)  # last_hidden_state: (K, T, D)
        hidden = out.last_hidden_state
        attn = tok["attention_mask"].unsqueeze(-1)  # (K, T, 1)

        pooled = (hidden * attn).sum(dim=1) / attn.sum(dim=1).clamp(min=1)
        pooled = F.normalize(pooled, p=2, dim=-1)
        return pooled

    def compute_semantic_distance(self, candidates: List[str]) -> Dict[str, float]:
        """
        Avg pairwise cosine distance among candidates.
        Higher => more diverse.
        """
        K = len(candidates)
        if K <= 1:
            return {"SemanticDist": float("nan")}

        emb = self._embed_texts(candidates)  # (K, D)
        sim = emb @ emb.T                    # (K, K) cosine sim
        dist = 1.0 - sim                     # cosine distance

        idx = torch.triu_indices(K, K, offset=1, device=dist.device)
        mean_dist = dist[idx[0], idx[1]].mean().item()
        return {"SemanticDist": mean_dist}

    # -------------------------
    # Convenience: all diversity metrics
    # -------------------------
    def evaluate_diversity(self, candidates: List[str]) -> Dict[str, float]:
        """
        Evaluate diversity metrics for a candidate set (K strings).
        """
        self_bleu = self.compute_self_bleu(candidates)
        dist1 = self.compute_distinct_1(candidates)
        dist2 = self.compute_distinct_2(candidates)
        sem = self.compute_semantic_distance(candidates)
        return self_bleu | dist1 | dist2 | sem





### --- Helper functions ---- ###
def clean_texts(text_list, BOS: str = "<|begin_of_text|>", EOS: str = "<|end_of_text|>"):
    cleaned = []
    for text in text_list:
        if text.startswith(BOS):
            text = text[len(BOS):]
        eos_index = text.find(EOS)
        if eos_index != -1:
            text = text[:eos_index]
        text = text.strip()
        cleaned.append(text)
    return cleaned

def count_tokens_until_eos(gen_ids, eos_id):
    """Count the number of tokens before and including EOS"""
    B, L = gen_ids.shape
    eos_mask = gen_ids == eos_id

    lengths = torch.where(
        eos_mask.any(dim=1),
        eos_mask.float().argmax(dim=1) + 1,
        torch.full((B,), L, device=gen_ids.device)
    )
    return lengths

def load_yan_tokenizer():
    yan_tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B", use_fast=True)
    yan_tok.add_special_tokens({'pad_token': '<|padding|>'})
    yan_tok.add_special_tokens({'mask_token': '<|mask|>'})
    return yan_tok

def get_dp_config():
    dp_config = {
        # don't change
        'mask_ratio': 0.08,
        'remove_answer_prob': 0.0,
        'corrupt_span_prob': 1.0,
        'max_span_len': 3,
        # placeholder
        'other_mask_token': None,
        'max_length': None,
        'split': None
    }
    return dp_config

def open_batch_by_task(task: str, batch: Dict[str, torch.Tensor]):
    assert task in ['last', 'qa', 'infill']
    if task in ['last', 'qa']:
        enc_ids, enc_att_mask = batch['enc_ids'], batch['enc_att_mask']    
        tgt_ids, tgt_att_mask = batch['tgt_ids'], batch['att_mask']
    else:
        enc_ids = batch['input_ids']
        tgt_ids = batch['tgt_ids']
        tgt_att_mask = enc_att_mask = batch['att_mask']
    return enc_ids, enc_att_mask, tgt_ids, tgt_att_mask

