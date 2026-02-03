# trainer/YANData.py

import datasets
from datasets import load_dataset, load_from_disk, DatasetDict, concatenate_datasets
from torch.utils.data import Dataset, DataLoader, Subset
import os
from typing import Any, Dict, List, Optional, Tuple, cast
from dataclasses import dataclass
import torch
from transformers import AutoTokenizer
from rich.console import Console

# ================================================================================== #
#   This script contains the functions to download and prepare the data.             #
# ================================================================================== #


# --------------- Helper functions --------------- #
def merge_datasets(datasets: List[DatasetDict], seed: int = 42) -> DatasetDict:
    """Merge datasets into one dataset"""
    all_splits = set().union(*(ds.keys() for ds in datasets))
    merged = {}
    for split in all_splits:
        to_concat = [ds[split] for ds in datasets if split in ds]
        merged_split = concatenate_datasets(to_concat)
        merged[split] = merged_split.shuffle(seed=seed)
    return DatasetDict(merged)


def change_key_name(dataset: DatasetDict, 
                    name_from: str = 'story', name_to: str = 'text') -> DatasetDict:
    """Change the key column name of the text and remove all other columns"""
    for split in dataset.keys():
        # 1) Change key name
        dataset[split] = dataset[split].rename_column(name_from, name_to)
        # 2) Remove all other columns
        cols = dataset[split].column_names
        cols_to_remove = [c for c in cols if c != "text"]
        dataset[split] = dataset[split].remove_columns(cols_to_remove)
    return dataset

def split_train_val_test(dataset: DatasetDict, val_ratio: float = 0.01, 
                         test_ratio: float = 0.01, seed: int = 42) -> DatasetDict:
    """Split the dataset into train / validation / test"""
    # 1) Merge all split into one and get the total number of samples
    dataset_merged = concatenate_datasets(list(dataset.values()))
    n_tot = dataset_merged.num_rows
    n_val = int(val_ratio * n_tot)
    n_test = int(test_ratio * n_tot)
    
    # 2) Split dataset
    train_val, test = dataset_merged.train_test_split(test_size=n_test, seed=seed).values()
    train, val = train_val.train_test_split(test_size=n_val, seed=seed).values()

    dataset_split = DatasetDict({"train": train, "validation": val, "test": test})
    return dataset_split


# --------------- Helper functions to process narrativeqa dataset --------------- #
def _distinct_answer_texts(answers: List[Dict[str, Any]]) -> List[str]:
    """
    Extract distinct answer strings from a list of answer dicts.
    Preserves first-seen order (stable de-dup).
    """
    seen = set()
    out = []
    for a in answers or []:
        t = a.get("text", "")
        if not isinstance(t, str):
            continue
        t = t.strip()
        if not t:
            continue
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


def _build_text(summary_text: str, question_text: str, answer_text: str) -> str:
    return (
        "####### SUMMARY #######\n\n"
        f"{summary_text}\n\n"
        "####### QUESTION #######\n\n"
        f"{question_text}\n\n"
        "####### ANSWER #######\n\n"
        f"{answer_text}"
    )


def preprocess_narrativeqa_to_text(
    ds: DatasetDict,
    *,
    keep_empty_answers: bool = False,
    drop_if_missing_summary_or_question: bool = True,
) -> DatasetDict:
    """
    Convert deepmind/narrativeqa DatasetDict with features:
      - document.summary.text (str)
      - question.text (str)
      - answers: list[dict] with 'text' (str)
    into a new DatasetDict containing only:
      - text (str)

    For each original example, we emit one output row per DISTINCT answer text.
    """

    def expand_split(split_ds) -> Dataset:
        texts: List[str] = []

        for ex in split_ds:
            # 1) summary text
            doc = ex.get("document") or {}
            summ = (doc.get("summary") or {}) if isinstance(doc, dict) else {}
            summary_text = summ.get("text", "") if isinstance(summ, dict) else ""
            if not isinstance(summary_text, str):
                summary_text = ""
            summary_text = summary_text.strip()
            # if len(summary_text) > 8000:
            #     summary_text = summary_text[:8000]
            

            # 2) question text
            q = ex.get("question") or {}
            question_text = q.get("text", "") if isinstance(q, dict) else ""
            if not isinstance(question_text, str):
                question_text = ""
            question_text = question_text.strip()

            if drop_if_missing_summary_or_question and (not summary_text or not question_text):
                continue

            # 3) answers (distinct)
            answers = ex.get("answers")
            if not isinstance(answers, list):
                answers = []

            ans_texts = _distinct_answer_texts(answers)

            if not ans_texts:
                if keep_empty_answers:
                    texts.append(_build_text(summary_text, question_text, ""))
                continue
            
            shortest_ans = min(ans_texts, key=len)
            texts.append(_build_text(summary_text, question_text, shortest_ans))

        return datasets.Dataset.from_dict({"text": texts}) # type: ignore

    out = {}
    for split in ds.keys():
        out[split] = expand_split(ds[split])

    return DatasetDict(out)



# --------------- Helper functions to process squad dataset --------------- #
def _distinct_answer_texts_squad(answers: Dict[str, Any]) -> List[str]:
    """
    SQuAD answers format: 
        answers: {"text": [str, ...], "answer_start": [int, ...]}
    Return distinct answer strings.
    """
    if not isinstance(answers, dict):
        return []

    texts = answers.get("text", [])
    if not isinstance(texts, list):
        return []

    seen = set()
    out: List[str] = []
    for t in texts:
        if not isinstance(t, str):
            continue
        t = t.strip()
        if not t:
            continue
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out

def preprocess_squad_to_text(
    ds: DatasetDict,
    *,
    keep_empty_answers: bool = False,
    drop_if_missing_summary_or_question: bool = True,
    max_summary_chars: int = 20000
) -> DatasetDict:
    """
    Convert SQuAD-style DatasetDict with features:
      - context (str)
      - question (str)
      - answers: dict with "text" (list[str]) and "answer_start" (list[int])
    into a new DatasetDict containing only:
      - text (str)
    By default, emits one row per example, choosing the shortest distinct answer text
    """

    def expand_split(split_ds) -> Dataset:
        texts: List[str] = []

        for ex in split_ds:
            # 1) summary text (from context)
            summary_text = ex.get("context", "")
            if not isinstance(summary_text, str):
                summary_text = ""
            summary_text = summary_text.strip()
            if max_summary_chars is not None and len(summary_text) > max_summary_chars:
                summary_text = summary_text[:max_summary_chars]

            # 2) question text
            question_text = ex.get("question", "")
            if not isinstance(question_text, str):
                question_text = ""
            question_text = question_text.strip()

            if drop_if_missing_summary_or_question and (not summary_text or not question_text):
                continue

            # 3) answers (distinct)
            answers = ex.get("answers", {})
            ans_texts = _distinct_answer_texts_squad(answers)

            if not ans_texts:
                if keep_empty_answers:
                    texts.append(_build_text(summary_text, question_text, ""))
                continue

            shortest_ans = min(ans_texts, key=len)
            texts.append(_build_text(summary_text, question_text, shortest_ans))

        return datasets.Dataset.from_dict({"text": texts})  # type: ignore

    out: Dict[str, Dataset] = {}
    for split in ds.keys():
        assert isinstance(split, str)
        out[split] = expand_split(ds[split])

    return DatasetDict(out) # type: ignore


# --------------- Helper functions to process rocstories dataset --------------- #
def preprocess_rocstories(ds: DatasetDict, sep: str = " ") -> DatasetDict:
    """Concatenate five sentencies in rocstories into 'text' and keep 'text' column only,"""
    def concat_example(example):
        sentences = [
            example["sentence1"],
            example["sentence2"],
            example["sentence3"],
            example["sentence4"],
            example["sentence5"],
        ]
        example["text"] = sep.join(sentences)
        return example
    new_ds = DatasetDict()
    for split in ds.keys():
        new_ds[split] = ds[split].map(concat_example, remove_columns=ds[split].column_names)
    return new_ds

# --------------- Helper functions to process xsum dataset --------------- #
def preprocess_xsum_to_text(ds: DatasetDict) -> DatasetDict:
    """
    Convert EdinburghNLP/xsum DatasetDict with features:
      - document (str)
      - summary (str)
      - id (str)
    into a new DatasetDict containing only:
      - text (str)
    """

    def build_text(document_text: str, answer_text: str) -> str:
        return (
            "####### DOCUMENT #######\n\n"
            f"{document_text}\n\n"
            "####### SUMMARY #######\n\n"
            f"{answer_text}"
        )

    def process_split(split_ds) -> Dataset:
        texts = []

        for ex in split_ds:
            doc = ex.get("document", "")
            ans = ex.get("summary", "")

            if not isinstance(doc, str):
                doc = ""
            if not isinstance(ans, str):
                ans = ""

            doc = doc.strip()
            ans = ans.strip()

            if not doc or not ans:
                continue

            texts.append(build_text(doc, ans))

        return datasets.Dataset.from_dict({"text": texts})  # type: ignore

    out = {}
    for split in ds.keys():
        out[split] = process_split(ds[split])

    return DatasetDict(out)




# --------------- Helper functions to process babiqa dataset --------------- #
def preprocess_babiqa_to_text(
    ds: DatasetDict,
    *,
    story_field: str = "story",
    statement_type_value: int = 0,
    question_type_value: int = 1,
    keep_empty_answers: bool = False,
    drop_if_missing_context_or_question: bool = True,
    context_joiner: str = " ",
    max_context_sentences: Optional[int] = None,
) -> DatasetDict:
    """
    Convert habanoz/babi_qa_en_valid_10k_qa1-like DatasetDict (features: ['story'])
    into a new DatasetDict with only:
      - text (str)

    Each 'story' contains lists: 'type', 'text', 'answer' (and others).
    We emit one output example per question (type == question_type_value).

    Context (SUMMARY) is built from all statements (type == statement_type_value)
    that occurred before the question. Optionally cap context to last N sentences.
    """

    def _safe_list(x: Any) -> List[Any]:
        return x if isinstance(x, list) else []

    def expand_split(split_ds: Dataset) -> Dataset:
        out_texts: List[str] = []

        for ex in split_ds:
            story = ex.get(story_field)
            if not isinstance(story, dict):
                continue

            types = _safe_list(story.get("type"))
            texts = _safe_list(story.get("text"))
            answers = _safe_list(story.get("answer"))

            # need aligned lengths at least for type/text; answer may be shorter in some messy cases
            n = min(len(types), len(texts))
            if n == 0:
                continue

            context_sents: List[str] = []

            for i in range(n):
                t = types[i]
                line = texts[i]
                if not isinstance(line, str):
                    line = ""
                line = line.strip()

                if t == statement_type_value:
                    if line:
                        context_sents.append(line)
                    continue

                if t == question_type_value:
                    question_text = line
                    answer_text = ""
                    if i < len(answers) and isinstance(answers[i], str):
                        answer_text = answers[i].strip()

                    # build context text
                    ctx = context_sents
                    if max_context_sentences is not None:
                        ctx = ctx[-max_context_sentences:]
                    summary_text = context_joiner.join([s for s in ctx if s]).strip()

                    # filtering rules
                    if drop_if_missing_context_or_question and (not summary_text or not question_text):
                        continue
                    if (not answer_text) and (not keep_empty_answers):
                        continue

                    out_texts.append(_build_text(summary_text, question_text, answer_text))


        return datasets.Dataset.from_dict({"text": out_texts})  # type: ignore

    out: Dict[str, Dataset] = {}
    for split in ds.keys():
        out[split] = expand_split(ds[split])

    return DatasetDict(out)



# --------------- Helper functions to process agnews dataset --------------- #
def _build_agnews_text(document_text: str, answer_text: str) -> str:
    return (
        "####### DOCUMENT #######\n\n"
        f"{document_text}\n\n"
        "####### ANSWER #######\n\n"
        f"{answer_text}"
    )

def preprocess_agnews_to_text(
    ds: DatasetDict,
    *,
    drop_if_missing_text: bool = True,
) -> DatasetDict:
    """
    Convert fancyzhx/ag_news DatasetDict with features:
      - text (str)
      - label (int)
    into a new DatasetDict containing only:
      - text (str)

    Each original example -> exactly one output example.
    """
    AG_NEWS_LABEL_MAP = {
        0: "0 World",
        1: "1 Sports",
        2: "2 Business",
        3: "3 Sci/Tech",
    }

    def expand_split(split_ds: Dataset) -> Dataset:
        texts: List[str] = []

        for ex in split_ds:
            doc_text = ex.get("text", "")
            if not isinstance(doc_text, str):
                doc_text = ""
            doc_text = doc_text.strip()

            if drop_if_missing_text and not doc_text:
                continue

            label_id = ex.get("label")
            label_text = AG_NEWS_LABEL_MAP.get(label_id)

            if label_text is None:
                continue

            texts.append(
                _build_agnews_text(
                    document_text=doc_text,
                    answer_text=label_text,
                )
            )

        return datasets.Dataset.from_dict({"text": texts})  # type: ignore

    out: Dict[str, Dataset] = {}
    for split in ds.keys():
        out[split] = expand_split(ds[split])

    return DatasetDict(out)


# --------------- Helper functions to process dbpedia dataset --------------- #
def _build_dbpedia14_text(document_text: str, answer_text: str) -> str:
    return (
        "####### DOCUMENT #######\n\n"
        f"{document_text}\n\n"
        "####### ANSWER #######\n\n"
        f"{answer_text}"
    )


def preprocess_dbpedia14_to_text(
    ds: DatasetDict,
    *,
    join_title_and_content: bool = True,
    title_content_separator: str = "\n\n",
    drop_if_missing_text: bool = True,
) -> DatasetDict:
    """
    Convert fancyzhx/dbpedia_14 DatasetDict with features:
      - label (int)
      - title (str)
      - content (str)
    into a new DatasetDict containing only:
      - text (str)

    Each original example -> exactly one output example.
    """
    DBPEDIA14_LABEL_MAP = {
        0: "0 Company",
        1: "1 EducationalInstitution",
        2: "2 Artist",
        3: "3 Athlete",
        4: "4 OfficeHolder",
        5: "5 MeanOfTransportation",
        6: "6 Building",
        7: "7 NaturalPlace",
        8: "8 Village",
        9: "9 Animal",
        10: "10 Plant",
        11: "11 Album",
        12: "12 Film",
        13: "13 WrittenWork",
    }

    def expand_split(split_ds: Dataset) -> Dataset:
        out_texts: List[str] = []

        for ex in split_ds:
            title = ex.get("title", "")
            content = ex.get("content", "")
            if not isinstance(title, str):
                title = ""
            if not isinstance(content, str):
                content = ""
            title = title.strip()
            content = content.strip()

            if join_title_and_content:
                doc_text = (title + title_content_separator + content).strip()
            else:
                doc_text = content

            if drop_if_missing_text and not doc_text:
                continue

            label_id = ex.get("label")
            answer_text = DBPEDIA14_LABEL_MAP.get(label_id)
            if answer_text is None:
                continue

            out_texts.append(_build_dbpedia14_text(doc_text, answer_text))

        return datasets.Dataset.from_dict({"text": out_texts})  # type: ignore

    out: Dict[str, Dataset] = {}
    for split in ds.keys():
        out[split] = expand_split(ds[split])

    return DatasetDict(out)


# --------------- Helper functions to process sst2 dataset --------------- #
def _build_sst2_text(document_text: str, answer_text: str) -> str:
    return (
        "####### DOCUMENT #######\n\n"
        f"{document_text}\n\n"
        "####### ANSWER #######\n\n"
        f"{answer_text}"
    )

def preprocess_sst2_to_text(
    ds: DatasetDict,
    *,
    drop_if_missing_text: bool = True,
) -> DatasetDict:
    """
    Convert stanfordnlp/sst2 DatasetDict with features:
      - sentence (str)
      - label (int)
    into a new DatasetDict containing only:
      - text (str)

    Field 'idx' is dropped.
    Each original example -> exactly one output example.
    """
    SST2_LABEL_MAP = {
        0: "0 negative",
        1: "1 positive",
    }

    def expand_split(split_ds: Dataset) -> Dataset:
        out_texts: List[str] = []

        for ex in split_ds:
            sentence = ex.get("sentence", "")
            if not isinstance(sentence, str):
                sentence = ""
            sentence = sentence.strip()

            if drop_if_missing_text and not sentence:
                continue

            label_id = ex.get("label")
            answer_text = SST2_LABEL_MAP.get(label_id)
            if answer_text is None:
                continue

            out_texts.append(
                _build_sst2_text(
                    document_text=sentence,
                    answer_text=answer_text,
                )
            )

        return datasets.Dataset.from_dict({"text": out_texts})  # type: ignore

    out: Dict[str, Dataset] = {}
    for split in ds.keys():
        out[split] = expand_split(ds[split])

    return DatasetDict(out)


# --------------- Download dataset --------------- #
def download(name: str, val_ratio: float = 0.01, 
             test_ratio: float = 0.01, seed: int = 42) -> DatasetDict:
    """Download dataset"""
    assert name in ['finewiki', 'tinystories', 'simplestories', 'narrativeqa', 'fineweb', 
                    'squad', 'rocstories', 'xsum', 'lambada', 'babiqa', 'agnews', 'dbpedia', 'sst2']
    if name == 'finewiki':
        ds = load_dataset("HuggingFaceFW/finewiki", name="en")  
    if name == 'tinystories':
        ds = load_dataset("skeskinen/TinyStories-GPT4")
    if name == 'simplestories':
        ds = load_dataset("SimpleStories/SimpleStories")
    if name == 'narrativeqa':
        ds = load_dataset("deepmind/narrativeqa")
    if name == 'fineweb':
        ds = load_dataset("HuggingFaceFW/fineweb-edu", "sample-10BT")
        ds = DatasetDict({"train": ds["train"].shuffle(seed=seed).select(range(2000000))}) # type: ignore
        ds = ds.remove_columns([c for c in ds["train"].column_names if c != "text"])
    if name == 'squad':
        ds = load_dataset("rajpurkar/squad")
    if name == 'rocstories':
        ds = load_dataset("inkoziev/roc_stories")
    if name == 'xsum':
        ds = load_dataset("EdinburghNLP/xsum")
    if name == 'lambada':
        ds = load_dataset("EleutherAI/lambada_openai", "en")
    if name == 'babiqa':
        ds = load_dataset("habanoz/babi_qa_en_valid_10k_qa1")
    if name == 'agnews':
        ds = load_dataset("fancyzhx/ag_news")
    if name == 'dbpedia':
        ds = load_dataset("fancyzhx/dbpedia_14")
    if name == 'sst2':
        ds = load_dataset("stanfordnlp/sst2")

    assert isinstance(ds, DatasetDict) # type: ignore
    dataset = split_train_val_test(ds, val_ratio, test_ratio, seed)

    if name in ['tinystories', 'simplestories']:
        dataset = change_key_name(dataset, 'story', 'text')
    if name == 'narrativeqa':
        dataset = preprocess_narrativeqa_to_text(dataset)
    if name == 'squad':
        dataset = preprocess_squad_to_text(dataset)
    if name == 'rocstories':
        dataset = preprocess_rocstories(dataset)
    if name == 'xsum':
        dataset = preprocess_xsum_to_text(dataset)
    if name == 'babiqa':
        dataset = preprocess_babiqa_to_text(dataset)
    if name == 'agnews':
        dataset = preprocess_agnews_to_text(dataset)
    if name == 'dbpedia':
        dataset = preprocess_dbpedia14_to_text(dataset)
    if name == 'sst2':
        dataset = preprocess_sst2_to_text(dataset)
    return dataset



def tokenize_or_load(
    dataset: DatasetDict | None = None, 
    model_name: str = 'meta-llama/Llama-3.1-8B',
    text_key: str = 'text', 
    data_dir: str = "../data/tinystories",
    max_length: int = 1024,
    tokenize_batch_size: int = 8,
    force_remap: bool = False,
    console: Console | None = None
) -> DatasetDict: 
    if os.path.exists(os.path.join(data_dir, 'train')) and not force_remap:
        return cast(DatasetDict, load_from_disk(data_dir))

    if dataset is None:
        raise ValueError("Tokenized dataset not found. Need to input dataset to be tokenized.")


    tokenizer = AutoTokenizer.from_pretrained(
        model_name, use_fast=True
    )
    if console is not None:
        console.print(f"[green]✓ Loaded tokenizer from: {model_name}[/green]")
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '<|padding|>'})
        
        
    def tokenize_per_batch(batch: Dict[str, Any]) -> Dict[str, Any]:
        texts: List[str] = [str(t) if t is not None else "" for t in batch[text_key]]

        enc = tokenizer(
            texts, 
            max_length=max_length - 1,              # Leave one space for EOS
            truncation=True,
            padding=False,
            return_attention_mask=True,
            add_special_tokens=True
        )

        input_ids = []
        att_masks = []

        for ids, mask in zip(enc["input_ids"], enc["attention_mask"]):
            ids = ids + [tokenizer.eos_token_id]    # Add EOS at the end
            mask = mask + [1]                       # EOS is treated as valid token

            input_ids.append(ids)
            att_masks.append(mask)

        return {'input_ids': input_ids, 'att_mask': att_masks}

    tokenized = {}
    for split in ['train', 'validation', 'test']:
        if split not in dataset:
            continue
        tokenized[split] = dataset[split].map(
            tokenize_per_batch,
            batched=True,
            batch_size=tokenize_batch_size,
            remove_columns=dataset[split].column_names,
            num_proc=os.cpu_count(),
        )
    tokenized = DatasetDict(tokenized)
    os.makedirs(data_dir, exist_ok=True)
    tokenized.save_to_disk(data_dir)
    if console is not None:
        console.print(f"[green]✓ Saved tokens to[/green] [bold cyan]{data_dir}[/bold cyan]")
        console.print(f"[bold blue]Tokenized data structure:[/bold blue]")
        console.print(tokenized)
        
    return tokenized
    

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
                 pin_memory: bool = False, subset_seed: Optional[int] = None):
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
        tokenized_datasets = tokenize_or_load(data_dir = data_dir)
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

