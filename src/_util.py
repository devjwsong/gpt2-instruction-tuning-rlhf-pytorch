from typing import Union, List, Dict, Tuple
import random

from torch.utils.data import Dataset
from tqdm import tqdm
import numpy as np
import torch

KEY2TAG = {
    'name': ("<name>", "</name>"),
    'summary': ("<summary>", ("</summary>")),
    'categories': ("<categories>", "</categories>"),
    'genres': ("<genres>", "</genres>"),
    'description': ("<description>", "</description>") 
}


def _format_tag(k: str, v: Union[str, List[str]]) -> str:
    start, end = KEY2TAG[k]
    if isinstance(v, str):
        return start + v + end
    
    return start + ', '.join(v) + end


class SFTDataset(Dataset):
    def __init__(self, 
                 samples: List[Dict], 
                 tokenizer: object,
                 max_len: int=1024,
                 min_gen_len: int=100
    ):
        self.input_ids = []  # (N ,L)
        self.labels = []  # (N, L)

        # Process each sample.
        for sample in tqdm(samples):
            sequence = ""
            for input_key in ['name', 'summary', 'categories', 'genres']:
                sequence += _format_tag(input_key, sample[input_key])
            sequence_ids = tokenizer(sequence)['input_ids']

            # Check the minimum generation requirement.
            desc_start, desc_end = KEY2TAG['description']
            desc_start_token_ids = tokenizer(desc_start)['input_ids']
            desc_end_token_ids = tokenizer(desc_end)['input_ids']
            cur_len = len(sequence_ids) + len(desc_start_token_ids) + len(desc_end_token_ids) + 1
            if cur_len + min_gen_len > max_len:
                continue

            # Append the label.
            sequence_ids += desc_start_token_ids
            desc_token_ids = tokenizer(sample['description'])['input_ids']
            gen_len = max_len - 1 - len(desc_end_token_ids) - len(sequence_ids)
            desc_token_ids = desc_token_ids[:gen_len]
            self.input_ids.append(sequence_ids + desc_token_ids + desc_end_token_ids + [tokenizer.eos_token_id])
            self.labels.append([-100] * len(sequence_ids) + desc_token_ids + desc_end_token_ids + [tokenizer.eos_token_id])

    def __len__(self) -> int:
        return len(self.input_ids)
    
    def __getitem__(self, idx) -> Tuple[List[int], List[int]]:
        return self.input_ids[idx], self.labels[idx]
    

class SFTPadCollate():
    def __init__(self, eos_id: int):
        self.eos_id = eos_id
        
    def pad_collate(self, batch) -> Tuple[torch.Tensor, torch.Tensor]:
        input_ids, labels = [], []
        for seqs in batch:
            input_ids.append(torch.LongTensor(seqs[0]))
            labels.append(torch.LongTensor(seqs[1]))
            
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=self.eos_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)
    
        return input_ids, labels
    

class RMDataset(Dataset):
    def __init__(self, 
                 samples: List[Dict], 
                 tokenizer: object,
                 max_len: int=1024,
                 min_target_len: int=100,
                 max_score: float=1.0
    ):
        self.input_ids = []  # (N ,L)
        self.labels = []  # (N)

        # Process each sample.
        for sample in tqdm(samples):
            sequence = ""
            for input_key in ['name', 'summary', 'categories', 'genres']:
                sequence += _format_tag(input_key, sample[input_key])
            sequence_ids = tokenizer(sequence)['input_ids']

            # Check the minimum target requirement.
            desc_start, desc_end = KEY2TAG['description']
            desc_start_token_ids = tokenizer(desc_start)['input_ids']
            desc_end_token_ids = tokenizer(desc_end)['input_ids']
            cur_len = len(sequence_ids) + len(desc_start_token_ids) + len(desc_end_token_ids) + 1
            if cur_len + min_target_len > max_len:
                continue

            # Append the target.
            sequence_ids += desc_start_token_ids
            desc_token_ids = tokenizer(sample['description'])['input_ids']
            gen_len = max_len - 1 - len(desc_end_token_ids) - len(sequence_ids)
            desc_token_ids = desc_token_ids[:gen_len]
            self.input_ids.append(sequence_ids + desc_token_ids + desc_end_token_ids + [tokenizer.eos_token_id])

            # Normalize & Add label.
            positive, negative = sample['positive'], sample['negative']
            score = torch.sigmoid(torch.IntTensor([positive - negative])).item()
            self.labels.append(2 * max_score * score - max_score)

    def __len__(self) -> int:
        return len(self.input_ids)
    
    def __getitem__(self, idx) -> Tuple[List[int], List[int]]:
        return self.input_ids[idx], self.labels[idx]


class RMPadCollate():
    def __init__(self, eos_id: int):
        self.eos_id = eos_id
        
    def pad_collate(self, batch) -> Tuple[torch.Tensor, torch.Tensor]:
        input_ids, labels = [], []
        for pair in batch:
            input_ids.append(torch.LongTensor(pair[0]))
            labels.append(pair[1])
            
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=self.eos_id)
        labels = torch.FloatTensor(labels)
    
        return input_ids, labels
    

class QueryDataset(Dataset):
    def __init__(self, 
                 samples: List[Dict], 
                 tokenizer: object,
                 max_len: int=1024,
                 min_gen_len: int=100
    ):
        self.input_ids = []  # (N ,L)

        # Process each sample.
        for sample in tqdm(samples):
            sequence = ""
            for input_key in ['name', 'summary', 'categories', 'genres']:
                sequence += _format_tag(input_key, sample[input_key])
            sequence_ids = tokenizer(sequence)['input_ids']

            # Check the minimum generation requirement.
            desc_start, desc_end = KEY2TAG['description']
            desc_start_token_ids = tokenizer(desc_start)['input_ids']
            desc_end_token_ids = tokenizer(desc_end)['input_ids']
            cur_len = len(sequence_ids) + len(desc_start_token_ids) + len(desc_end_token_ids) + 1
            if cur_len + min_gen_len > max_len:
                continue

            self.input_ids.append(sequence_ids + desc_start_token_ids)

    def __len__(self) -> int:
        return len(self.input_ids)
    
    def __getitem__(self, idx) -> List[int]:
        return self.input_ids[idx]



def _fix_seed(seed: int=0):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)


def _masked_whitening(values: torch.tensor, masks: torch.tensor, eps: float=1e-7) -> torch.tensor:
    masked_means = (values * masks).sum() / masks.sum()  # ()
    diffs = values - masked_means  # ()
    masked_vars = (diffs ** 2 * masks).sum() / masks.sum()  # ()
    return (values - masked_means) * torch.rsqrt(masked_vars + eps)  # (N, L)
