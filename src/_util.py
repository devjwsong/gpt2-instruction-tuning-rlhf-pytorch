from typing import Union, List, Dict, Tuple
import random

from torch.utils.data import Dataset
from tqdm import tqdm
import numpy as np
import torch

KEY2TAG = {
    'instruction': ("<instruction>", "</instruction>"),
    'response': ("<response>", "</response>")
}


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
            inst_start, inst_end = KEY2TAG['instruction']
            sequence = inst_start + sample['instruction'] + inst_end
            sequence_ids = tokenizer(sequence)['input_ids']

            # Check the minimum generation requirement.
            resp_start, resp_end = KEY2TAG['response']
            resp_start_token_ids = tokenizer(resp_start)['input_ids']
            resp_end_token_ids = tokenizer(resp_end)['input_ids']
            cur_len = len(sequence_ids) + len(resp_start_token_ids) + len(resp_end_token_ids) + 1
            if cur_len + min_gen_len > max_len:
                continue

            # Append the label.
            sequence_ids += resp_start_token_ids
            resp_token_ids = tokenizer(sample['response'])['input_ids']
            gen_len = max_len - 1 - len(resp_end_token_ids) - len(sequence_ids)
            resp_token_ids = resp_token_ids[:gen_len]
            self.input_ids.append(sequence_ids + resp_token_ids + resp_end_token_ids + [tokenizer.eos_token_id])
            self.labels.append([-100] * len(sequence_ids) + resp_token_ids + resp_end_token_ids + [tokenizer.eos_token_id])

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

        # Find the current max scores.
        default_max_score = 0.0
        for sample in samples:
            default_max_score = max(default_max_score, sample['score'])

        # Process each sample.
        for sample in tqdm(samples):
            inst_start, inst_end = KEY2TAG['instruction']
            sequence = inst_start + sample['instruction'] + inst_end
            sequence_ids = tokenizer(sequence)['input_ids']

            # Check the minimum generation requirement.
            resp_start, resp_end = KEY2TAG['response']
            resp_start_token_ids = tokenizer(resp_start)['input_ids']
            resp_end_token_ids = tokenizer(resp_end)['input_ids']
            cur_len = len(sequence_ids) + len(resp_start_token_ids) + len(resp_end_token_ids) + 1
            if cur_len + min_target_len > max_len:
                continue

            # Append the target.
            sequence_ids += resp_start_token_ids
            resp_token_ids = tokenizer(sample['response'])['input_ids']
            gen_len = max_len - 1 - len(resp_end_token_ids) - len(sequence_ids)
            resp_token_ids = resp_token_ids[:gen_len]
            self.input_ids.append(sequence_ids + resp_token_ids + resp_end_token_ids + [tokenizer.eos_token_id])

            # Normalize & Add label.
            score = sample['score']
            if default_max_score > 1.0:
                score = 1.0 + (score - 1.0) * ((max_score - 1.0) / (default_max_score - 1.0))
            self.labels.append(score)

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
            inst_start, inst_end = KEY2TAG['instruction']
            sequence = inst_start + sample['instruction'] + inst_end
            sequence_ids = tokenizer(sequence)['input_ids']

            # Check the minimum generation requirement.
            resp_start, resp_end = KEY2TAG['response']
            resp_start_token_ids = tokenizer(resp_start)['input_ids']
            resp_end_token_ids = tokenizer(resp_end)['input_ids']
            cur_len = len(sequence_ids) + len(resp_start_token_ids) + len(resp_end_token_ids) + 1
            if cur_len + min_gen_len > max_len:
                continue

            self.input_ids.append(sequence_ids + resp_start_token_ids)

    def __len__(self) -> int:
        return len(self.input_ids)
    
    def __getitem__(self, idx) -> List[int]:
        return self.input_ids[idx]
    

class PrefDataset(Dataset):
    def __init__(self, 
                 samples: List[Dict], 
                 tokenizer: object,
                 max_len: int=1024,
                 min_gen_len: int=100
    ):
        self.chosen_input_ids, self.rejected_input_ids = [], []  # (N, C_L), (N, R_L)
        self.chosen_labels, self.rejected_labels = [], []  # (N, C_L), (N, R_L)

        # Process each sample.
        for sample in tqdm(samples):
            inst_start, inst_end = KEY2TAG['instruction']
            sequence = inst_start + sample['instruction'] + inst_end
            sequence_ids = tokenizer(sequence)['input_ids']

            # Check the minimum generation requirement.
            resp_start, resp_end = KEY2TAG['response']
            resp_start_token_ids = tokenizer(resp_start)['input_ids']
            resp_end_token_ids = tokenizer(resp_end)['input_ids']
            cur_len = len(sequence_ids) + len(resp_start_token_ids) + len(resp_end_token_ids) + 1
            if cur_len + min_gen_len > max_len:
                continue

            # Process pairs (chosen response, rejected response)
            sequence_ids += resp_start_token_ids
            chosen_response, rejected_response = sample['chosen']['response'], sample['rejected']['response']

            resp_token_ids = tokenizer(chosen_response)['input_ids']
            gen_len = max_len - 1 - len(resp_end_token_ids) - len(sequence_ids)
            resp_token_ids = resp_token_ids[:gen_len]
            self.chosen_input_ids.append(sequence_ids + resp_token_ids + resp_end_token_ids + [tokenizer.eos_token_id])
            self.chosen_labels.append([-100] * len(sequence_ids) + resp_token_ids + resp_end_token_ids + [tokenizer.eos_token_id])

            resp_token_ids = tokenizer(rejected_response)['input_ids']
            gen_len = max_len - 1 - len(resp_end_token_ids) - len(sequence_ids)
            resp_token_ids = resp_token_ids[:gen_len]
            self.rejected_input_ids.append(sequence_ids + resp_token_ids + resp_end_token_ids + [tokenizer.eos_token_id])
            self.rejected_labels.append([-100] * len(sequence_ids) + resp_token_ids + resp_end_token_ids + [tokenizer.eos_token_id])

    def __len__(self) -> int:
        return len(self.chosen_input_ids)
    
    def __getitem__(self, idx) -> Tuple[List[int], List[int]]:
        return self.chosen_input_ids[idx], self.rejected_input_ids[idx], self.chosen_labels[idx], self.rejected_labels[idx]


class PrefPadCollate():
    def __init__(self, eos_id: int):
        self.eos_id = eos_id
        
    def pad_collate(self, batch) -> Tuple[torch.Tensor, torch.Tensor]:
        chosen_input_ids, rejected_input_ids = [], []  # (B, C_L), (B, R_L)
        chosen_labels, rejected_labels = [], []  # (B, C_L), (B, R_L)
        for seqs in batch:
            chosen_input_ids.append(torch.LongTensor(seqs[0]))
            rejected_input_ids.append(torch.LongTensor(seqs[1]))
            chosen_labels.append(torch.LongTensor(seqs[2]))
            rejected_labels.append(torch.LongTensor(seqs[3]))
            
        chosen_input_ids = torch.nn.utils.rnn.pad_sequence(chosen_input_ids, batch_first=True, padding_value=self.eos_id)
        rejected_input_ids = torch.nn.utils.rnn.pad_sequence(rejected_input_ids, batch_first=True, padding_value=self.eos_id)
        chosen_labels = torch.nn.utils.rnn.pad_sequence(chosen_labels, batch_first=True, padding_value=-100)
        rejected_labels = torch.nn.utils.rnn.pad_sequence(rejected_labels, batch_first=True, padding_value=-100)
    
        return chosen_input_ids, rejected_input_ids, chosen_labels, rejected_labels


def _fix_seed(seed: int=0):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)


def _masked_averaging(values: torch.tensor, masks: torch.tensor):
     return (values * masks).sum() / masks.sum()  # ()


def _masked_whitening(values: torch.tensor, masks: torch.tensor, eps: float=1e-5) -> torch.tensor:
    masked_means = _masked_averaging(values, masks)  # ()
    diffs = values - masked_means  # ()
    masked_vars = (diffs ** 2 * masks).sum() / masks.sum()  # ()
    return (values - masked_means) * torch.rsqrt(masked_vars + eps)  # (N, L)
