from typing import List, Tuple, Dict
import json
import argparse
import random
import os

from tqdm import tqdm
from langdetect import detect
from datasets import load_dataset

DATASET_PATH = "openbmb/UltraFeedback"


def _split_train_eval(samples: List[Dict], train_ratio: float=0.8) -> Tuple[List, List]:
    num_train_samples = int(len(samples) * train_ratio)
    return samples[:num_train_samples], samples[num_train_samples:]


def main(args: argparse.Namespace):
    # Download from HuggingFace dataset repo.
    ds = load_dataset(DATASET_PATH)
    print(f"Total number of samples: {len(ds['train'])}")

    # Language filtering and separate by sources.
    source2samples = {}
    print(f"Leaving only English data...")
    for obj in tqdm(ds['train']):
        source = obj['source']
        if source not in source2samples:
            source2samples[source] = []

        instruction = obj['instruction']
        try:
            if detect(instruction) == 'en':
                completions = obj['completions']
                annotations = []
                for completion in completions:
                    response = completion['response']
                    if detect(response) == 'en':
                        score = completion['overall_score']
                        annotations.append({
                            'response': response,
                            'score': score
                        })

                if annotations:
                    source2samples[source].append({
                        'instruction': instruction,
                        'annotations': annotations
                    })

        except Exception as e:
            pass

    print(f"Numbers off samples after filtering + parsing:")
    total_count = 0
    for source, samples in source2samples.items():
        print(f"{source}: {len(samples)}")
        total_count += len(samples)
    print(f"Total: {total_count}")
    print()

    # Get the preference dataset using minimum difference.
    sft_dataset, pref_dataset = [], []
    for source, samples in source2samples.items():
        for sample in samples:
            instruction = sample['instruction']
            annotations = sample['annotations']
            annotations = sorted(annotations, key=lambda x:x['score'], reverse=True)

            total_count += len(annotations)
            chosen_idx, rejected_idx = 0, 1
            sampled_idxs = set()
            while chosen_idx < rejected_idx and rejected_idx < len(annotations):
                if annotations[chosen_idx]['score'] - annotations[rejected_idx]['score'] >= args.min_diff:
                    pref_dataset.append({
                        'instruction': instruction, 
                        'chosen': {'response': annotations[chosen_idx]['response'], 'score': annotations[rejected_idx]['response']},
                        'rejected': {'response': annotations[rejected_idx]['response'], 'score': annotations[rejected_idx]['response']}
                    })
                    sampled_idxs.add(chosen_idx)
                    sampled_idxs.add(rejected_idx)
                    
                    chosen_idx = rejected_idx + 1
                    rejected_idx = chosen_idx + 1
                else:
                    rejected_idx += 1
            
            for i in range(len(annotations)):
                if i not in sampled_idxs:
                    sft_dataset.append({'instruction': instruction, 'response': annotations[i]['response'], 'score': annotations[i]['score']})

    # Random sampling per source and score.
    print("Random sampling & Splitting...")
    random.seed(args.seed)
    random.shuffle(sft_dataset)
    random.shuffle(pref_dataset)
    sft_train_set, sft_eval_set = _split_train_eval(sft_dataset, args.train_ratio)
    pref_train_set, pref_eval_set = _split_train_eval(pref_dataset, args.train_ratio)

    print(f"[Supervised Fine-Tuning]")
    print(f"Total: {len(sft_train_set) + len(sft_eval_set)}")
    print(f"Train: {len(sft_train_set)}")
    print(f"Eval: {len(sft_eval_set)}")
    print()

    print(f"[Preference tuning: Reward Modeling / PPO (Proximal Policy Optimization / DPO (Direct Preference Optimization))]")
    print(f"Total: {len(pref_train_set) + len(pref_eval_set)}")
    print(f"Train: {len(pref_train_set)}")
    print(f"Eval: {len(pref_eval_set)}")
    print()

    # Save the data samples.
    sft_data_dir = args.data_dir + "/sft"
    if not os.path.isdir(sft_data_dir):
        os.makedirs(sft_data_dir)
    with open(f"{sft_data_dir}/train_samples.json", 'w') as f:
        json.dump(sft_train_set, f)
    with open(f"{sft_data_dir}/eval_samples.json", 'w') as f:
        json.dump(sft_eval_set, f)

    pref_data_dir = args.data_dir + "/pref"
    if not os.path.isdir(pref_data_dir):
        os.makedirs(pref_data_dir)
    with open(f"{pref_data_dir}/train_samples.json", 'w') as f:
        json.dump(pref_train_set, f)
    with open(f"{pref_data_dir}/eval_samples.json", 'w') as f:
        json.dump(pref_eval_set, f)

    print("Data loading done.")


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42, help="The random seed for sampling / shuffling.")
    parser.add_argument('--data_dir', type=str, default=".data", help="The name of the parent directory where data files are stored.")
    parser.add_argument('--train_ratio', type=float, default=0.8, help="The ratio of the data samples for train set.")
    parser.add_argument('--min_diff', type=float, default=2.0, help="The minimum difference between chosen and rejected responses.")
              
    args = parser.parse_args()

    main(args)
