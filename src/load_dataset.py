from typing import List, Tuple, Dict
import json
import argparse
import random
import os

from tqdm import tqdm
from langdetect import detect
from datasets import load_dataset

DATASET_PATH = "openbmb/UltraFeedback"
NUM_BINS = 10


def _split_train_eval(samples: List[Dict], train_ratio: float=0.8) -> Tuple[List, List]:
    num_train_samples = int(len(samples) * train_ratio)
    return samples[:num_train_samples], samples[num_train_samples:]


def main(args: argparse.Namespace):
    # Download from HuggingFace dataset repo.
    ds = load_dataset(DATASET_PATH)
    print(f"Total number of samples: {len(ds)}")

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
                for completion in completions:
                    response = completion['response']
                    if detect(response) == 'en':
                        score = completion['overall_score']
                        source2samples[source].append({
                            'instruction': instruction,
                            'response': response,
                            'score': score
                        })

        except Exception as e:
            print(e)
    print(f"Numbers off samples after filtering + parsing:")
    total_count = 0
    for source, samples in source2samples.items():
        print(f"{source}: {len(samples)}")
        total_count += len(samples)
    print(f"Total: {total_count}")
    print()

    # Random sampling per source and score.
    print("Random sampling & Splitting...")
    sft_train_set, sft_eval_set = [], []
    rm_train_set, rm_eval_set = [], []
    rlhf_train_set, rlhf_eval_set = [], []
    random.seed(args.seed)
    for source, samples in source2samples.items():
        print(f"Processing source: {source}...")
        bins = [[] for i in range(NUM_BINS)]
        for sample in tqdm(samples):
            score = sample['score']
            bin_number = int(score)
            bins[bin_number-1].append(sample)

        for b, bin in enumerate(bins):
            print(f"Bin [{b+1}:{b+2}): {len(bin)}")
            random.shuffle(bin)

            # Split by stage.
            num_sft_split, num_rm_split = int(len(bin) * args.sft_ratio), int(len(bin) * args.rm_ratio)
            sft_split, rm_split, rlhf_split = bin[:num_sft_split], bin[num_sft_split:num_sft_split+num_rm_split], bin[num_sft_split+num_rm_split:]
            sft_train_split, sft_eval_split = _split_train_eval(sft_split, args.train_ratio)
            rm_train_split, rm_eval_split = _split_train_eval(rm_split, args.train_ratio)
            rlhf_train_split, rlhf_eval_split = _split_train_eval(rlhf_split, args.train_ratio)

            sft_train_set += sft_train_split
            sft_eval_set += sft_eval_split
            rm_train_set += rm_train_split
            rm_eval_set += rm_eval_split
            rlhf_train_set += rlhf_train_split
            rlhf_eval_set += rlhf_eval_split
        print()

    print(f"[Supervised Fine-Tuning]")
    print(f"Total: {len(sft_train_set) + len(sft_eval_set)}")
    print(f"Train: {len(sft_train_set)}")
    print(f"Eval: {len(sft_eval_set)}")
    print()

    print(f"[Reward Model Training]")
    print(f"Total: {len(rm_train_set) + len(rm_eval_set)}")
    print(f"Train: {len(rm_train_set)}")
    print(f"Eval: {len(rm_eval_set)}")
    print()

    print(f"[Reinforcement Learning from Human Feedback]")
    print(f"Total: {len(rlhf_train_set) + len(rlhf_eval_set)}")
    print(f"Train: {len(rlhf_train_set)}")
    print(f"Eval: {len(rlhf_eval_set)}")
    print()

    # Save the data samples.
    sft_data_dir = args.data_dir + "/sft"
    if not os.path.isdir(sft_data_dir):
        os.makedirs(sft_data_dir)
    with open(f"{sft_data_dir}/train_samples.json", 'w') as f:
        json.dump(sft_train_set, f)
    with open(f"{sft_data_dir}/eval_samples.json", 'w') as f:
        json.dump(sft_eval_set, f)

    rm_data_dir = args.data_dir + "/rm"
    if not os.path.isdir(rm_data_dir):
        os.makedirs(rm_data_dir)
    with open(f"{rm_data_dir}/train_samples.json", 'w') as f:
        json.dump(rm_train_set, f)
    with open(f"{rm_data_dir}/eval_samples.json", 'w') as f:
        json.dump(rm_eval_set, f)

    rlhf_data_dir = args.data_dir + "/rlhf"
    if not os.path.isdir(rlhf_data_dir):
        os.makedirs(rlhf_data_dir)
    with open(f"{rlhf_data_dir}/train_samples.json", 'w') as f:
        json.dump(rlhf_train_set, f)
    with open(f"{rlhf_data_dir}/eval_samples.json", 'w') as f:
        json.dump(rlhf_eval_set, f)

    print("Data loading done.")


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42, help="The random seed.")
    parser.add_argument('--data_dir', type=str, default=".data", help="The name of the parent directory where data files are stored.")
    parser.add_argument('--sft_ratio', type=float, default=0.4, help="The ratio of the data samples for supervised fine-tuning.")
    parser.add_argument('--rm_ratio', type=float, default=0.2, help="The ratio of the data samples for reward model training.")
    parser.add_argument('--train_ratio', type=float, default=0.8, help="The ratio of the data samples for train set.")
              
    args = parser.parse_args()

    main(args)
