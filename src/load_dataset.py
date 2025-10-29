from typing import List, Tuple
import json
import argparse
import random
import os
import math

from tqdm import tqdm
import kagglehub
import numpy as np


def _process_sample(sample: dict) -> dict:
    new_sample = {
        'name': sample['name'],
        'summary': sample['short_description'],
        'positive': sample['positive'],
        'negative': sample['negative'],
        'categories': sample['categories'],
        'genres': sample['genres'],
        'description': sample['detailed_description']
    }
    return new_sample


def _split_train_eval(pairs: List[Tuple[str, int]], train_ratio: float=0.8) -> Tuple[List, List]:
    num_train_samples = int(len(pairs) * train_ratio)
    return pairs[:num_train_samples], pairs[num_train_samples:]


def main(args: argparse.Namespace):
    # Download latest version
    path = kagglehub.dataset_download("fronkongames/steam-games-dataset")
    with open(path + "/games.json", 'r', encoding='utf-8') as f:
        games = json.load(f)
    print(f"Total number of samples: {len(games)}")

    # Calculate the scores by positive - negative and store in bins.
    game_ids = list(games.keys())
    num_samples_per_bins = math.ceil(len(game_ids) / args.num_bins)
    score_pairs = []
    for game_id in tqdm(game_ids):
        obj = games[game_id]
        positive, negative = obj['positive'], obj['negative']
        score_pairs.append((game_id, positive - negative))
    score_pairs = sorted(score_pairs, key=lambda x:x[1])
    bins = [[] for i in range(args.num_bins)]
    cur_bin = 0
    for game_id, score in score_pairs:
        bins[cur_bin].append((game_id, score))

        if len(bins[cur_bin]) == num_samples_per_bins:
            cur_bin += 1

    print("[Score Distribution]")
    for b, bin in enumerate(bins):
        scores = [pair[1] for pair in bin]
        print(f"Bin {b} =>")
        print(f"Max: {np.max(scores)} || Min: {np.min(scores)} || Mean: {np.mean(scores)} || Median: {np.median(scores)}")
        print()

    # Random sampling & split.
    print("Random sampling & Splitting...")
    sft_train_set, sft_eval_set = [], []
    rm_train_set, rm_eval_set = [], []
    ppo_train_set, ppo_eval_set = [], []
    random.seed(args.seed)
    for bin in bins:
        random.shuffle(bin)

        # Split by stage.
        num_sft_split, num_rm_split = int(len(bin) * args.sft_ratio), int(len(bin) * args.rm_ratio)
        sft_split, rm_split, ppo_split = bin[:num_sft_split], bin[num_sft_split:num_sft_split+num_rm_split], bin[num_sft_split+num_rm_split:]
        sft_train_split, sft_eval_split = _split_train_eval(sft_split, args.train_ratio)
        rm_train_split, rm_eval_split = _split_train_eval(rm_split, args.train_ratio)
        ppo_train_split, ppo_eval_split = _split_train_eval(ppo_split, args.train_ratio)

        sft_train_set += [_process_sample(games[game_id]) for game_id, _ in sft_train_split]
        sft_eval_set += [_process_sample(games[game_id]) for game_id, _ in sft_eval_split]
        rm_train_set += [_process_sample(games[game_id]) for game_id, _ in rm_train_split]
        rm_eval_set += [_process_sample(games[game_id]) for game_id, _ in rm_eval_split]
        ppo_train_set += [_process_sample(games[game_id]) for game_id, _ in ppo_train_split]
        ppo_eval_set += [_process_sample(games[game_id]) for game_id, _ in ppo_eval_split]

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

    print(f"[RLHF: PPO Training]")
    print(f"Total: {len(ppo_train_set) + len(ppo_eval_set)}")
    print(f"Train: {len(ppo_train_set)}")
    print(f"Eval: {len(ppo_eval_set)}")
    print()

    # Save the data samples.
    sft_data_dir = args.data_dir + "/supervised"
    if not os.path.isdir(sft_data_dir):
        os.makedirs(sft_data_dir)
    with open(f"{sft_data_dir}/train_samples.json", 'w') as f:
        json.dump(sft_train_set, f)
    with open(f"{sft_data_dir}/eval_samples.json", 'w') as f:
        json.dump(sft_eval_set, f)

    rm_data_dir = args.data_dir + "/reward"
    if not os.path.isdir(rm_data_dir):
        os.makedirs(rm_data_dir)
    with open(f"{rm_data_dir}/train_samples.json", 'w') as f:
        json.dump(rm_train_set, f)
    with open(f"{rm_data_dir}/eval_samples.json", 'w') as f:
        json.dump(rm_eval_set, f)

    ppo_data_dir = args.data_dir + "/ppo"
    if not os.path.isdir(ppo_data_dir):
        os.makedirs(ppo_data_dir)
    with open(f"{ppo_data_dir}/train_samples.json", 'w') as f:
        json.dump(ppo_train_set, f)
    with open(f"{ppo_data_dir}/eval_samples.json", 'w') as f:
        json.dump(ppo_eval_set, f)

    print("Data loading done.")


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42, help="The random seed.")
    parser.add_argument('--data_dir', type=str, default=".data", help="The name of the parent directory where data files are stored.")
    parser.add_argument('--sft_ratio', type=float, default=0.4, help="The ratio of the data samples for supervised fine-tuning.")
    parser.add_argument('--rm_ratio', type=float, default=0.2, help="The ratio of the data samples for reward model training.")
    parser.add_argument('--train_ratio', type=float, default=0.8, help="The ratio of the data samples for train set.")
    parser.add_argument('--num_bins', type=int, default=10, help="The number of bins to distribute the samples by scores.")
              
    args = parser.parse_args()

    main(args)
