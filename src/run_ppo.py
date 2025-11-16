from typing import List, Tuple
from copy import deepcopy
import argparse
import json

from model import RewardModel, PolicyWithValueHead
from _util import KEY2TAG, QueryDataset, _masked_whitening
from transformers import AutoTokenizer
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence
from torch.nn import functional as F
import torch


def generate_by_policy(
    queries: List[List[int]],
    policy: PolicyWithValueHead,
    tokenizer: AutoTokenizer,
    **generation_kwargs
) -> Tuple[List[List[int]], List[str]]:
    full_seqs, responses = [], []
    device = next(policy.parameters()).device
    for q, query_ids in enumerate(tqdm(queries)):
        output = policy.generate(
            torch.LongTensor(query_ids).unsqueeze(0).to(device), 
            **generation_kwargs
        ).squeeze(0).tolist() # (Q_L + R_L)
        output[-1] = tokenizer.eos_token_id  # Set the last token as EOS by default for consistency.
        full_seqs.append(output)

        # Extract the natural language response.
        generated = output[len(query_ids):-1]  # Remove query and EOS token part.
        _, desc_end = KEY2TAG['description']
        desc_end_token_ids = tokenizer(desc_end)['input_ids']
        if generated[len(generated)-len(desc_end_token_ids):] == desc_end_token_ids:
            generated = generated[:-len(desc_end_token_ids)]  # Remove </description> if it is included.
        response = tokenizer.decode(generated)
        responses.append(response)

    return full_seqs, responses


def get_rewards(full_seqs: List[torch.tensor], reward_model: RewardModel) -> Tuple[torch.tensor, torch.tensor]:
    # Pad the sequences and convert them into one tensor.
    reward_token_id = reward_model.reward_token_id
    input_ids = pad_sequence(full_seqs, batch_first=True, padding_value=reward_token_id)  # (B, L)
    
    device = next(reward_model.parameters()).device
    reward_model.eval()
    with torch.no_grad():
        rewards, reward_locs = reward_model(input_ids.to(device))  # (B), (B)

    return rewards, reward_locs


def _train(
    args: argparse.Namespace,
    policy: PolicyWithValueHead,
    reward_model: RewardModel,
    tokenizer: AutoTokenizer,
    train_query_set: QueryDataset,
    eval_query_set: QueryDataset,
    **generation_kwargs
):
    print("[Training]")
    # Copy the parameters to the reference policy. 
    # This model is fixed into the one from SFT and never changes.
    ref_policy = deepcopy(policy)
    ref_policy.eval()

    for outer_epoch in range(1, args.num_outer_epochs+1):
        print(f"[Epoch {outer_epoch}]")

        # Generate the outputs and get the rewards.
        print("Running interence using policy...")
        for b in range(0, len(train_query_set), args.infer_batch_size):
            policy.eval()
            batch_set = train_query_set[b:b+args.infer_batch_size]
            full_seqs, _ = generate_by_policy(batch_set, policy, tokenizer, **generation_kwargs)  # (B, Q_L + R_L), (B, R_L, V)
            final_rewards, reward_locs = get_rewards(full_seqs, reward_model)  # (B), (B)

            # Compute per-token KL divergence.
            device = next(policy.parameters()).device
            batch_seqs = pad_sequence(full_seqs, batch_first=True, padding_value=tokenizer.eos_token_id)  # (B, L)
            with torch.no_grad():
                logits, values = policy(batch_seqs.to(device)).logits  # (B, L, V), (B, L)
                ref_logits, _ = ref_policy(batch_seqs.to(device)).logits  # (B, L, V)
            log_probs = F.log_softmax(logits, dim=-1)  # (B, L, V)
            ref_log_probs = F.log_softmax(ref_logits, dim=-1)  # (B, L, V)
            log_probs = torch.gather(log_probs[:, :-1], dim=-1, index=batch_seqs[:, 1:]).squeeze(-1)  # (B, L-1) => This is used for clipping loss!
            ref_log_probs = torch.gather(ref_log_probs[:, :-1], dim=-1, index=batch_seqs[:, 1:]).squeeze(-1),  # (B, L-1)
            kl_divs = log_probs - ref_log_probs  # (B, L-1)

            # Mark the reward parts and set the per-token rewards/values.
            query_lens = [len(query_ids) for query_ids in batch_set]  # (B)
            query_lens = torch.LongTensor(query_lens).to(kl_divs.device)  # (B)
            rewards = args.beta * kl_divs  # (B, L-1)
            masks = torch.ones_like(rewards)  # (B, L-1)
            seq_len = masks.shape[1]
            seq_range = torch.arange(seq_len)
            masks *= (seq_range >= (query_lens-1).unsqueeze(1)).long()
            masks *= (seq_range < reward_locs.unsqueeze(1)).long()

            rewards *= masks
            values = values[:,:-1] * masks
            rewards.scatter_add_(dim=1, index=(reward_locs-1).unsqueeze(1), src=final_rewards.unsqueeze(1))

            # Compute Advantage. (GAE)
            last_gae_lam = 0
            advantage_reversed = []
            for t in reversed(range(seq_len)):
                next_values = values[:,t+1] if t < seq_len - 1 else 0.0  # (B)
                deltas = rewards[:,t] + args.gamma * next_values - values[:,t]  # (B)
                last_gae_lam = deltas + args.gamma * args.gae_lambda * last_gae_lam  # (B)
                advantage_reversed.append(last_gae_lam)
            advantages = torch.stack(advantage_reversed[::-1], dim=1)  # (B, L-1)
            returns = advantages + values  # (B, L-1)
            advantages = _masked_whitening(advantages, masks)

            # Inner training loop for PPO.
            policy.train()
            for inner_epoch in range(1, args.num_inner_epochs+1):
                pass



def main(args: argparse.Namespace):
    # Set the GPU.
    device = torch.device(f"cuda:{args.gpu_id}") if torch.cuda.is_available() else torch.device('cpu')

    # Load the models.
    policy = PolicyWithValueHead(args.sft_model_path).to(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.sft_model_path)
    reward_model = RewardModel(args.sft_model_path, tokenizer.eos_token_id, args.max_reward).to(args.device)
    state_dict = torch.load(f"{args.rm_model_path}/model.pth")
    reward_model.load_state_dict(state_dict)
    reward_model = reward_model.to(device)

    # Load the dataset.
    print("Loading the data samples...")
    with open(f"{args.data_dir}/train_samples.json", 'r') as f:
        train_samples = json.load(f)
    with open(f"{args.data_dir}/eval_samples.json", 'r') as f:
        eval_samples = json.load(f)
    print("[# of data samples]")
    print(f"Train samples: {len(train_samples)}")
    print(f"Eval samples: {len(eval_samples)}")
    print()

    # Set up the query datasets.
    print("[# of data samples after pre-processing]")
    train_query_set = QueryDataset(train_samples, tokenizer, args.max_len, args.min_gen_len)
    eval_query_set = QueryDataset(eval_samples, tokenizer, args.max_len, args.min_gen_len)
    print(f"{len(train_query_set)} samples processed from train set.")
    print(f"{len(eval_query_set)} samples processed from eval set.")

    #  Training loop.
    _, desc_end = KEY2TAG['description']
    desc_end_token_ids = tokenizer(desc_end)['input_ids']
    generation_kwargs = {
        'max_length': args.max_len,
        'min_new_tokens': args.min_gen_len + len(desc_end_token_ids) + 1,
        'do_sample': args.do_sample,
        'temperature': args.temperature,
        'top_k': args.top_k,
        'top_p': args.top_p
    }
    _train(
        args=args,
        policy=policy,
        reward_model=reward_model,
        tokenizer=tokenizer,
        train_query_set=train_query_set,
        eval_query_set=eval_query_set,
        **generation_kwargs
    )



if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42, help="The random seed.")
    parser.add_argument('--sft_model_path', type=str, required=True, help="The checkpoint path of the supervised fine-tuned model.")
    parser.add_argument('--rm_model_path', type=str, required=True, help="The checkpoint path of the reward model.")
    parser.add_argument('--gpu_id', type=int, default=0, help="The GPU ID to use if CUDA is available.")
    parser.add_argument('--max_reward', type=float, default=1.0, help="The maximum reward value. The reward range is set to [-max, max].")
    parser.add_argument('--data_dir', type=str, default=".data/ppo", help="The name of the directory where data files are stored.")
    parser.add_argument('--max_len', type=int, default=1024, help="The maximum number of tokens.")
    parser.add_argument('--min_gen_len', type=int, default=100, help="The minumum number of tokens to generate, except for tags and EOS token.")
    parser.add_argument('--do_sample', action='store_true', help="Whether or not to use sampling.")
    parser.add_argument('--temperature', type=float, default=1.0, help="The temperature value to control diversity during generation.")
    parser.add_argument('--top_k', type=int, default=50, help="The number of highest probability vocabulary tokens to keep for top-k-filtering.")
    parser.add_argument('--top_p', type=float, default=1.0, help=" Only the smallest set of most probable tokens with probabilities that add up to top_p or higher are kept for generation.")
    parser.add_argument('--num_outer_epochs', type=int, default=3, help="The number of epochs. (Outer training loop)")
    parser.add_argument('--num_inner_epochs', type=int, default=5, help="The number of epochs. (Innter PPO loop)'")
    parser.add_argument('--infer_batch_size', type=int, default=16, help="The batch size of inference.")
    parser.add_argument('--beta', type=float, default=0.0, help="The coefficient for per-token KL divergence.")
    parser.add_argument('--gae_lambda', type=float, default=0.0, help="The delta value for GAE computation.")
    parser.add_argument('--gamma', type=float, default=1.0, help="The discount factor for PPO.")

    args = parser.parse_args()

    main(args)
