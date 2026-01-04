import argparse
import json

from _util import _fix_seed, PrefDataset, PrefPadCollate
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.utils.data import DataLoader
import torch


def main(args: argparse.Namespace):
    # Set the GPU.
    device = torch.device(f"cuda:{args.gpu_id}") if torch.cuda.is_available() else torch.device('cpu')

    # Load the models.
    _fix_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    model = AutoModelForCausalLM.from_pretrained(args.model_id).to(device)

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

    # Preprocess Dataset objects.
    print("[# of data samples after pre-processing]")
    train_set = PrefDataset(train_samples, tokenizer, args.max_len, args.min_gen_len)
    eval_set = PrefDataset(eval_samples, tokenizer, args.max_len, args.min_gen_len)
    print(f"{len(train_set)} samples processed from train set.")
    print(f"{len(eval_set)} samples processed from eval set.")

    ppd = PrefPadCollate(tokenizer.eos_token_id)
    train_loader = DataLoader(train_set, 
                              collate_fn=ppd.pad_collate, 
                              batch_size=args.batch_size,
                              shuffle=True, num_workers=4, pin_memory=True)
    eval_loader = DataLoader(eval_set, 
                             collate_fn=ppd.pad_collate, 
                             batch_size=args.batch_size,
                             num_workers=4, pin_memory=True)


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42, help="The random seed.")
    parser.add_argument('--sft_model_path', type=str, required=True, help="The checkpoint path of the supervised fine-tuned model.")
    parser.add_argument('--ckpt_dir', type=str, default=".model/ppo", help="The name of the directory to save checkpoints.")
    parser.add_argument('--gpu_id', type=int, default=0, help="The GPU ID to use if CUDA is available.")
    parser.add_argument('--data_dir', type=str, default=".data/pref", help="The name of the directory where data files are stored.")
    parser.add_argument('--max_len', type=int, default=1024, help="The maximum number of tokens.")
    parser.add_argument('--min_gen_len', type=int, default=100, help="The minumum number of tokens to generate, except for tags and EOS token.")
    parser.add_argument('--num_epochs', type=int, default=3, help="The number of epochs.")
    parser.add_argument('--batch_size', type=int, default=16, help="The batch size.")
    parser.add_argument('--learning_rate', type=float, default=2e-5, help="The learning rate.")
    parser.add_argument('--max_gradient_norm', type=float, default=1.0, help="The maximum value for gradient clipping.")
    parser.add_argument('--beta', type=float, default=0.0, help="The coefficient for per-token KL divergence.")

    args = parser.parse_args()

    main(args)
