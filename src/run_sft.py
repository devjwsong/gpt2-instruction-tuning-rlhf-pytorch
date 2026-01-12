from typing import Tuple
import argparse
import json
import os
import math

from _util import _fix_seed, SFTDataset, SFTPadCollate
from transformers import AutoTokenizer, AutoModelForCausalLM, get_polynomial_decay_schedule_with_warmup
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm
import torch
import numpy as np


def _evaluate(
    model: AutoModelForCausalLM,
    eval_loader: DataLoader
) -> Tuple[float, float]:
    print("[Validation]")
    model.eval()

    valid_losses, valid_ppls = [], []
    with torch.no_grad():
        for batch in tqdm(eval_loader):
            input_ids, labels = batch
            input_ids, labels = input_ids.to(model.device), labels.to(model.device)

            loss = model(input_ids=input_ids, labels = labels)[0]
            
            valid_losses.append(loss.detach())
            ppl = torch.exp(loss.detach())
            valid_ppls.append(ppl)
        
        valid_losses = [loss.item() for loss in valid_losses]
        valid_ppls = [ppl.item() if not math.isinf(ppl.item()) else 1e+8 for ppl in valid_ppls]
        valid_loss = np.mean(valid_losses)
        valid_ppl = np.mean(valid_ppls)
        
        if math.isnan(valid_ppl):
            valid_ppl = 1e+8
            
    return valid_loss, valid_ppl


def _train(
    args: argparse.Namespace,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    train_loader: DataLoader,
    eval_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler
):
    _fix_seed(args.seed)
    print("[Training]")

    best_loss = 1e+8
    for epoch in range(1, args.num_epochs+1):
        print(f"[Epoch {epoch}]")
        model.train()

        train_losses, train_ppls = [], []
        for batch in tqdm(train_loader):
            input_ids, labels = batch
            input_ids, labels = input_ids.to(model.device), labels.to(model.device)

            loss = model(input_ids=input_ids, labels=labels)[0]

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            train_losses.append(loss.detach())
            ppl = torch.exp(loss.detach())
            train_ppls.append(ppl)

        # Aggregate the epoch result.
        train_losses = [loss.item() for loss in train_losses]
        train_ppls = [ppl.item() if not math.isinf(ppl.item()) else 1e+8 for ppl in train_ppls]
        train_loss = np.mean(train_losses)
        train_ppl = np.mean(train_ppls)
        print(f"Train loss: {train_loss} || Train perplexity: {train_ppl}")

        valid_loss, valid_ppl =_evaluate(model, eval_loader)
        if valid_loss < best_loss:
            best_loss = valid_loss
            print("Best validation loss updated. Checkpointing...")
            model_name = args.model_id.split('/')[-1]
            ckpt_path = f"{args.ckpt_dir}/{model_name}_sft_epoch={epoch}_loss={best_loss:.4f}"
            model.save_pretrained(ckpt_path)
            tokenizer.save_pretrained(ckpt_path)

        print(f"Best valid loss: {best_loss}")
        print(f"Valid loss: {valid_loss} || Valid perplexity: {valid_ppl}")
        print()


def main(args: argparse.Namespace):
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

    # Set the GPU.
    device = torch.device(f"cuda:{args.gpu_id}") if torch.cuda.is_available() else torch.device('cpu')

    # Load the tokenizer and model.
    _fix_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    model = AutoModelForCausalLM.from_pretrained(args.model_id).to(device)

    # Preprocess Dataset objects.
    print("[# of data samples after pre-processing]")
    train_set = SFTDataset(train_samples, tokenizer, args.max_len, args.min_gen_len)
    eval_set = SFTDataset(eval_samples, tokenizer, args.max_len, args.min_gen_len)
    print(f"{len(train_set)} samples processed from train set.")
    print(f"{len(eval_set)} samples processed from eval set.")

    ppd = SFTPadCollate(tokenizer.eos_token_id)
    train_loader = DataLoader(train_set, 
                              collate_fn=ppd.pad_collate, 
                              batch_size=args.batch_size,
                              shuffle=True, num_workers=4, pin_memory=True)
    eval_loader = DataLoader(eval_set, 
                             collate_fn=ppd.pad_collate, 
                             batch_size=args.batch_size,
                             num_workers=4, pin_memory=True)
    
    # Set the optimizer and learning rate scheduler.
    num_batches = len(train_loader)
    total_train_steps = args.num_epochs * num_batches
    warmup_steps = int(args.warmup_ratio * total_train_steps)
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    scheduler = get_polynomial_decay_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_train_steps,
        power=2
    )

    if not os.path.isdir(args.ckpt_dir):
        os.makedirs(args.ckpt_dir)

    # Training loop.
    _train(
        args=args,
        model=model,
        tokenizer=tokenizer,
        train_loader=train_loader,
        eval_loader=eval_loader,
        optimizer=optimizer,
        scheduler=scheduler
    )
    print("DONE.")


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42, help="The random seed.")
    parser.add_argument('--data_dir', type=str, default=".data/sft", help="The name of the directory where data files are stored.")
    parser.add_argument('--ckpt_dir', type=str, default=".model/sft", help="The name of the directory to save checkpoints.")
    parser.add_argument('--model_id', type=str, default="openai-community/gpt2", help="The model ID of the pre-trained GPT-2 model in Hugging Face Hub.")
    parser.add_argument('--gpu_id', type=int, default=0, help="The GPU ID to use if CUDA is available.")
    parser.add_argument('--max_len', type=int, default=1024, help="The maximum number of tokens.")
    parser.add_argument('--min_gen_len', type=int, default=1, help="The minumum number of tokens to generate, except for tags and EOS token.")
    parser.add_argument('--batch_size', type=int, default=16, help="The batch size.")
    parser.add_argument('--num_epochs', type=int, default=5, help="The number of epochs.")
    parser.add_argument('--learning_rate', type=float, default=2e-5, help="The learning rate.")
    parser.add_argument('--warmup_ratio', type=float, default=0.1, help="The ratio of warm-up steps to the total training steps.")

    args = parser.parse_args()

    main(args)
