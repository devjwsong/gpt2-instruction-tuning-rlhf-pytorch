import argparse
import json
import os

from _util import _fix_seed, RMDataset, RMPadCollate
from model import RewardModel
from transformers import AutoTokenizer, AutoModelWithLMHead, get_polynomial_decay_schedule_with_warmup
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm
import torch
import numpy as np


def _evaluate(
    model: AutoModelWithLMHead,
    eval_loader: DataLoader,
    loss_func: nn.MSELoss
) -> float:
    print("[Validation]")
    model.eval()

    valid_losses = []
    with torch.no_grad():
        for batch in tqdm(eval_loader):
            input_ids, labels = batch
            model_device = next(model.parameters()).device
            input_ids, labels = input_ids.to(model_device), labels.to(model_device)
            preds = model(input_ids)

            loss = loss_func(preds, labels)  # ()
            valid_losses.append(loss.detach())
        
        valid_losses = [loss.item() for loss in valid_losses]
        valid_loss = np.mean(valid_losses)
            
    return valid_loss


def _train(
    args: argparse.Namespace,
    model: RewardModel,
    tokenizer: AutoTokenizer,
    train_loader: DataLoader,
    eval_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler
):
    _fix_seed(args.seed)
    print("[Training]")

    loss_func = nn.MSELoss()
    best_loss = 1e+8
    for epoch in range(1, args.num_epochs+1):
        print(f"[Epoch {epoch}]")
        model.train()

        train_losses = []
        for batch in tqdm(train_loader):
            input_ids, labels = batch
            model_device = next(model.parameters()).device
            input_ids, labels = input_ids.to(model_device), labels.to(model_device)
            preds = model(input_ids)

            loss = loss_func(preds, labels)  # ()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            train_losses.append(loss.detach())

        # Aggregate the epoch result.
        train_losses = [loss.item() for loss in train_losses]
        train_loss = np.mean(train_losses)
        print(f"Train loss: {train_loss}")

        valid_loss =_evaluate(model, eval_loader, loss_func)
        if valid_loss < best_loss:
            best_loss = valid_loss
            print("Best validation loss updated. Checkpointing...")
            model_name = args.sf_model_path.split('/')[-1].split('_')[0]
            ckpt_path = f"{args.ckpt_dir}/{model_name}_rm_epoch={epoch}_loss={best_loss:.4f}"
            if not os.path.isdir(ckpt_path):
                os.makedirs(ckpt_path)
            
            torch.save(model.state_dict(), ckpt_path + "/model.pth")
            tokenizer.save_pretrained(ckpt_path)

        print(f"Best valid loss: {best_loss}")
        print(f"Valid loss: {valid_loss}")
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
    tokenizer = AutoTokenizer.from_pretrained(args.sf_model_path)
    model = RewardModel(
        args.sf_model_path, 
        reward_token_id=tokenizer.eos_token_id,
        max_reward=args.max_reward,
    ).to(device)

    # Preprocess Dataset objects.
    print("[# of data samples after pre-processing]")
    train_set = RMDataset(train_samples, tokenizer, args.max_len, args.min_target_len, args.max_reward)
    eval_set = RMDataset(eval_samples, tokenizer, args.max_len, args.min_target_len, args.max_reward)
    print(f"{len(train_set)} samples processed from train set.")
    print(f"{len(eval_set)} samples processed from eval set.")

    ppd = RMPadCollate(tokenizer.eos_token_id)
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
    optimizer = AdamW(model.parameters())
    scheduler = get_polynomial_decay_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_train_steps,
        power=2
    )

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
    parser.add_argument('--data_dir', type=str, default=".data/rm", help="The name of the directory where data files are stored.")
    parser.add_argument('--sf_model_path', type=str, required=True, help="The checkpoint path of the supervised fine-tuned model.")
    parser.add_argument('--ckpt_dir', type=str, default=".model/sft", help="The name of the directory to save checkpoints.")
    parser.add_argument('--gpu_id', type=int, default=0, help="The GPU ID to use if CUDA is available.")
    parser.add_argument('--max_len', type=int, default=1024, help="The maximum number of tokens.")
    parser.add_argument('--min_target_len', type=int, default=100, help="The minumum number of tokens of target output, except for tags and EOS token.")
    parser.add_argument('--batch_size', type=int, default=16, help="The batch size.")
    parser.add_argument('--num_epochs', type=int, default=1, help="The number of epochs.")
    parser.add_argument('--learning_rate', type=float, default=1e-4, help="The learning rate.")
    parser.add_argument('--warmup_ratio', type=float, default=0.0, help="The ratio of warm-up steps to the total training steps.")
    parser.add_argument('--max_reward', type=float, default=1.0, help="The maximum reward value. The reward range is set to [-max, max].")

    args = parser.parse_args()

    main(args)
