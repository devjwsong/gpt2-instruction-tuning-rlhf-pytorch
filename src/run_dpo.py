import argparse
import json
import os
import math

from copy import deepcopy
from tqdm import tqdm
from _util import _fix_seed, PrefDataset, PrefPadCollate
from transformers import AutoTokenizer, AutoModelForCausalLM, get_polynomial_decay_schedule_with_warmup
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
import torch
import numpy as np


def _get_logprobs(
    model: AutoModelForCausalLM,
    input_ids: torch.LongTensor,
    labels: torch.LongTensor
) -> torch.Tensor:
    logits = model(input_ids, output_hidden_states=True).logits  # (B, L, V)

    # Shift for next word prediction mapping
    logits, labels = logits[:, :-1, :], labels[:, 1:].clone()  # (B, L-1, V), (B, L-1)
    masks = (labels != -100)  # (B, L-1)
    labels[labels == -100] = 0  # For gathering, -100 cannot be used.

    # Pick the response tokens from logits and calculate the log probabilities.
    logprobs = torch.gather(F.log_softmax(logits, dim=-1), dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)  # (B, L-1)
    logprobs *= masks  # (B, L-1)

    return logprobs.sum(dim=-1)  # (B)


def _evaluate(
    args: argparse.Namespace,
    model: AutoModelForCausalLM,
    ref_model: AutoModelForCausalLM,
    eval_loader: DataLoader
) -> float:
    print("[Validation]")
    model.eval()

    valid_losses = []
    for batch in tqdm(eval_loader):
        chosen_input_ids, rejected_input_ids, chosen_labels, rejected_labels = batch
        chosen_input_ids, rejected_input_ids, chosen_labels, rejected_labels = \
            chosen_input_ids.to(model.device), rejected_input_ids.to(model.device), chosen_labels.to(model.device), rejected_labels.to(model.device)
        
        with torch.no_grad():
            cur_chosen_logprobs = _get_logprobs(model, chosen_input_ids, chosen_labels)  # (B)
            cur_rejected_logprobs = _get_logprobs(model, rejected_input_ids, rejected_labels)  # (B)
            ref_chosen_logprobs = _get_logprobs(ref_model, chosen_input_ids, chosen_labels)  # (B)
            ref_rejected_logprobs = _get_logprobs(ref_model, rejected_input_ids, rejected_labels)  # (B)

        # Finalize the loss.
        chosen_kl_divs = cur_chosen_logprobs - ref_chosen_logprobs
        rejected_kl_divs = cur_rejected_logprobs - ref_rejected_logprobs
        loss = -1 * torch.log(torch.sigmoid(args.beta * (chosen_kl_divs - rejected_kl_divs)))  # (B)
        loss = loss.mean()  # ()
        valid_losses.append(loss.detach().item())

    valid_loss = np.mean(valid_losses)
    return valid_loss


def _train(
    args: argparse.Namespace,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    train_loader: DataLoader,
    eval_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler
):
    scaler = torch.cuda.amp.GradScaler()

    _fix_seed(args.seed)
    print("[Training]")
    ref_model = deepcopy(model)
    ref_model.eval()

    best_loss = 1e+8
    for epoch in range(1, args.num_epochs+1):
        print(f"[Epoch {epoch}]")
        model.train()

        train_losses = []
        for b, batch in enumerate(tqdm(train_loader)):
            chosen_input_ids, rejected_input_ids, chosen_labels, rejected_labels = batch
            chosen_input_ids, rejected_input_ids, chosen_labels, rejected_labels = \
                chosen_input_ids.to(model.device), rejected_input_ids.to(model.device), chosen_labels.to(model.device), rejected_labels.to(model.device)
            
            # Calculate 4 logprobs: 1) Model + Chosen, 2) Model + Rejected, 3) Ref model + Chosen, 4) Ref model + Rejected
            if args.use_fp16:
                with torch.cuda.amp.autocast():
                    cur_chosen_logprobs = _get_logprobs(model, chosen_input_ids, chosen_labels)  # (B)
                    cur_rejected_logprobs = _get_logprobs(model, rejected_input_ids, rejected_labels)  # (B)
                    with torch.no_grad():
                        ref_chosen_logprobs = _get_logprobs(ref_model, chosen_input_ids, chosen_labels)  # (B)
                        ref_rejected_logprobs = _get_logprobs(ref_model, rejected_input_ids, rejected_labels)  # (B)

                # Finalize the loss.
                chosen_kl_divs = cur_chosen_logprobs - ref_chosen_logprobs
                rejected_kl_divs = cur_rejected_logprobs - ref_rejected_logprobs
                loss = -1 * torch.log(torch.sigmoid(args.beta * (chosen_kl_divs - rejected_kl_divs)))  # (B)
                loss = loss.mean()  # ()

                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scheduler.step()
                scaler.update()

            else:
                cur_chosen_logprobs = _get_logprobs(model, chosen_input_ids, chosen_labels)  # (B)
                cur_rejected_logprobs = _get_logprobs(model, rejected_input_ids, rejected_labels)  # (B)
                with torch.no_grad():
                    ref_chosen_logprobs = _get_logprobs(ref_model, chosen_input_ids, chosen_labels)  # (B)
                    ref_rejected_logprobs = _get_logprobs(ref_model, rejected_input_ids, rejected_labels)  # (B)

                # Finalize the loss.
                chosen_kl_divs = cur_chosen_logprobs - ref_chosen_logprobs
                rejected_kl_divs = cur_rejected_logprobs - ref_rejected_logprobs
                loss = -1 * torch.log(torch.sigmoid(args.beta * (chosen_kl_divs - rejected_kl_divs)))  # (B)
                loss = loss.mean()  # ()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()

            train_losses.append(loss.detach().item())
            if b % args.log_step == 0:
                print(f"Step {b} / Training loss: {train_losses[-1]}")

        # Aggregate the epoch result.
        train_loss = np.mean(train_losses)
        print(f"Train loss: {train_loss}")

        valid_loss =_evaluate(args, model, ref_model, eval_loader)
        if valid_loss < best_loss:
            best_loss = valid_loss
            print("Best validation loss updated. Checkpointing...")
            model_name = args.sft_model_path.split('/')[-1].split('_')[0]
            ckpt_path = f"{args.ckpt_dir}/{model_name}_dpo_epoch=epoch={epoch}_loss={best_loss:.4f}"
            model.save_pretrained(ckpt_path)
            tokenizer.save_pretrained(ckpt_path)

        print(f"Best valid loss: {best_loss}")
        print(f"Valid loss: {valid_loss}")
        print()


def main(args: argparse.Namespace):
    # Set the GPU.
    device = torch.device(f"cuda:{args.gpu_id}") if torch.cuda.is_available() else torch.device('cpu')

    # Load the models.
    tokenizer = AutoTokenizer.from_pretrained(args.sft_model_path)
    model = AutoModelForCausalLM.from_pretrained(args.sft_model_path).to(device)

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
    
    # Set the optimizer and learning rate scheduler.
    num_batches = len(train_loader)
    total_train_steps = args.num_epochs * num_batches
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    scheduler = get_polynomial_decay_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
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
    parser.add_argument('--seed', type=int, default=42, help="The random seed for data shuffling.")
    parser.add_argument('--sft_model_path', type=str, required=True, help="The checkpoint path of the supervised fine-tuned model.")
    parser.add_argument('--ckpt_dir', type=str, default=".model/dpo", help="The name of the directory to save checkpoints.")
    parser.add_argument('--gpu_id', type=int, default=0, help="The GPU ID to use if CUDA is available.")
    parser.add_argument('--data_dir', type=str, default=".data/pref", help="The name of the directory where data files are stored.")
    parser.add_argument('--max_len', type=int, default=1024, help="The maximum number of tokens.")
    parser.add_argument('--min_gen_len', type=int, default=1, help="The minumum number of tokens to generate, except for tags and EOS token.")
    parser.add_argument('--num_epochs', type=int, default=1, help="The number of epochs.")
    parser.add_argument('--log_step', type=int, default=100, help="The training step period to log the loss.")
    parser.add_argument('--batch_size', type=int, default=16, help="The batch size.")
    parser.add_argument('--learning_rate', type=float, default=1e-5, help="The learning rate.")
    parser.add_argument('--beta', type=float, default=0.1, help="The coefficient for per-token KL divergence penalty.")
    parser.add_argument('--use_fp16', action='store_true', help="Whether to use float16 mixed precision or not.")

    args = parser.parse_args()

    main(args)
