"""
MoE GPT Training Script for MoE Comparison
Based on nanoGPT architecture
"""

import os
import time
import math
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import GPT2Tokenizer

from model import GPTConfig, GPT

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
# I/O
out_dir = 'out-moe'
eval_interval = 500  # Evaluate every N iterations
log_interval = 10    # Log every N iterations
eval_iters = 50      # Number of iterations for evaluation
always_save_checkpoint = True

# Data
dataset_name = 'wikitext'
dataset_config = 'wikitext-103-v1'
max_seq_length = 256

# Model (MoE with 8 experts)
n_layer = 4
n_head = 4
n_embd = 256
block_size = 256
dropout = 0.0
bias = True

# MoE Configuration
use_moe = True
num_experts = 8
top_k = 1

# Training
batch_size = 8
gradient_accumulation_steps = 4  # Effective batch size = 8 * 4 = 32
max_iters = 1000     # Total training iterations
learning_rate = 3e-4
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0

# Learning rate decay
decay_lr = True
warmup_iters = 100
lr_decay_iters = max_iters
min_lr = 3e-5

# System
device = 'cuda' if torch.cuda.is_available() else 'cpu'
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
compile_model = False  # torch.compile (requires PyTorch 2.0+)

# -----------------------------------------------------------------------------
# Training helpers
# -----------------------------------------------------------------------------
def get_lr(it):
    """Learning rate decay scheduler (cosine with warmup)"""
    # 1) Linear warmup
    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)
    # 2) Constant at min_lr after decay
    if it > lr_decay_iters:
        return min_lr
    # 3) Cosine decay
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)

def estimate_loss(model, train_loader, val_loader, ctx, device, eval_iters):
    """Estimate loss on train and val sets"""
    out = {}
    model.eval()
    
    for split, loader in [('train', train_loader), ('val', val_loader)]:
        losses = []
        for k, batch in enumerate(loader):
            if k >= eval_iters:
                break
            batch = batch.to(device)
            with torch.no_grad():
                with ctx:
                    logits, loss = model(batch, targets=batch)
            losses.append(loss.item())
        out[split] = sum(losses) / len(losses) if losses else 0.0
    
    model.train()
    return out

# -----------------------------------------------------------------------------
# Main training function
# -----------------------------------------------------------------------------
def main():
    print("="*70)
    print("MOE GPT TRAINING")
    print("="*70)
    print(f"Device: {device}")
    print(f"Dtype: {dtype}")
    print(f"Model: {n_layer}L, {n_head}H, {n_embd}D (MoE: {num_experts} experts, top_k={top_k})")
    print(f"Batch size: {batch_size} × {gradient_accumulation_steps} = {batch_size * gradient_accumulation_steps}")
    print(f"Max iterations: {max_iters}")
    print("="*70)

    # Setup
    os.makedirs(out_dir, exist_ok=True)
    torch.manual_seed(1337)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    device_type = 'cuda' if 'cuda' in device else 'cpu'
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
    ctx = torch.amp.autocast(device_type=device_type, dtype=ptdtype) if device_type == 'cuda' else torch.amp.autocast(device_type=device_type, enabled=False)

    # Data Loading
    print("\nLoading dataset...")
    data_load_start = time.time()
    
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    # Load dataset
    train_dataset = load_dataset(dataset_name, dataset_config, split="train")
    val_dataset = load_dataset(dataset_name, dataset_config, split="validation")

    print(f"Train examples: {len(train_dataset):,}")
    print(f"Val examples: {len(val_dataset):,}")

    # Tokenize
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_seq_length,
            padding='max_length',
            return_tensors='pt'
        )

    print("Tokenizing train dataset...")
    train_dataset = train_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["text"],
        desc="Tokenizing train"
    )

    print("Tokenizing validation dataset...")
    val_dataset = val_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["text"],
        desc="Tokenizing val"
    )

    # DataLoaders - Fixed collate function
    def collate_fn(batch):
        # Extract input_ids and convert to tensor
        input_ids = [item['input_ids'] for item in batch]
        return torch.tensor(input_ids)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
        collate_fn=collate_fn
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        collate_fn=collate_fn
    )

    data_load_time = time.time() - data_load_start
    print(f"✓ Data loaded: {len(train_loader)} train batches, {len(val_loader)} val batches")
    print(f"  Data loading time: {data_load_time:.2f}s")

    # Model
    print("\nInitializing model...")
    model_init_start = time.time()
    
    model_args = dict(
        n_layer=n_layer,
        n_head=n_head,
        n_embd=n_embd,
        block_size=block_size,
        bias=bias,
        vocab_size=50304,
        dropout=dropout,
        use_moe=use_moe,          # Enable MoE
        num_experts=num_experts,  # 8 experts
        top_k=top_k               # Top-1 routing
    )

    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    model.to(device)

    # Compile model (optional, requires PyTorch 2.0+)
    if compile_model:
        print("Compiling model...")
        model = torch.compile(model)

    model_init_time = time.time() - model_init_start
    print(f"  Model initialization time: {model_init_time:.2f}s")

    # Optimizer
    print("Setting up optimizer...")
    optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
    
    # GradScaler - Fixed for compatibility
    if device_type == 'cuda':
        try:
            scaler = torch.amp.GradScaler('cuda', enabled=(dtype == 'float16'))
        except:
            scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))
    else:
        scaler = torch.cuda.amp.GradScaler(enabled=False)

    # Training loop
    print("\n" + "="*70)
    print("STARTING TRAINING")
    print("="*70)

    # START TIMING
    training_start_time = time.time()
    
    train_loader_iter = iter(train_loader)
    best_val_loss = 1e9
    t0 = time.time()
    local_iter_num = 0
    running_mfu = -1.0

    for iter_num in range(max_iters):
        
        # Determine learning rate
        lr = get_lr(iter_num) if decay_lr else learning_rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
        # Evaluate and save checkpoint
        if iter_num % eval_interval == 0:
            losses = estimate_loss(model, train_loader, val_loader, ctx, device, eval_iters)
            print(f"\nstep {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
            
            if losses['val'] < best_val_loss or always_save_checkpoint:
                best_val_loss = losses['val']
                if iter_num > 0:
                    checkpoint = {
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'model_args': model_args,
                        'iter_num': iter_num,
                        'best_val_loss': best_val_loss,
                    }
                    checkpoint_path = os.path.join(out_dir, 'ckpt.pt')
                    print(f"Saving checkpoint to {checkpoint_path}")
                    torch.save(checkpoint, checkpoint_path)
        
        # Training step with gradient accumulation
        optimizer.zero_grad(set_to_none=True)
        loss_accum = 0.0
        
        for micro_step in range(gradient_accumulation_steps):
            # Get batch
            try:
                batch = next(train_loader_iter)
            except StopIteration:
                train_loader_iter = iter(train_loader)
                batch = next(train_loader_iter)
            
            batch = batch.to(device)
            
            # Forward pass
            with ctx:
                logits, loss = model(batch, targets=batch)
                loss = loss / gradient_accumulation_steps
            
            # Backward pass
            scaler.scale(loss).backward()
            loss_accum += loss.item()
        
        # Gradient clipping
        if grad_clip != 0.0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        
        # Optimizer step
        scaler.step(optimizer)
        scaler.update()
        
        # Timing and logging
        t1 = time.time()
        dt = t1 - t0
        t0 = t1
        
        if iter_num % log_interval == 0:
            lossf = loss_accum
            if local_iter_num >= 5:
                mfu = model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
                running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu
            print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%")
        
        local_iter_num += 1

    # END TIMING
    training_end_time = time.time()
    total_training_time = training_end_time - training_start_time

    print("\n" + "="*70)
    print("TRAINING COMPLETE")
    print("="*70)
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Total training time: {total_training_time:.2f}s ({total_training_time/60:.2f} minutes)")
    print(f"Average time per iteration: {total_training_time/max_iters:.3f}s")
    print(f"Data loading time: {data_load_time:.2f}s")
    print(f"Model init time: {model_init_time:.2f}s")
    print(f"Pure training time: {total_training_time:.2f}s")
    print(f"Checkpoints saved to: {out_dir}/")

# -----------------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------------
if __name__ == '__main__':
    main()