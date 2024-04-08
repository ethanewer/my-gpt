import numpy as np
import torch
import torch.nn.functional as F
import os
import time
from model import *

OUT_DIR = "out"
EVAL_INTERVAL = 1000
LOG_INTERVAL = 10

DATA_DIR = "data"

BATCH_SIZE = 4
BLOCK_SIZE = 1024

DEVICE = "mps"

lr = 6e-4
WARMUP_ITERS = 2000
LR_DECAY_ITERS = 600000
MIN_LR = 6e-5


def get_batch(split):
    if split == "train":
        data = np.memmap(os.path.join(DATA_DIR, "train.bin"), dtype=np.uint16, mode="r")
    else:
        data = np.memmap(os.path.join(DATA_DIR, "val.bin"), dtype=np.uint16, mode="r")

    def make_block(i):
        return torch.from_numpy((data[i:i + BLOCK_SIZE]).astype(np.int64))
    

    idxs = torch.randint(len(data) - BLOCK_SIZE, (BATCH_SIZE,))
    x = torch.stack([make_block(i) for i in idxs]).to(DEVICE)
    y = torch.stack([make_block(i + 1) for i in idxs]).to(DEVICE)
    return x, y


def get_lr(iter_num):
    if iter_num < WARMUP_ITERS: 
        return lr * iter_num / WARMUP_ITERS 
    
    if iter_num > LR_DECAY_ITERS:
        return MIN_LR
    
    decay_ratio = (iter_num - WARMUP_ITERS) / (LR_DECAY_ITERS - WARMUP_ITERS)
    assert 0 <= decay_ratio and decay_ratio <= 1
    coeff = 0.5 * (1.0 + np.cos(np.pi * decay_ratio))
    return MIN_LR + coeff * (lr - MIN_LR)


def configure_optimizer(params, lr, betas, weight_decay):
    params = [p for p in params if p.requires_grad]
    # any parameters that is 2D will be weight decayed, otherwise no
    # weight tensors in matmuls and embeddings have weight decay
    # biases and layernorms don't have weight decay
    decay_params = [p for p in params if p.dim() >= 2]
    nodecay_params = [p for p in params if p.dim() < 2]
    optim_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": nodecay_params, "weight_decay": 0.0},
    ]
    optimizer = optim.AdamW(optim_groups, lr=lr, betas=betas)
    return optimizer


model = GenerativeTransformer(
    n_embed=768, 
    n_head=12, 
    block_size=BLOCK_SIZE, 
    vocab_size=50304, # 50257 for gpt2
    n_layer=12, 
    dropout=0.0, 
    bias=True,
).to(device=DEVICE)

optimizer = configure_optimizer(
    model.parameters(), 
    lr=lr, 
    betas=(0.9, 0.95), 
    weight_decay=0.1,
)

# checkpoint = torch.load('out/checkpoint.pt')
# model.load_state_dict(checkpoint['model'])
# optimizer.load_state_dict(checkpoint['optimizer'])


@torch.no_grad()
def estimate_loss(n_iters=100):
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(n_iters)
        for k in range(n_iters):
            x, y = get_batch(split)
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, logits.shape[-1]), y.view(-1))
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


def eval_and_save_checkpoint(iter_num):
    losses = estimate_loss()
    print(f"train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
    if losses["val"] < best_val_loss:
        best_val_loss = losses["val"]
        if iter_num > 0:
            checkpoint = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "iter_num": iter_num,
                "best_val_loss": best_val_loss,
            }
            print(f"saving checkpoint to {OUT_DIR}")
            torch.save(checkpoint, os.path.join(OUT_DIR, "checkpoint.pt"))


iter_num = 1
best_val_loss = 1e9

x, y = get_batch("train")
t0 = time.time()

while True:
    lr = get_lr(iter_num)
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    
    logits = model(x)
    loss = F.cross_entropy(logits.view(-1, logits.shape[-1]), y.view(-1))
    
    x, y = get_batch("train")

    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    if iter_num % LOG_INTERVAL == 0:
        t1 = time.time()
        dt = t1 - t0
        t0 = t1
        print(f"iter {iter_num}: loss {loss.item():.4f}, time {dt:.2f}s")

    if iter_num % EVAL_INTERVAL == 0:
        eval_and_save_checkpoint(iter_num)

    iter_num += 1