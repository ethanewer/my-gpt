import numpy as np
import torch
from torch import optim
import torch.nn.functional as F
import os
import time
from model import *

OUT_DIR = "out"
EVAL_INTERVAL = 1000
LOG_INTERVAL = 1

DATA_DIR = "../data"

BATCH_SIZE = 3
BLOCK_SIZE = 1024

DEVICE = "mps"
DTYPE = torch.bfloat16

START_LR = 1e-3
WARMUP_ITERS = 2000
LR_DECAY_ITERS = 600000
MIN_LR = 1e-4


def get_batch(split: str) -> tuple[Tensor, Tensor]:
    if split == "train":
        data = np.memmap(os.path.join(DATA_DIR, "train.bin"), dtype=np.uint16, mode="r")
    else:
        data = np.memmap(os.path.join(DATA_DIR, "val.bin"), dtype=np.uint16, mode="r")

    def make_block(i):
        return torch.from_numpy(data[i:i + BLOCK_SIZE].astype(np.int64))
    

    idxs = torch.randint(len(data) - BLOCK_SIZE, (BATCH_SIZE,))
    x = torch.stack([make_block(i) for i in idxs]).to(DEVICE)
    y = torch.stack([make_block(i + 1) for i in idxs]).to(DEVICE)
    return x, y


def get_lr(i: int) -> float:
    if i < WARMUP_ITERS: 
        return START_LR * i / WARMUP_ITERS 
    
    if i > LR_DECAY_ITERS:
        return MIN_LR
    
    decay_ratio = (i - WARMUP_ITERS) / (LR_DECAY_ITERS - WARMUP_ITERS)
    assert 0 <= decay_ratio and decay_ratio <= 1
    coeff = 0.5 * (1.0 + np.cos(np.pi * decay_ratio))
    return MIN_LR + coeff * (START_LR - MIN_LR)


def configure_optimizer(
    params: Sequence[nn.Parameter], lr: float, 
    betas: tuple[float, float], weight_decay: float,
) -> optim.Adam:
    grad_params = [p for p in params if p.requires_grad]
    decay_params = [p for p in grad_params if p.dim() >= 2]
    nodecay_params = [p for p in grad_params if p.dim() < 2]
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
).to(device=DEVICE, dtype=DTYPE)

optimizer = configure_optimizer(
    model.parameters(), 
    lr=lr, 
    betas=(0.9, 0.95), 
    weight_decay=0.1,
)

# checkpoint = torch.load(f"{OUT_DIR}/checkpoint.pt")
# model.load_state_dict(checkpoint["model"])
# optimizer.load_state_dict(checkpoint["optimizer"])


@torch.no_grad()
def evaluate_loss(n_iters=50) -> dict[str, float]:
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


def eval_and_save_checkpoint(iter_num: int) -> None:
    losses = evaluate_loss()
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
            torch.save(checkpoint, os.path.join(OUT_DIR, "checkpoint.pt"))
            print(f"saved checkpoint to {OUT_DIR}")


iter_num = 1
best_val_loss = float("inf")
t0 = time.time()

while True:
    lr = get_lr(iter_num)
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    
    x, y = get_batch("train")

    logits = model(x)
    loss = F.cross_entropy(logits.view(-1, logits.shape[-1]), y.view(-1))
    
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