import numpy as np
from mlx import core as mx, nn, optimizers

import os
import time
from functools import partial
from model import *

OUT_DIR = "out"
EVAL_INTERVAL = 1000
LOG_INTERVAL = 10

DATA_DIR = "../data"

BATCH_SIZE = 4
BLOCK_SIZE = 1024

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
        return mx.array(data[i:i + BLOCK_SIZE].astype(np.int64))
    

    idxs = np.random.randint(0, len(data) - BLOCK_SIZE, [BATCH_SIZE])
    x = mx.stack([make_block(i) for i in idxs])
    y = mx.stack([make_block(i + 1) for i in idxs])
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


model = GenerativeTransformer(
    n_embed=768, 
    n_head=12, 
    block_size=BLOCK_SIZE, 
    vocab_size=50304, # 50257 for gpt2
    n_layer=12, 
    dropout=0.0, 
    bias=True,
)

mx.eval(model.parameters())

model.load_weights(f"{OUT_DIR}/model.npz")


optimizer = optimizers.AdamW(lr, (0.9, 0.95), 1e-7, 0.1)

state = [model.state, optimizer.state]


def loss_fn(model, x, y):
    return nn.losses.cross_entropy(model(x), y, reduction="mean")


@partial(mx.compile, inputs=state, outputs=state)
def train_step(x, y):
    loss, grads = nn.value_and_grad(model, loss_fn)(model, x, y)
    optimizer.update(model, grads)
    return loss


@partial(mx.compile, inputs=state)
def eval_step(x, y):
    return loss_fn(model, x, y)


def estimate_loss(n_iters=1):
    out = {}
    for split in ["train", "val"]:
        losses = np.zeros(n_iters, dtype=np.float32)
        for k in range(n_iters):
            x, y = get_batch(split)
            loss = eval_step(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    return out


iter_num = 1
best_val_loss = float('inf')
t0 = time.time()

while True:
    optimizer.learning_rate = get_lr(iter_num)

    x, y = get_batch("train")

    loss = train_step(x, y)
    mx.eval(state)

    if iter_num % LOG_INTERVAL == 0:
        t1 = time.time()
        dt = t1 - t0
        t0 = t1
        print(f"iter {iter_num}: loss {loss.item():.4f}, time {dt:.2f}s")

    if iter_num % EVAL_INTERVAL == 0:
        losses = estimate_loss()
        print(f"train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        if losses["val"] < best_val_loss:
            best_val_loss = losses["val"]
            model.save_weights(f"{OUT_DIR}/model.npz")        

    iter_num += 1