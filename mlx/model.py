import numpy as np
from mlx import core as mx
from mlx import nn


class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, bias=True, eps=1e-5):
        super().__init__()
        self.weight = mx.ones(normalized_shape)
        self.bias = mx.ones(normalized_shape) if bias else None
        self.eps = eps
    

    def __call__(self, x): 
        return mx.fast.layer_norm(x, self.weight, self.bias, self.eps) 
    


class SelfAttention(nn.Module):
    def __init__(
        self, n_embed: int, n_head: int, 
        mask: mx.array, dropout: float, bias=True
    ) -> None:
        super().__init__()
        self.n_embed = n_embed
        self.n_head = n_head
        assert n_embed % n_head == 0 
        self.D = n_embed // n_head

        self.c_attn = nn.Linear(n_embed, 3 * n_embed, bias=bias)
        self.c_proj = nn.Linear(n_embed, n_embed, bias=bias)
        self.dropout = nn.Dropout(dropout)

        self.mask = mask
        self.scale = 1.0 / np.sqrt(self.D)
    

    def __call__(self, x: mx.array):
        B, T, n_embed = x.shape
        assert n_embed == self.n_embed

        tmp = self.c_attn(x)
        tmp = tmp.split(self.n_embed, axis=2)

        q, k, v = mx.split(self.c_attn(x), 3, axis=2)

        # reshape to (B, N, T, D)
        q = q.reshape((B, T, self.n_head, self.D)).transpose((0, 2, 1, 3)) 
        k = k.reshape((B, T, self.n_head, self.D)).transpose((0, 2, 1, 3))
        v = v.reshape((B, T, self.n_head, self.D)).transpose((0, 2, 1, 3))
        
        y = mx.fast.scaled_dot_product_attention(
            q, k, v, 
            mask=self.mask[:T, :T], 
            scale=self.scale,
        )
        
        y = y.transpose((0, 2, 1, 3)).reshape((B, T, self.n_embed)) # concat head outputs 
        y = self.c_proj(y)
        y = self.dropout(y)
        return y



class MLP(nn.Module):
    def __init__(self, n_embed, dropout, bias=True):
        super().__init__()
        self.c_fc = nn.Linear(n_embed, 4 * n_embed, bias=bias)
        self.c_proj = nn.Linear(4 * n_embed, n_embed, bias=bias)
        self.dropout = nn.Dropout(dropout)


    def __call__(self, x):
        x = nn.gelu_fast_approx(self.c_fc(x))
        x = self.c_proj(x)
        x = self.dropout(x)
        return x



class Block(nn.Module):
    def __init__(
        self, n_embed: int, n_head: int, 
        mask: mx.array, dropout: float, bias=True,
    ) -> None:
        super().__init__()
        self.ln_1 = LayerNorm(n_embed, bias=bias)
        self.attn = SelfAttention(n_embed, n_head, mask, dropout, bias)
        self.ln_2 = LayerNorm(n_embed, bias=bias)
        self.mlp = MLP(n_embed, dropout, bias)


    def __call__(self, x: mx.array):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x



class GenerativeTransformer(nn.Module):
    def __init__(
        self, n_embed: int, n_head: int, block_size: int, 
        vocab_size: int, n_layer: int, dropout: float, bias=True,
    ) -> None:
        super().__init__()
        self.block_size = block_size
        self.wte = nn.Embedding(vocab_size, n_embed)
        self.wpe = nn.Embedding(block_size, n_embed)
        self.drop = nn.Dropout(dropout)

        mask = np.zeros((block_size, block_size), dtype=np.float32)
        mask[np.tril(np.ones((block_size, block_size))) == 0] = -np.inf
        mask = mx.array(mask)
        
        self.h = [Block(n_embed, n_head, mask, dropout, bias) for _ in range(n_layer)]
        self.ln_f = LayerNorm(n_embed, bias=bias)

        self.lm_head = nn.Linear(n_embed, vocab_size, bias=False)
        self.wte.weight = self.lm_head.weight

        def init_weights(_, m: nn.Module):
            if isinstance(m, nn.Linear) or isinstance(m, nn.Embedding):
                m.weight = nn.init.normal(0.0, 0.02)(m.weight)
                if hasattr(m, "bias") and m.bias is not None:
                    m.bias = mx.zeros_like(m.bias)
        

        self.apply_to_modules(init_weights)
                
    

    def __call__(self, x_idx: mx.array):
        _, T = x_idx.shape

        assert T <= self.block_size, \
            f"cannot forward sequence of length {T}, block size is only {self.block_size}"
        
        pos = mx.arange(0, T, dtype=mx.int64)

        tok_emb = self.wte(x_idx) # shape (B, T, C)
        pos_emb = self.wpe(pos) # shape (T, C)

        # (B, T, C) + (T, C) = (B, T, C)
        # elementwise addition for each batch
        x = self.drop(tok_emb + pos_emb)
        for blk in self.h:
            x = blk(x)
        x = self.ln_f(x)
        x = self.lm_head(x)
        return x
    

    def generate(self, x_idx: mx.array, max_new_tokens: int, temperature=1.0):
        # Take a conditioning sequence of indices x_idx (int64 tensor of shape (B, T)) and 
        # complete the sequence max_new_tokens times, feeding the predictions back into 
        # the model each time. Most likely you"ll want to make sure to be in model.eval() 
        # mode of operation for this.
        for _ in range(max_new_tokens):
            if x_idx.shape[1] <= self.block_size:
                x_idx_cropped = x_idx 
            else:
                x_idx_cropped = x_idx[:, -self.block_size:]

            logits = self(x_idx_cropped)
            logits = logits[:, -1, :] / temperature
            next_idx = mx.random.categorical(logits)[None]
            x_idx = mx.concatenate((x_idx, next_idx), axis=1)
        return x_idx   