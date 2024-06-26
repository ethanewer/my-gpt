from typing import Sequence
import numpy as np
import torch
from torch import nn, Tensor
import torch.nn.functional as F


class LayerNorm(nn.Module):
    def __init__(self, normalized_shape: Sequence[int], bias=True, eps=1e-5) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.ones(normalized_shape)) if bias else None
        self.eps = eps
    

    def forward(self, x: Tensor) -> Tensor: 
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, self.eps) 
    


class SelfAttention(nn.Module):
    def __init__(self, n_embed: int, n_head: int, dropout: float, bias=True) -> None:
        super().__init__()
        self.n_embed = n_embed
        self.n_head = n_head
        assert n_embed % n_head == 0 
        self.D = n_embed // n_head

        self.c_attn = nn.Linear(n_embed, 3 * n_embed, bias=bias)
        self.c_proj = nn.Linear(n_embed, n_embed, bias=bias)

        self.dropout = dropout
        self.res_dropout = nn.Dropout(dropout)

        self.attn_scale = 1.0 / np.sqrt(self.D)
    

    def forward(self, x: Tensor) -> Tensor:
        B, T, n_embed = x.shape
        assert n_embed == self.n_embed

        q, k, v = self.c_attn(x).split(self.n_embed, dim=2)

        q = q.view(B, T, self.n_head, self.D).transpose(1, 2) # gives shape (B, N, T, D)
        k = k.view(B, T, self.n_head, self.D).transpose(1, 2) # gives shape (B, N, T, D)
        v = v.view(B, T, self.n_head, self.D).transpose(1, 2) # gives shape (B, N, T, D)

        y = F.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, self.n_embed) # concat head outputs 
        y = self.c_proj(y)
        y = self.res_dropout(y)
        return y



class MLP(nn.Module):
    def __init__(self, n_embed: int, dropout: float, bias=True):
        super().__init__()
        self.c_fc = nn.Linear(n_embed, 4 * n_embed, bias=bias)
        self.c_proj = nn.Linear(4 * n_embed, n_embed, bias=bias)
        self.dropout = nn.Dropout(dropout)


    def forward(self, x: Tensor) -> Tensor:
        x = F.gelu(self.c_fc(x))
        x = self.c_proj(x)
        x = self.dropout(x)
        return x



class Block(nn.Module):
    def __init__(self, n_embed: int, n_head: int, dropout: float, bias=True) -> None:
        super().__init__()
        self.ln_1 = LayerNorm(n_embed, bias=bias)
        self.attn = SelfAttention(n_embed, n_head, dropout, bias)
        self.ln_2 = LayerNorm(n_embed, bias=bias)
        self.mlp = MLP(n_embed, dropout, bias)


    def forward(self, x: Tensor) -> Tensor:
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

        args = (n_embed, n_head, dropout, bias)

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(vocab_size, n_embed),
            wpe = nn.Embedding(block_size, n_embed),
            drop = nn.Dropout(dropout),
            h = nn.ModuleList([Block(*args) for _ in range(n_layer)]),
            ln_f = LayerNorm(n_embed, bias=bias),
        ))

        self.lm_head = nn.Linear(n_embed, vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight

        def init_weights(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02) 


        self.apply(init_weights)
    

    def forward(self, x_idx: Tensor) -> Tensor:
        device = x_idx.device
        _, T = x_idx.shape

        assert T <= self.block_size, \
            f"cannot forward sequence of length {T}, block size is only {self.block_size}"
        
        pos = torch.arange(0, T, dtype=torch.int64, device=device)

        tok_emb = self.transformer.wte(x_idx) # shape (B, T, C)
        pos_emb = self.transformer.wpe(pos) # shape (T, C)

        # (B, T, C) + (T, C) = (B, T, C)
        # elementwise addition for each batch
        x = self.transformer.drop(tok_emb + pos_emb)
        for blk in self.transformer.h:
            x = blk(x)
        x = self.transformer.ln_f(x)
        x = self.lm_head(x)
        return x
    

    @torch.no_grad()
    def generate(self, x_idx: Tensor, max_new_tokens: int, temperature=1.0) -> Tensor:
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

            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            x_idx = torch.cat((x_idx, idx_next), dim=1)

        return x_idx    
    
    

def GPT(model_type: str, override_args=None) -> GenerativeTransformer:
    assert model_type in {"gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"}
    override_args = override_args or {} # default to empty dict
    # only dropout can be overridden see more notes below
    assert all(k == "dropout" for k in override_args)
    from transformers import GPT2LMHeadModel
    print("loading weights from pretrained gpt: %s" % model_type)

    # n_layer, n_head and n_embed are determined from model_type
    config_args = {
        "gpt2":         dict(n_layer=12, n_head=12, n_embed=768),  # 124M params
        "gpt2-medium":  dict(n_layer=24, n_head=16, n_embed=1024), # 350M params
        "gpt2-large":   dict(n_layer=36, n_head=20, n_embed=1280), # 774M params
        "gpt2-xl":      dict(n_layer=48, n_head=25, n_embed=1600), # 1558M params
    }[model_type]
    
    print("forcing vocab_size=50257, block_size=1024, bias=True")
    config_args["vocab_size"] = 50257 # always 50257 for GPT model checkpoints
    config_args["block_size"] = 1024 # always 1024 for GPT model checkpoints
    config_args["bias"] = True # always True for GPT model checkpoints
    
    if "dropout" in override_args:
        print(f"overriding dropout rate to {override_args['dropout']}")
        config_args["dropout"] = override_args["dropout"]
    else:
        config_args["dropout"] = 0.0

    model = GenerativeTransformer(**config_args)
    sd = model.state_dict()
    sd_keys = sd.keys()
    # discard this mask / buffer, not a param
    sd_keys = [k for k in sd_keys if not k.endswith(".attn.bias")] 

    # init a huggingface/transformers model
    model_hf = GPT2LMHeadModel.from_pretrained(model_type)
    sd_hf = model_hf.state_dict()

    sd_keys_hf = sd_hf.keys()
    sd_keys_hf = [k for k in sd_keys_hf if not k.endswith(".attn.masked_bias")]
    sd_keys_hf = [k for k in sd_keys_hf if not k.endswith(".attn.bias")]
    transposed = [
        "attn.c_attn.weight", 
        "attn.c_proj.weight", 
        "mlp.c_fc.weight", 
        "mlp.c_proj.weight",
    ]

    assert len(sd_keys_hf) == len(sd_keys), \
        f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
    
    for k in sd_keys_hf:
        if any(k.endswith(w) for w in transposed):
            assert sd_hf[k].shape[::-1] == sd[k].shape
            with torch.no_grad():
                sd[k].copy_(sd_hf[k].T)
        else:
            assert sd_hf[k].shape == sd[k].shape
            with torch.no_grad():
                sd[k].copy_(sd_hf[k])

    return model