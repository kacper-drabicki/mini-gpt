import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

@dataclass
class GPTConfig:
    block_size: int = 256 # maximum sequence length
    vocab_size: int = 50257
    embd_dim: int = 128
    head_dim: int = 16
    dropout_rate: float = 0.2
    n_heads: int = 3
    n_blocks: int = 5
    

class Head(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.query = nn.Linear(config.embd_dim, config.head_dim, bias=False)
        self.key = nn.Linear(config.embd_dim, config.head_dim, bias=False)
        self.value = nn.Linear(config.embd_dim, config.head_dim, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(config.block_size, config.block_size)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B,T,C = x.shape # batch_size, time_steps/block_size, channels/embd_dim

        q = self.query(x) # (B,T,C) -> (B,T,H)
        k = self.key(x) # (B,T,C) -> (B,T,H)

        weights = q @ k.transpose(-2, -1) # (B,T,H) @ (B,H,T) -> (B,T,T)
        weights = weights / k.shape[-1] ** 0.5
        weights = weights.masked_fill(self.tril[:T,:T] == 0, float('-inf'))
        probs = F.softmax(weights, dim=-1)

        v = self.value(x) # (B,T,C) -> (B,T,H)
        out = probs @ v # (B,T,T) @ (B,T,H) -> (B,T,H)
        return out


class MultiHeadAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.heads = nn.ModuleList([Head(config) for _ in range(config.n_heads)])
        self.project = nn.Linear(config.n_heads * config.head_dim, config.embd_dim, bias=False)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.cat([head(x) for head in self.heads], dim=-1)
        out = self.project(out)
        out = self.dropout(out)

        return out # (B,T,C)

class FeedForward(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.embd_dim, 4*config.embd_dim),
            nn.GELU(),
            nn.Linear(4*config.embd_dim, config.embd_dim),
            nn.Dropout(config.dropout_rate)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x) # (B,T,C)


class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.attention = MultiHeadAttention(config)
        self.ffwd = FeedForward(config)
        self.ln1 = nn.LayerNorm(config.embd_dim)
        self.ln2 = nn.LayerNorm(config.embd_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x # (B,T,C)


class MiniGPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.block_size = config.block_size
        self.token_embedding = nn.Embedding(config.vocab_size, config.embd_dim)
        self.pos_embedding = nn.Embedding(config.block_size, config.embd_dim)
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_blocks)])
        self.ln = nn.LayerNorm(config.embd_dim)
        self.lm_head = nn.Linear(config.embd_dim, config.vocab_size)

         # Initialize all parameters
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B,T = x.shape

        token_emb = self.token_embedding(x)
        pos_emb = self.pos_embedding(torch.arange(T, device=x.device))
        x = token_emb + pos_emb
        x = self.blocks(x)
        x = self.ln(x)
        logits = self.lm_head(x)

        return logits

    @torch.no_grad()
    def generate(self, x, max_new_tokens: int = 500):

        for _ in range(max_new_tokens):
            x_cut = x[:, -self.block_size:]
            logits = self(x_cut)
            logits = logits[:,-1,:]
            probs = F.softmax(logits, dim=-1)
            idx = torch.multinomial(probs, num_samples=1)
            x = torch.cat((x, idx), dim=-1)
        return x
