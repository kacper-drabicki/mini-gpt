import torch
import torch.nn as nn
import torch.nn.functional as F

class Head(nn.Module):

    def __init__(self, embd_dim: int, head_dim: int, block_size: int):
        super().__init__()
        self.query = nn.Linear(embd_dim, head_dim, bias=False)
        self.key = nn.Linear(embd_dim, head_dim, bias=False)
        self.value = nn.Linear(embd_dim, head_dim, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

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

    def __init__(self, n_heads: int, embd_dim: int, head_dim: int, block_size: int):
        super().__init__()
        self.heads = nn.ModuleList([Head(embd_dim, head_dim, block_size) for _ in range(n_heads)])
        self.project = nn.Linear(n_heads * head_dim, embd_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.cat([head(x) for head in self.heads], dim=-1)
        out = self.project(out)

        return out # (B,T,C)

class FeedForward(nn.Module):

    def __init__(self, embd_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embd_dim, 4*embd_dim),
            nn.GELU(),
            nn.Linear(4*embd_dim, embd_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x) # (B,T,C)


class Block(nn.Module):

    def __init__(self, n_heads: int, embd_dim: int, head_dim: int, block_size: int):
        super().__init__()
        self.attention = MultiHeadAttention(n_heads, embd_dim, head_dim, block_size)
        self.ffwd = FeedForward(embd_dim)
        self.ln1 = nn.LayerNorm(embd_dim)
        self.ln2 = nn.LayerNorm(embd_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x # (B,T,C)


class MiniGPT(nn.Module):

    def __init__(self, n_blocks: int, n_heads: int, embd_dim: int, head_dim: int, vocab_size: int, block_size: int):
        super().__init__()
        self.block_size = block_size
        self.token_embedding = nn.Embedding(vocab_size, embd_dim)
        self.pos_embedding = nn.Embedding(block_size, embd_dim)
        self.blocks = nn.Sequential(*[Block(n_heads, embd_dim, head_dim, block_size) for _ in range(n_blocks)])
        self.ln = nn.LayerNorm(embd_dim)
        self.lm_head = nn.Linear(embd_dim, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B,T = x.shape

        token_emb = self.token_embedding(x)
        pos_emb = self.pos_embedding(torch.arange(T, device=x.device))
        x = token_emb + pos_emb
        x = self.blocks(x)
        x = self.ln(x)
        logits = self.lm_head(x)

        return logits

    def generate(self, x, max_new_tokens: int = 500):

        for _ in range(max_new_tokens):
            x_cut = x[:, -self.block_size:]
            logits = self(x_cut)
            logits = logits[:,-1,:]
            probs = F.softmax(logits, dim=-1)
            idx = torch.multinomial(probs, num_samples=1)
            x = torch.cat((x, idx), dim=-1)
        return x
