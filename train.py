from tokenizer import CharacterTokenizer
from model import MiniGPT
import torch
import torch.nn.functional as F

#--------------------------------------------------------------
# config
model_path = 'experiments/checkpoints/baseline.pt'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
batch_size = 64
block_size = 256
embd_dim = 128
head_dim = 16
n_heads = 3
n_blocks = 5
n_iters = 5000
lr = 3e-4
#--------------------------------------------------------------

with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

tokenizer = CharacterTokenizer(text)

data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]

def get_batch(split):
    data = train_data if split == 'train' else val_data
    idx = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i: i+block_size] for i in idx])
    y = torch.stack([data[i+1: i+block_size+1] for i in idx])
    x, y = x.to(device), y.to(device)
    return x, y

model = MiniGPT(n_blocks=n_blocks, n_heads=n_heads, embd_dim=embd_dim, head_dim=head_dim, vocab_size=tokenizer.vocab_size, block_size=block_size)
model = model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

for i in range(n_iters):

    optimizer.zero_grad()

    x, y = get_batch('train')
    logits = model(x)
    B, T, V = logits.shape
    loss = F.cross_entropy(logits.view(B*T, V), y.view(B*T))
    loss.backward()

    if i % 100 == 0 or i == n_iters -1:
        print(f'TRAIN Iter {i}:', loss.item())
        
    optimizer.step()

x, y = get_batch('val')
logits = model(x)
B, T, V = logits.shape
loss = F.cross_entropy(logits.view(B*T, V), y.view(B*T))
print(f'VAL Iter {i}:', loss.item())

torch.save(model.state_dict(), model_path)
