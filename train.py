import time
import json
import torch
import torch.nn.functional as F
from datetime import datetime
from pathlib import Path
from dataclasses import asdict
from torch.utils.data import Dataset, DataLoader
from tokenizer import CharacterTokenizer
from model import MiniGPT, GPTConfig


timestamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%SZ")

experiment_dir = Path('experiments') / timestamp
experiment_dir.mkdir(parents=True, exist_ok=False)

metrics_path = experiment_dir / 'metrics.jsonl'
checkpoint_last_path = experiment_dir / 'checkpoint_last.pt'
config_path = experiment_dir / 'config.json'

#------------------------------------------------------
seed = 42
dataset: str = 'input.txt'
train_fraction = 0.9
device = 'cuda' if torch.cuda.is_available() else 'cpu'
batch_size: int = 256
learning_rate: float = 3e-4
n_epochs: int = 3
# n_iters: int = 10000 # works with get_batch() that was replaced with DataLoader; use n_epochs
#------------------------------------------------------
torch.manual_seed(seed)
#------------------------------------------------------
with open(dataset, 'r', encoding='utf-8') as f:
    text = f.read()

tokenizer = CharacterTokenizer(text)

data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
n = int(train_fraction*len(data))
train_data = data[:n]
val_data = data[n:]
#------------------------------------------------------

config = GPTConfig()

model = MiniGPT(config)
model = model.to(device)

#------------------------------------------------------
with config_path.open('w') as f:
    json.dump(
        {
        'created_at': timestamp,
        'seed': seed,
        'device': device,
        'dataset': dataset,
        'train_fraction': train_fraction,
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'n_epochs': n_epochs,
        'model_config': asdict(config),
        'tokenizer_chars': tokenizer.chars}, f
        )
#------------------------------------------------------
def get_batch(split):
    data = train_data if split == 'train' else val_data
    idx = torch.randint(len(data) - config.block_size, (batch_size,))
    x = torch.stack([data[i: i+config.block_size] for i in idx])
    y = torch.stack([data[i+1: i+config.block_size+1] for i in idx])
    x, y = x.to(device), y.to(device)
    return x, y
#------------------------------------------------------
class TextDataset(Dataset):
    def __init__(self, data, block_size):
        self.data = data
        self.block_size = block_size

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        x = self.data[idx:idx + self.block_size]
        y = self.data[idx + 1:idx + self.block_size + 1]

        return x, y

train_dataset = TextDataset(train_data, config.block_size)
val_dataset = TextDataset(val_data, config.block_size)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
#------------------------------------------------------

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

#------------------------------------------------------
def compute_loss(logits, targets):
    B, T, V = logits.shape # batch_size, sequence_length, vocab_size
    return F.cross_entropy(logits.view(B*T, V), targets.view(B*T))

def log_metrics(epoch: int, train_loss: float, val_loss: float) -> None:
    with metrics_path.open('a') as f:
        f.write(
            json.dumps({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'val_loss': val_loss
            }) + "\n")
#------------------------------------------------------
# train_iter = iter(train_loader)

for epoch in range(n_epochs):

    start = time.perf_counter()

    model.train()

    train_loss = 0.0
    n_batches = 0
    # x, y = get_batch('train')
    # x, y = next(train_iter)
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)

        logits = model(x)
        loss = compute_loss(logits, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()    
        optimizer.step()

        train_loss += loss.item()
        n_batches += 1

    train_loss /= n_batches

    #------------------------------------------------------
    if epoch % 1 == 0 or epoch == n_epochs - 1:
        model.eval()

        val_loss = 0.0
        n_val_batches = 0

        for x_val, y_val in val_loader:
            x_val, y_val = x_val.to(device), y_val.to(device)
            with torch.no_grad():
                logits = model(x_val)
            val_loss += compute_loss(logits, y_val).item()
            n_val_batches += 1

        val_loss /= n_val_batches

        log_metrics(epoch, train_loss, val_loss)
    #------------------------------------------------------
    epoch_time = time.perf_counter() - start
    print(f'Epoch time: {epoch_time:.4f} seconds')
#------------------------------------------------------
torch.save(
    {
    'epoch': epoch + 1,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'model_config': asdict(config),
    'tokenizer_chars': tokenizer.chars
    }, checkpoint_last_path
)

