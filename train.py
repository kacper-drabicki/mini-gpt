import time
import json
import torch
import torch.nn.functional as F
from datetime import datetime
from pathlib import Path
from dataclasses import asdict
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
n_steps: int = 5000
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
train_data, val_data = train_data.to(device), val_data.to(device)
#------------------------------------------------------

config = GPTConfig()

raw_model = MiniGPT(config)
raw_model = raw_model.to(device)

model = torch.compile(raw_model)

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
        'n_steps': n_steps,
        'model_config': asdict(config),
        'tokenizer_chars': tokenizer.chars}, f
        )
#------------------------------------------------------
def get_batch(split: str):
    data = train_data if split == 'train' else val_data
    idx = torch.randint(len(data) - config.block_size, (batch_size,))
    x = torch.stack([data[i: i+config.block_size] for i in idx])
    y = torch.stack([data[i+1: i+config.block_size+1] for i in idx])
    return x, y
#------------------------------------------------------

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

#------------------------------------------------------
def compute_loss(logits, targets):
    B, T, V = logits.shape # batch_size, sequence_length, vocab_size
    return F.cross_entropy(logits.view(B*T, V), targets.view(B*T))

def log_metrics(step: int, train_loss: float, val_loss: float) -> None:
    with metrics_path.open('a') as f:
        f.write(
            json.dumps({
            'step': step + 1,
            'train_loss': train_loss,
            'val_loss': val_loss
            }) + "\n")

tokens_per_step = batch_size * config.block_size
#------------------------------------------------------
for step in range(n_steps):

    start = time.perf_counter()

    model.train()

    x, y = get_batch('train')
    # automatically use BF16 where beneficial for faster, lower-memory computation
    with torch.autocast(device_type=device, dtype=torch.bfloat16): 
        logits = model(x)
        loss = compute_loss(logits, y)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()    
    optimizer.step()

    #------------------------------------------------------
    if step % 100 == 0 or step == n_steps - 1:
        model.eval()

        x_val, y_val = get_batch('val')
        with torch.no_grad():
            logits = model(x_val)
        val_loss = compute_loss(logits, y_val)

        log_metrics(step, loss.item(), val_loss.item())
    #------------------------------------------------------
    step_time = time.perf_counter() - start
    tokens_per_second = tokens_per_step / step_time
    print(f'Step {step} : time {step_time:.4f} seconds | tokens/s {tokens_per_second:,.0f}' )
#------------------------------------------------------
torch.save(
    {
    'step': step + 1,
    'model_state_dict': raw_model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'model_config': asdict(config),
    'tokenizer_chars': tokenizer.chars
    }, checkpoint_last_path
)

