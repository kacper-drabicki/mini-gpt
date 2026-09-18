from config import *
from model import MiniGPT
from tokenizer import CharacterTokenizer
import torch

#------------------------------------------------------
device = 'cuda' if torch.cuda.is_available() else 'cpu'
max_tokens = 2000
prompt = 'WILLIAM:'
#------------------------------------------------------
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

tokenizer = CharacterTokenizer(text)

model = MiniGPT(n_blocks=n_blocks, n_heads=n_heads, embd_dim=embd_dim, head_dim=head_dim, vocab_size=tokenizer.vocab_size, block_size=block_size)
model.load_state_dict(torch.load(model_path, weights_only=True))
model.to(device)

context = torch.tensor(tokenizer.encode(prompt), dtype=torch.long, device=device).view(1,-1)
text = model.generate(context, max_new_tokens=max_tokens)
print(tokenizer.decode(text[0].tolist()))

