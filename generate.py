import argparse
import torch
import tiktoken
from pathlib import Path
from model import MiniGPT, GPTConfig

#------------------------------------------------------
device = 'cuda' if torch.cuda.is_available() else 'cpu'
max_new_tokens = 500
prompt = 'WILLIAM:'
#------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', required=True)
    args = parser.parse_args()

    experiment_dir = Path(args.dir)
    checkpoint = torch.load(experiment_dir / 'checkpoint_last.pt', map_location=device)

    tokenizer = tiktoken.get_encoding('gpt2')

    model = MiniGPT(GPTConfig(**checkpoint["model_config"]))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    context = torch.tensor(tokenizer.encode(prompt), dtype=torch.long, device=device).view(1,-1)
    text = model.generate(context, max_new_tokens=max_new_tokens)
    print(tokenizer.decode(text[0].tolist()))

if __name__ == '__main__':
    main()

