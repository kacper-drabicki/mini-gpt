import time
import argparse
import torch
from pathlib import Path
from model import MiniGPT, GPTConfig
from tokenizer import CharacterTokenizer

#------------------------------------------------------
device = 'cuda' if torch.cuda.is_available() else 'cpu'
max_new_tokens = 1000
prompt = 'WILLIAM:\n'
#------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', required=True)
    args = parser.parse_args()

    experiment_dir = Path(args.dir)
    checkpoint = torch.load(experiment_dir / 'checkpoint_last.pt', map_location=device)

    tokenizer = CharacterTokenizer(checkpoint['tokenizer_chars'])

    model = MiniGPT(GPTConfig(**checkpoint["model_config"]))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    context = torch.tensor(tokenizer.encode(prompt), dtype=torch.long, device=device).view(1,-1)

    start = time.perf_counter()
    text = model.generate(context, max_new_tokens=max_new_tokens)
    generation_time = time.perf_counter() - start

    print(tokenizer.decode(text[0].tolist()))
    print(f'Generation time: {generation_time:.4f} seconds')

if __name__ == '__main__':
    main()

