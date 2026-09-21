## Mini GPT

A small character-level GPT trained from scratch with PyTorch.

### Setup

```bash
uv sync
source .venv/bin/activate
```

### Train

```bash
python train.py
```

Training creates a timestamped experiment directory under `experiments/`.

### Generate text

```bash
python generate.py --dir experiments/<timestamp>
```
