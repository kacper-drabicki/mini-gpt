## Mini GPT

A small character-level GPT trained from scratch with PyTorch.

### Setup

```bash
uv sync
```

### Train

```bash
uv run python3 train.py
```

Training creates a timestamped experiment directory under `experiments/`.

### Generate text

```bash
uv run python3 generate.py --dir experiments/<timestamp>
```
