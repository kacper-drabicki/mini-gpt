Experiment: baseline
Date: 18.09.2026
Changes: initial implementation

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

Final train loss: 1.455451250076294
Final val loss: 1.6481983661651611
