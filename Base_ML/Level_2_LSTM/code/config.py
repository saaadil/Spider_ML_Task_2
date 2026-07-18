import platform
from pathlib import Path
import torch
import numpy as np

SEED = 42  # lock rng
torch.manual_seed(SEED)  # cpu seed
torch.cuda.manual_seed_all(SEED)  # gpu seed
np.random.seed(SEED)  # numpy seed
torch.backends.cudnn.deterministic = True  # strict math
torch.backends.cudnn.benchmark = False  # no dynamic scaling

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # get hardware
print(f"\n[Setup] PyTorch: {torch.__version__}")
print(f"[Setup] Device: {device}")
if device.type == 'cuda':
    print(f"[Setup] GPU: {torch.cuda.get_device_name(0)}")
    print(f"[Setup] VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
else:
    print("[Setup] WARNING: No GPU found.")

SCRIPT_DIR = Path(__file__).resolve().parent  # find script
BASE = SCRIPT_DIR.parent  # parent dir
WDIR = BASE / "model_weights"  # weights folder
ODIR = BASE / "outputs"  # plots go here
DATA_DIR = BASE / "data"  # csv folder

for d in [WDIR, ODIR, DATA_DIR]:  # loop dirs
    d.mkdir(parents=True, exist_ok=True)  # make if missing

print(f"[Setup] Outputs -> {ODIR}")
print(f"[Setup] Weights -> {WDIR}")

N_WORKERS = 0 if platform.system() == 'Windows' else 4  # cpu threads

CONFIG = {  # master settings
    'batch_size': 32,
    'num_workers': N_WORKERS,
    'val_ratio': 0.15,
    'test_ratio': 0.15,
    'seq_len': 72,
    'horizon': 12,
    'd_model': 128,
    'n_heads': 4,
    'n_encoder_layers': 3,
    'd_ff': 512,
    'dropout': 0.1,
    'lstm_hidden': 128,
    'lstm_layers': 2,
    'lstm_dropout': 0.2,
    'num_epochs': 50,
    'lr': 1e-4,
    'lstm_lr': 1e-3,
    'weight_decay': 5e-4,
    'milestones': [25, 38, 46],  # lr drops
    'gamma': 0.1,
    'grad_clip': 1.0,
    'save_every': 25,
}

TEMP_BINS = [0, 1, 2, 3]  # temp boundaries
TEMP_LABELS = ['Very Cold(<0°C)', 'Cold(0-10°C)', 'Mild(10-20°C)', 'Warm(>20°C)']  # text labels