import platform
from pathlib import Path
import torch
import numpy as np

seed_val = 42  # lock rng
torch.manual_seed(seed_val)  # cpu seed
torch.cuda.manual_seed_all(seed_val)  # gpu seed
np.random.seed(seed_val)  # numpy seed
torch.backends.cudnn.deterministic = True  # strict math
torch.backends.cudnn.benchmark = False  # no dynamic scaling

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # get device
print(f"\n[Setup] PyTorch: {torch.__version__}")
print(f"[Setup] Device: {device}")
if device.type == 'cuda':
    print(f"[Setup] GPU: {torch.cuda.get_device_name(0)}")
    print(f"[Setup] VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
else:
    print("[Setup] WARNING: No GPU found.")

scriptDir = Path(__file__).resolve().parent  # find script
base_dir = scriptDir.parent  # parent dir
w_dir = base_dir / "model_weights"  # weights folder
OutDir = base_dir / "outputs"  # plots go here
data_path = base_dir / "data"  # csv folder

for d in [w_dir, OutDir, data_path]:  # loop dirs
    d.mkdir(parents=True, exist_ok=True)  # make if missing

workers = 0 if platform.system() == 'Windows' else 4  # cpu threads

my_conf = {  # master config
    'batch_size': 128,
    'num_workers': workers,
    'val_ratio': 0.1,
    'stage_channels': [64, 128, 256],
    'blocks_per_stage': 2,
    'num_classes': 10,
    'num_epochs': 100,
    'lr': 0.001,
    'weight_decay': 5e-4,
    'milestones': [50, 75, 90],  # lr drops
    'gamma': 0.1,
    'label_smooth': 0.1,
    'save_every': 25,
}

temp_bins = [0, 1, 2, 3]  # temp boundaries
classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']  # cifar classes