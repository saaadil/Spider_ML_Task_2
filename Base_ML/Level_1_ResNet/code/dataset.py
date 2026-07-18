import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset

from config import data_path, my_conf, workers, classes

cifar_mean = (0.4914, 0.4822, 0.4465)  # rgb mean
cifar_std = (0.2470, 0.2435, 0.2616)  # rgb std

def get_dataloaders(conf, data_dir, seed=42):  # make loaders
    
    tr_trans = transforms.Compose([
        transforms.RandomCrop(32, padding=4, padding_mode='reflect'),  # crop and pad
        transforms.RandomHorizontalFlip(p=0.5),  # flip random
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),  # color jitter
        transforms.ToTensor(),  # to tensor
        transforms.Normalize(cifar_mean, cifar_std),  # norm stats
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.15), ratio=(0.3, 3.3)),  # random erase
    ])

    test_trans = transforms.Compose([
        transforms.ToTensor(),  # to tensor
        transforms.Normalize(cifar_mean, cifar_std),  # norm stats
    ])

    full_tr = torchvision.datasets.CIFAR10(str(data_dir), train=True, download=True, transform=tr_trans)  # train set
    full_v = torchvision.datasets.CIFAR10(str(data_dir), train=True, download=False, transform=test_trans)  # val set
    test_set = torchvision.datasets.CIFAR10(str(data_dir), train=False, download=True, transform=test_trans)  # test set

    ntot = len(full_tr)  # total train
    nval = int(ntot * conf['val_ratio'])  # calc val size
    
    rng = torch.Generator().manual_seed(seed)  # seeded generator
    all_idx = torch.randperm(ntot, generator=rng).tolist()  # shuffle indices
    
    v_idx = all_idx[:nval]  # val indices
    t_idx = all_idx[nval:]  # train indices

    tr_set = Subset(full_tr, t_idx)  # make train subset
    val_set = Subset(full_v, v_idx)  # make val subset

    kw = dict(
        batch_size=conf['batch_size'],
        num_workers=conf['num_workers'],
        pin_memory=True,
        persistent_workers=(conf['num_workers'] > 0),
    )

    tr_loader = DataLoader(tr_set, shuffle=True, **kw)  # train loader
    v_loader = DataLoader(val_set, shuffle=False, **kw)  # val loader
    te_loader = DataLoader(test_set, shuffle=False, **kw)  # test loader

    return tr_loader, v_loader, te_loader  # return all