from torchvision import transforms
from torch.utils.data import Subset
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR100
from pathlib import Path
import numpy as np
from src.utils.constants import BASE

def train_dataloader(batch_size = 64, num_workers = 0) -> DataLoader:
    train_idx = np.load(BASE/"data"/"splits"/"cifar100_train_idx.npy")
    npz_load = np.load(BASE/"data"/"stats"/"cifar100_stats.npz")
    x_train_mean = npz_load["mean"]
    x_train_std = npz_load["std"]

    train_cifar100 = CIFAR100(
            root=BASE/"data/raw",
            train=True,
            download=False,
            transform=transforms.Compose([
                transforms.RandomCrop(32, padding = 4), 
                transforms.RandomHorizontalFlip(), 
                transforms.RandAugment(num_ops=2, magnitude=9),
                transforms.ToTensor(), 
                # transforms.RandomErasing(p=0.25),
                transforms.Normalize(mean = x_train_mean, std = x_train_std)
                ])
            )
    train_dataset = Subset(train_cifar100, train_idx)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers = num_workers
    )

    return train_loader

def val_dataloader(batch_size = 64, num_workers = 0) -> DataLoader:
    val_idx = np.load(BASE/"data"/"splits"/"cifar100_val_idx.npy")
    npz_load = np.load(BASE/"data"/"stats"/"cifar100_stats.npz")
    x_train_mean = npz_load["mean"]
    x_train_std = npz_load["std"]

    val_cifar100 = CIFAR100(
                root=BASE/"data/raw",
                train=True,
                download=False,
                transform=transforms.Compose(
                    [transforms.ToTensor(), 
                     transforms.Normalize(mean = x_train_mean, std = x_train_std)]
                     )
                )
    
    val_dataset = Subset(val_cifar100, val_idx)

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    return val_loader

def test_dataloader(batch_size = 64, num_workers = 0) -> DataLoader:
    npz_load = np.load(BASE/"data"/"stats"/"cifar100_stats.npz")
    x_train_mean = npz_load["mean"]
    x_train_std = npz_load["std"]

    test_cifar100 = CIFAR100(
            root=BASE/"data/raw",
            train=False,
            download=False,
            transform=transforms.Compose(
                [transforms.ToTensor(), 
                transforms.Normalize(mean = x_train_mean, std = x_train_std)]
                    )
            )

    test_loader = DataLoader(
        test_cifar100,
        batch_size=batch_size,
        shuffle=False,
        num_workers = num_workers
    )

    return test_loader


def main():

    train_loader = train_dataloader()
    val_loader = val_dataloader()

    train_images, train_labels = next(iter(train_loader))
    val_images, val_labels = next(iter(val_loader))
    print(f"Train: {train_images.shape}, {train_images.dtype}, {train_labels.shape}, {train_labels.dtype}, {len(train_loader)}")
    print(f"Val: {val_images.shape}, {val_images.dtype}, {val_labels.shape}, {val_labels.dtype}, {len(val_loader)}")


if __name__ == "__main__":
    main()