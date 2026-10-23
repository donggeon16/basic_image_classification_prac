from torchvision import transforms
from torch.utils.data import Subset
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from pathlib import Path
import numpy as np

BASE = Path(__file__).resolve().parent.parent.parent


def train_dataloader(batch_size = 64, num_workers = 0) -> DataLoader:
    train_idx = np.load(BASE/"data"/"splits"/"cifar10_train_idx.npy")
    npz_load = np.load(BASE/"data"/"stats"/"cifar10_stats.npz")
    x_train_mean = npz_load["mean"]
    x_train_std = npz_load["std"]

    train_cifar10 = CIFAR10(
            root=BASE/"data/raw",
            train=True,
            download=False,
            transform=transforms.Compose(
                [transforms.RandomCrop(32, padding = 4), 
                 transforms.RandomHorizontalFlip(), 
                 transforms.ToTensor(), 
                 transforms.Normalize(mean = x_train_mean, std = x_train_std)]
                 )
            )
    train_dataset = Subset(train_cifar10, train_idx)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers = num_workers
    )

    return train_loader

def val_dataloader(batch_size = 64, num_workers = 0) -> DataLoader:
    val_idx = np.load(BASE/"data"/"splits"/"cifar10_val_idx.npy")
    npz_load = np.load(BASE/"data"/"stats"/"cifar10_stats.npz")
    x_train_mean = npz_load["mean"]
    x_train_std = npz_load["std"]

    val_cifar10 = CIFAR10(
                root=BASE/"data/raw",
                train=True,
                download=False,
                transform=transforms.Compose(
                    [transforms.ToTensor(), 
                     transforms.Normalize(mean = x_train_mean, std = x_train_std)]
                     )
                )
    
    val_dataset = Subset(val_cifar10, val_idx)

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    return val_loader


def main():

    train_loader = train_dataloader()
    val_loader = val_dataloader()

    train_images, train_labels = next(iter(train_loader))
    val_images, val_labels = next(iter(val_loader))
    print(f"Train: {train_images.shape}, {train_images.dtype}, {train_labels.shape}, {train_labels.dtype}, {len(train_loader)}")
    print(f"Val: {val_images.shape}, {val_images.dtype}, {val_labels.shape}, {val_labels.dtype}, {len(val_loader)}")


if __name__ == "__main__":
    main()