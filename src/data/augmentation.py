import torch
import numpy as np


def mixup_batch(images, labels, alpha=0.2):
    lam = np.random.beta(alpha, alpha)

    indices = torch.randperm(
        images.size(0),
        device=images.device
    )

    mixed_images = lam * images + (1 - lam) * images[indices]

    labels_a = labels
    labels_b = labels[indices]

    return mixed_images, labels_a, labels_b, lam