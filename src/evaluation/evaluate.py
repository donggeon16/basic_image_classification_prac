import torch
import torch.nn as nn
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from src.data.cifar100 import test_dataloader
from src.models.resnet import ResNet18
from src.models.mobilenetv2 import MobileNetv2
from src.utils.constants import DEVICE, BASE

def test_model(model, test_loader, criterion, save_dir, device = DEVICE):
    model = model.to(device)
    checkpoint = torch.load(BASE/"runs"/save_dir/"best.pt", map_location = device, weights_only = True)
    model.load_state_dict(checkpoint["model_state_dict"])

    model.eval()
    with torch.no_grad():
        total_loss = 0
        total_samples = 0
        total_correct = 0
        confusion_matrix = None
        

        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            predictions = outputs.argmax(dim=1)

            if confusion_matrix is None:
                num_classes = outputs.shape[1]
                confusion_matrix = torch.zeros((num_classes, num_classes), dtype=torch.int64)

            for pred, label in zip(predictions.cpu().tolist(), labels.cpu().tolist()):
                confusion_matrix[label, pred] += 1

            correct = (predictions == labels).sum().item()
            total_correct += correct

            loss = criterion(outputs, labels)
            total_loss += loss.item() * labels.size(0)
            total_samples += labels.size(0)
    
    average_loss = total_loss / total_samples
    accuracy = total_correct / total_samples
    assert confusion_matrix is not None
    class_correct = confusion_matrix.diag()
    class_total = confusion_matrix.sum(dim=1)
    class_accuracy = class_correct.float() / class_total.clamp(min=1)

    # visualization
    classes = test_loader.dataset.classes
    plt.xticks(range(len(classes)), classes, rotation=45, ha="right")
    plt.yticks(range(len(classes)), classes)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.imshow(confusion_matrix.numpy(), cmap = "Blues")
    plt.colorbar()
    plt.show()

    return average_loss, accuracy, class_accuracy


def main():
    avg_loss, acc, cl_acc = test_model(model = MobileNetv2(100), test_loader = test_dataloader(), criterion=nn.CrossEntropyLoss(), save_dir="cifar100_MobileNetV2_kd_200_cosine_randaug_mixup0.2_T4.0_dkd_a1.0_b8.0_warmup20")
    print(f"average_loss = {avg_loss:.4f}, accuracy = {acc * 100:.3f}%\nclass accuracy = {cl_acc}")




if __name__ == "__main__":
    main()

