from src.training.trainer import train_one_epoch, validate_one_epoch
from src.models.resnet import ResNet18
from src.data.cifar10 import train_dataloader, val_dataloader
import torch
import torch.nn as nn

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():

    # --------------------------------
    # HYPERPARAMETERS
    lr = 0.01
    epochs = 5
    # --------------------------------

    model = ResNet18().to(DEVICE)
    train_loader = train_dataloader()
    val_loader = val_dataloader()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr = lr)

    # train and valid
    for epoch in range(epochs):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
        val_loss, accuracy = validate_one_epoch(model, val_loader, criterion, DEVICE)
        print(f"Epoch [{epoch+1}/{epochs}]")
        print(f"Train Loss: {train_loss}")
        print(f"Val Loss: {val_loss}")
        print(f"Val Accuracy: {accuracy*100:.3f}%")

if __name__ == "__main__":
    main()