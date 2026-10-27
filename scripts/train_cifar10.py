from src.training.trainer import train_one_epoch, validate_one_epoch
from src.models.resnet import ResNet18
from src.data.cifar10 import train_dataloader, val_dataloader
from src.utils.checkpoint import save_checkpoint, load_checkpoint
import torch
import torch.nn as nn
from src.utils.constants import BASE, DEVICE

def main():

    model = ResNet18(10).to(DEVICE)
    train_loader = train_dataloader()
    val_loader = val_dataloader()

    # --------------------------------
    # HYPERPARAMETERS
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr = 0.01)
    epochs = 20
    save_dir = "cifar10_ResNet18"
    file_name = "best.pt"
    # --------------------------------

    # train and valid
    start_epoch = 0
    best_val_acc = 0.0

    if file_name:
        start_epoch, _, _, best_val_acc = load_checkpoint(model, optimizer, save_dir, file_name, map_location=DEVICE)

    for epoch in range(start_epoch, epochs):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
        val_loss, accuracy = validate_one_epoch(model, val_loader, criterion, DEVICE)

        if accuracy > best_val_acc:
            best_val_acc = accuracy
            save_checkpoint(epoch = epoch, model=model, optimizer=optimizer, val_loss=val_loss, val_acc=accuracy, save_dir=save_dir, is_best=True)

        print(f"Epoch [{epoch+1}/{epochs}]")
        print(f"Train Loss: {train_loss}")
        print(f"Val Loss: {val_loss}")
        print(f"Val Accuracy: {accuracy*100:.3f}%")
        save_checkpoint(epoch = epoch, model=model, optimizer=optimizer, val_loss=val_loss, val_acc=accuracy, save_dir=save_dir)

if __name__ == "__main__":
    main()