from src.training.trainer import train_one_epoch, validate_one_epoch
from src.models.resnet import ResNet18
from src.data.cifar100 import train_dataloader, val_dataloader
from src.utils.checkpoint import save_checkpoint, load_checkpoint
from src.utils.seed import set_seed
from src.utils.training_history import TrainingHistory
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from pathlib import Path

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    set_seed(42)
    history = TrainingHistory()
    model = ResNet18(100).to(DEVICE)
    train_loader = train_dataloader()
    val_loader = val_dataloader()

    # --------------------------------
    # HYPERPARAMETERS
    optimizers = {
        "SGD": torch.optim.SGD(model.parameters(), lr = 0.01, momentum=0.9, weight_decay=5e-4),
        "AdamW": torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01),

        }
    criterion = nn.CrossEntropyLoss()
    optimizer = optimizers["SGD"]
    epochs = 200
    mixed_alpha = 0.2
    schedulers = {
        "None": None,
        "Cosine": torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5),

        }
    scheduler = schedulers["Cosine"]
    save_dir = f"cifar100_ResNet18_teacher_{epochs}_cosine_randaug_mixup0.2"
    load_file_name = "best.pt"
    # --------------------------------

    # train and valid
    start_epoch = 0
    best_val_acc = 0.0

    if load_file_name:
        start_epoch, _, _, best_val_acc = load_checkpoint(model, optimizer, save_dir, load_file_name, map_location=DEVICE, scheduler=scheduler)
        print(f"Best Val Accuracy: {best_val_acc:.3f}%")

    for epoch in range(start_epoch, epochs):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, mixup_alpha=mixed_alpha, device=DEVICE)
        val_loss, accuracy = validate_one_epoch(model, val_loader, criterion, DEVICE)
        if scheduler is not None:
            scheduler.step()

        if accuracy > best_val_acc:
            best_val_acc = accuracy
            save_checkpoint(epoch = epoch, model=model, optimizer=optimizer, val_loss=val_loss, val_acc=accuracy, save_dir=save_dir, is_best=True, scheduler=scheduler)

        print(f"Epoch [{epoch+1}/{epochs}]")
        print(f"Train Loss: {train_loss}")
        print(f"Val Loss: {val_loss}")
        print(f"Val Accuracy: {accuracy*100:.3f}%")
        save_checkpoint(epoch = epoch, model=model, optimizer=optimizer, val_loss=val_loss, val_acc=accuracy, save_dir=save_dir, scheduler=scheduler)

        history.update(epoch + 1, train_loss, val_loss, accuracy)
        history.save_plot(save_dir)

if __name__ == "__main__":
    main()