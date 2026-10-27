from src.training.kd_trainer import train_one_epoch_kd, validate_one_epoch
from src.models.mobilenetv2 import MobileNetv2
from src.models.resnet import ResNet18
from src.data.cifar100 import train_dataloader, val_dataloader
from src.utils.checkpoint import save_checkpoint, load_checkpoint, load_teacher_checkpoint
from src.utils.seed import set_seed
from src.utils.training_history import TrainingHistory
import torch
import torch.nn as nn
from src.utils.constants import BASE, DEVICE

def main():
    history = TrainingHistory()
    set_seed(42)
    teacher = ResNet18(100).to(DEVICE)
    set_seed(42)
    student = MobileNetv2(100).to(DEVICE)
    train_loader = train_dataloader()
    val_loader = val_dataloader()

    # --------------------------------
    # HYPERPARAMETERS
    optimizers = {
        "SGD": torch.optim.SGD(student.parameters(), lr = 0.01, momentum=0.9, weight_decay=5e-4),
        "AdamW": torch.optim.AdamW(student.parameters(), lr=1e-3, weight_decay=0.01),

        }
    criterion = nn.CrossEntropyLoss()
    optimizer = optimizers["SGD"]
    epochs = 200
    mixed_alpha = 0.2
    temperature = 4.0
    alpha = 0.5
    schedulers = {
        "None": None,
        "Cosine": torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5),

        }
    scheduler = schedulers["Cosine"]
    save_dir = f"cifar100_MobileNetV2_kd_{epochs}_cosine_randaug_mixup{mixed_alpha}_T{temperature}_a{alpha}"
    teacher_dir = f"cifar100_ResNet18_teacher_{epochs}_cosine_randaug_mixup0.2"
    load_file_name = ""
    # --------------------------------

    # train and valid
    start_epoch = 0
    best_val_acc = 0.0

    load_teacher_checkpoint(teacher, teacher_dir)
    teacher.requires_grad_(False)

    if load_file_name:
        start_epoch, _, _, best_val_acc = load_checkpoint(student, optimizer, save_dir, load_file_name, map_location=DEVICE, scheduler=scheduler)
        print(f"Best Val Accuracy: {best_val_acc*100:.3f}%")

    for epoch in range(start_epoch, epochs):
        train_loss = train_one_epoch_kd(student, teacher, train_loader, criterion, optimizer, temperature=temperature, alpha=alpha, mixup_alpha=mixed_alpha, device=DEVICE)
        val_loss, accuracy = validate_one_epoch(student, val_loader, criterion, DEVICE)
        if scheduler is not None:
            scheduler.step()

        if accuracy > best_val_acc:
            best_val_acc = accuracy
            save_checkpoint(epoch = epoch, model=student, optimizer=optimizer, val_loss=val_loss, val_acc=accuracy, save_dir=save_dir, is_best=True, scheduler=scheduler)

        print(f"Epoch [{epoch+1}/{epochs}]")
        print(f"Train Loss: {train_loss}")
        print(f"Val Loss: {val_loss}")
        print(f"Val Accuracy: {accuracy*100:.3f}%")
        save_checkpoint(epoch = epoch, model=student, optimizer=optimizer, val_loss=val_loss, val_acc=accuracy, save_dir=save_dir, scheduler=scheduler)

        history.update(epoch + 1, train_loss, val_loss, accuracy)
        history.save_plot(save_dir)

if __name__ == "__main__":
    main()