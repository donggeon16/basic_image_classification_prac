import torch
from src.data.augmentation import mixup_batch
from src.utils.constants import DEVICE
from src.losses.kd_loss import distillation_loss, dkd_loss

def train_one_epoch_kd(student, teacher, train_loader, criterion, optimizer, term, temperature = 4.0, mixup_alpha = 0.0, device = DEVICE):
    student.train()
    teacher.eval()
    total_loss = 0
    total_samples = 0

    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        # mixup
        if mixup_alpha > 0:
            images, labels_a, labels_b, lam = mixup_batch(images, labels, alpha=mixup_alpha)

        student_logits = student(images)
        with torch.no_grad():
            teacher_logits = teacher(images)

        if mixup_alpha > 0:
            ce_loss = (lam * criterion(student_logits, labels_a) + (1 - lam) * criterion(student_logits, labels_b))
        else:
            ce_loss = criterion(student_logits, labels)

        if term["method"] == "kd":
            loss = distillation_loss(student_logits, teacher_logits, ce_loss, temperature=temperature, alpha=term["alpha"])

        elif term["method"] == "dkd":
            loss = dkd_loss(
                student_logits,
                teacher_logits,
                ce_loss=ce_loss,
                temperature=temperature,
                alpha=term["alpha"],
                beta=term["beta"],
                labels=labels_a if mixup_alpha > 0 else labels,
                labels_b=labels_b if mixup_alpha > 0 else None,
                lam=lam if mixup_alpha > 0 else 1.0,
                warmup_factor=term["warmup_factor"]
            )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * labels.size(0)
        total_samples += labels.size(0)

    return total_loss / total_samples

def validate_one_epoch(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0
    total_samples = 0
    total_correct = 0
    with torch.no_grad():
        for images, labels in val_loader:

            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)

            predictions = outputs.argmax(dim=1)
            correct = (predictions == labels).sum().item()
            total_correct += correct

            loss = criterion(outputs, labels)
            total_loss += loss.item() * labels.size(0)
            total_samples += labels.size(0)
    
    average_loss = total_loss / total_samples
    accuracy = total_correct / total_samples

    return average_loss, accuracy



def main():
    pass

if __name__ == "__main__":
    main()