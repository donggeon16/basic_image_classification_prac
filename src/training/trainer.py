import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_one_epoch(model, train_loader, criterion, optimizer, device = DEVICE):
    model.train()
    total_loss = 0
    total_samples = 0

    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
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