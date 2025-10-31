import os
import csv
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR
#import matplotlib.pyplot as plt

from modules import ConvNeXt
from dataset import ADNIDataLoader
from utils import ModelParameters



def setup_training_env():
    """Initialize model parameters, device, model, optimizer, and scheduler."""

    params = ModelParameters()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using device: {device}")

    # Data
    data = ADNIDataLoader(batch_size=params.batch_size, data_dir=params.data_file_path)
    train_loader, val_loader, test_loader = data.load()

    # Model
    model = ConvNeXt(
        in_chans=1,
        num_classes=2,
        depths=(2, 2, 6, 2),
        dims=(48, 96, 192, 384),
        drop_path_rate=params.drop_path_rate
    ).to(device)

    # Loss, optimizer, scheduler
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(),
                            lr=params.learning_rate,
                            weight_decay=params.weight_decay)

    scheduler = OneCycleLR(
        optimizer,
        max_lr=params.learning_rate * 2,
        steps_per_epoch=len(train_loader),
        epochs=params.epochs,
        pct_start=0.05
    )

    return model, train_loader, val_loader, criterion, optimizer, scheduler, device, params


def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, device, params):
    """Main training loop (no test step here — that’s handled in predict.py)."""

    scaler = torch.amp.GradScaler(enabled=torch.cuda.is_available())

    best_acc = 0.0
    start_time = time.time()
    print("\n=== Training started ===")

    for epoch in range(params.epochs):
        model.train()
        running_loss = 0.0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            # Mixed precision
            with torch.amp.autocast(device_type="cuda", enabled=torch.cuda.is_available()):
                outputs = model(images)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            running_loss += loss.item()

        # Average loss over epoch
        avg_train_loss = running_loss / len(train_loader)
        params.training_losses.append(avg_train_loss)

        # Validation step
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        params.estimated_test_losses.append(val_loss)
        params.estimated_test_accuracy.append(val_acc)

        if val_acc > best_acc:
            best_acc = val_acc
            save_model(model, params, best=True)

        print(f"Epoch [{epoch+1:03d}/{params.epochs}] "
              f"| Train Loss: {avg_train_loss:.4f} "
              f"| Val Loss: {val_loss:.4f} "
              f"| Val Acc: {val_acc:.2f}% "
              f"| LR: {optimizer.param_groups[0]['lr']:.6f}")

    total_time = time.time() - start_time
    print(f"\nTraining completed in {total_time/60:.2f} min. Best Val Acc = {best_acc:.2f}%")

    # Final save
    save_model(model, params, best=False)
    save_metrics(params)
    #plot_training_curves(params)

    
def evaluate(model, dataloader, criterion, device):
    """Validation loop."""
    model.eval()
    running_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            with torch.amp.autocast(device_type="cuda", enabled=torch.cuda.is_available()):
                outputs = model(images)
                loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    avg_loss = running_loss / len(dataloader)
    acc = 100 * correct / total
    return avg_loss, acc
    
    
def save_model(model, params, best=False):
    """Save model weights to .pth file."""
    os.makedirs("outputs", exist_ok=True)
    tag = "best" if best else "final"
    path = f"outputs/ConvNeXt_{tag}.pth"
    torch.save(model.state_dict(), path)
    print(f" Saved {tag} model to {path}")
    
    
def save_metrics(params):
    """Save training and validation metrics to CSV."""
    os.makedirs("outputs", exist_ok=True)

    with open("outputs/losses.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["train_loss"] + params.training_losses)
        writer.writerow(["val_loss"] + params.estimated_test_losses)

    with open("outputs/accuracy.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["val_accuracy"] + params.estimated_test_accuracy)

    print(" Saved metrics to outputs/")    
    
    
    
def plot_training_curves(params):
    """Plot loss and accuracy curves."""
    epochs = range(1, len(params.training_losses) + 1)
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, params.training_losses, label="Train Loss")
    plt.plot(epochs, params.estimated_test_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Loss Curves")

    plt.subplot(1, 2, 2)
    plt.plot(epochs, params.estimated_test_accuracy, label="Val Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.title("Validation Accuracy")

    os.makedirs("outputs", exist_ok=True)
    plt.savefig("outputs/training_curves.png", bbox_inches="tight")
    plt.close()
    print(" Saved training curves to outputs/training_curves.png")

    
    
if __name__ == "__main__":
    model, train_loader, val_loader, criterion, optimizer, scheduler, device, params = setup_training_env()
    train_model(model,  train_loader, val_loader, criterion, optimizer, scheduler, device, params)  