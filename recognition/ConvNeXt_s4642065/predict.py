"""
@file    predict.py
@brief   Perform inference or evaluation with a trained MiniConvNeXt model
@author  Aaron
@date    2025-10-31
"""

import torch
import torch.nn.functional as F
import argparse
import time
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from modules import ConvNeXt
from dataset import ADNIDataLoader
from utils import ModelParameters


# ---------------------------------------------------------------------
def evaluate_model(model, dataloader, criterion, device):
    """
    Evaluate a trained model on a dataset (e.g. test set).

    Args:
        model: Trained PyTorch model.
        dataloader: DataLoader for the evaluation dataset.
        criterion: Loss function.
        device: torch.device ('cuda' or 'cpu').

    Returns:
        (accuracy, average_loss, confusion_matrix)
    """
    print("== Evaluating on dataset ==")
    model.eval()

    correct, total, running_loss = 0, 0, 0.0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            with torch.amp.autocast(device_type="cuda", enabled=torch.cuda.is_available()):
                outputs = model(images)
                loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = running_loss / len(dataloader)
    accuracy = 100 * correct / total
    conf_matrix = confusion_matrix(all_labels, all_preds)

    print(f"Accuracy: {accuracy:.2f}% | Avg Loss: {avg_loss:.4f}")
    print("Confusion Matrix:\n", conf_matrix)

    os.makedirs("outputs", exist_ok=True)
    plt.savefig("outputs/confusion_matrix.png", bbox_inches="tight")



    # Optional: visualize confusion matrix
    disp = ConfusionMatrixDisplay(conf_matrix)
    disp.plot(cmap="Blues", values_format='d')
    plt.title("Test Set Confusion Matrix")
    plt.savefig("outputs/confusion_matrix.png", bbox_inches="tight")
    plt.close()

    return accuracy, avg_loss, conf_matrix


# ---------------------------------------------------------------------
def predict_single_image(model, image_path, transform, device, class_names=None):
    """
    Run inference on a single image and print the predicted class.

    Args:
        model: Trained model.
        image_path: Path to the input image.
        transform: Transform to apply to the image.
        device: torch.device.
        class_names: Optional list of class names.
    """
    model.eval()

    image = Image.open(image_path).convert("RGB")
    img_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad(), torch.amp.autocast(device_type="cuda", enabled=torch.cuda.is_available()):
        outputs = model(img_tensor)
        probs = F.softmax(outputs, dim=1)
        pred_idx = torch.argmax(probs, dim=1).item()
        confidence = float(probs[0][pred_idx]) * 100

    if class_names:
        pred_class = class_names[pred_idx]
        print(f"Predicted: {pred_class} ({confidence:.2f}%)")
    else:
        print(f"Predicted class index: {pred_idx} ({confidence:.2f}%)")

    # Visualization
    plt.imshow(image)
    plt.title(f"Predicted: {pred_class if class_names else pred_idx}\nConfidence: {confidence:.2f}%")
    plt.axis("off")
    plt.show()


# ---------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MiniConvNeXt Prediction Script")

    parser.add_argument("model", type=str, help="Path to trained model .pth file")
    parser.add_argument("path", type=str, help="Path to image or dataset directory")
    parser.add_argument("-e", "--evaluation", action="store_true", help="Run full dataset evaluation mode")
    parser.add_argument("-b", "--batch_size", type=int, default=64, help="Batch size for evaluation")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize params and model
    params = ModelParameters()
    model = ConvNeXt(
        in_chans=1,
        num_classes=2,
        depths=(2, 2, 6, 2),
        dims=(48, 96, 192, 384),
        drop_path_rate=params.drop_path_rate
    ).to(device)

    # Load trained weights
    try:
        model.load_state_dict(torch.load(args.model, map_location=device))
        print(f" Loaded model from {args.model}")
    except Exception as e:
        print(f" Error loading model: {e}")
        exit(1)

    # ------------------------------------------------------------------
    if args.evaluation:
        print("Running dataset evaluation...")

        # Load test dataset
        data = ADNIDataLoader(batch_size=args.batch_size, data_dir=args.path, compute_norm=True)
        _, _, test_loader = data.load()  # reuse your load() split

        criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1).to(device)
        accuracy, avg_loss, conf = evaluate_model(model, test_loader, criterion, device)
        print(f"Final Test Accuracy: {accuracy:.2f}% | Avg Loss: {avg_loss:.4f}")
        exit(0)

    # ------------------------------------------------------------------
    else:
        print("Running single image inference...")
        # Prepare transform (same as test)
        transform = ADNIDataLoader(compute_norm=False)._transform_test()
        predict_single_image(model, args.path, transform, device)
