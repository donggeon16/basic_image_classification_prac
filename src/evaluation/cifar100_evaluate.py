"""CIFAR-100 test evaluation and per-run reports.

Run from the project root:
    python -m src.evaluation.evaluate

Copy this file to src/evaluation/evaluate.py and edit EXPERIMENTS paths
according to your existing runs/ directories. The test DataLoader must use
shuffle=False and deterministic preprocessing for cross-model comparisons.
"""

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import SequentialSampler

from src.data.cifar100 import test_dataloader
from src.models.resnet import ResNet18
from src.models.mobilenetv2 import MobileNetv2
from src.utils.constants import BASE, DEVICE


# The strings must match the actual directory names under runs/.
EXPERIMENTS = {
    "teacher": (
        ResNet18,
        "cifar100_ResNet18_teacher_200_cosine_randaug_mixup0.2",
    ),
    "baseline": (
        MobileNetv2,
        "cifar100_MobileNetV2_student_200_cosine_randaug_mixup0.2",
    ),
    "kd": (
        MobileNetv2,
        "cifar100_MobileNetV2_kd_200_cosine_randaug_mixup0.2_T4.0_kd_a0.5",
    ),
    "dkd": (
        MobileNetv2,
        "cifar100_MobileNetV2_kd_200_cosine_randaug_mixup0.2_T4.0_dkd_a1.0_b8.0_warmup20",
    ),
}

REPORTS_ROOT = BASE / "reports" / "raw" / "cifar100"
STATS_FILE = BASE / "data" / "stats" / "cifar100_stats.npz"


def write_csv(path: Path, headers, rows):
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.writer(file)
        writer.writerow(headers)
        writer.writerows(rows)


def unnormalize(image: torch.Tensor, mean: torch.Tensor, std: torch.Tensor):
    """[3, H, W] CPU tensor -> displayable [H, W, 3] array."""
    return (image * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()


def save_gallery(samples, classes, path, mean, std, title):
    """samples: (confidence, index, image_cpu, true_id, predicted_id)."""
    if not samples:
        return
    n = len(samples)
    cols = 5
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3.4), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")

    for ax, (confidence, index, image, true_id, pred_id) in zip(axes.flat, samples):
        ax.imshow(unnormalize(image, mean, std))
        ax.set_title(
            f"#{index}  conf={confidence:.2f}\n"
            f"GT: {classes[true_id]}\nPred: {classes[pred_id]}",
            fontsize=8,
        )
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


@torch.inference_mode()
def test_model(model, test_loader, criterion, save_dir, report_name=None, device=DEVICE):
    """Evaluate one run, save reports and return aligned test predictions."""
    if not isinstance(test_loader.sampler, SequentialSampler):
        raise ValueError("Test DataLoader must use shuffle=False for image-index comparison.")

    checkpoint_path = BASE / "runs" / save_dir / "best.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device).eval()

    classes = test_loader.dataset.classes  # torchvision CIFAR100(train=False)
    num_classes = len(classes)
    report_dir = REPORTS_ROOT / (report_name or save_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    stats = np.load(STATS_FILE)
    mean = torch.tensor(stats["mean"], dtype=torch.float32).view(3, 1, 1)
    std = torch.tensor(stats["std"], dtype=torch.float32).view(3, 1, 1)

    cm = torch.zeros((num_classes, num_classes), dtype=torch.int64)
    all_true, all_pred = [], []
    incorrect_rows = []
    high_confidence_errors = []
    total_samples = 0
    total_correct = 0
    total_loss = 0.0

    for images, labels in test_loader:
        outputs = model(images.to(device))
        target = labels.cpu()
        pred = outputs.argmax(dim=1).cpu()
        confidence = outputs.softmax(dim=1).amax(dim=1).cpu()

        batch_size = target.numel()
        total_loss += criterion(outputs, labels.to(device)).item() * batch_size
        total_correct += (pred == target).sum().item()
        cm += torch.bincount(
            target * num_classes + pred,
            minlength=num_classes * num_classes,
        ).reshape(num_classes, num_classes)

        all_true.extend(target.tolist())
        all_pred.extend(pred.tolist())

        wrong_indices = (pred != target).nonzero(as_tuple=True)[0].tolist()
        for i in wrong_indices:
            index = total_samples + i  # valid only for sequential test ordering
            conf = float(confidence[i])
            true_id, pred_id = int(target[i]), int(pred[i])
            incorrect_rows.append((index, classes[true_id], classes[pred_id], conf))
            high_confidence_errors.append(
                (conf, index, images[i].cpu().clone(), true_id, pred_id)
            )
            # Keep the 20 most confident wrong predictions across the whole test set.
            high_confidence_errors.sort(key=lambda item: item[0], reverse=True)
            del high_confidence_errors[20:]

        total_samples += batch_size

    if total_samples == 0:
        raise ValueError("Empty test set")
    accuracy = total_correct / total_samples
    avg_loss = total_loss / total_samples
    class_total = cm.sum(dim=1)
    class_correct = cm.diag()
    class_accuracy = class_correct.float() / class_total.clamp_min(1)

    write_csv(
        report_dir / "class_accuracy.csv",
        ("class_id", "class_name", "correct", "total", "accuracy_percent"),
        [
            (i, name, int(class_correct[i]), int(class_total[i]), float(class_accuracy[i]) * 100)
            for i, name in enumerate(classes)
        ],
    )
    write_csv(
        report_dir / "confusion_matrix.csv",
        ("true\\pred", *classes),
        [(classes[i], *cm[i].tolist()) for i in range(num_classes)],
    )
    write_csv(
        report_dir / "incorrect_predictions.csv",
        ("test_index", "true_class", "predicted_class", "predicted_confidence"),
        incorrect_rows,
    )

    # Exclude correct predictions and rank directed class-confusion pairs by count.
    off_diagonal = cm.clone()
    off_diagonal.fill_diagonal_(0)
    values, indices = off_diagonal.flatten().topk(min(10, num_classes ** 2))
    top_confusions = []
    for count, index in zip(values.tolist(), indices.tolist()):
        if count == 0:
            break
        actual_id, predicted_id = divmod(index, num_classes)
        top_confusions.append((
            classes[actual_id], classes[predicted_id], count,
            count / max(int(class_total[actual_id]), 1) * 100,
        ))
    write_csv(
        report_dir / "top_confusions.csv",
        ("true_class", "predicted_class", "count", "percent_of_true_class"),
        top_confusions,
    )

    if top_confusions:
        labels = [f"{r[0]} → {r[1]}" for r in top_confusions][::-1]
        counts = [r[2] for r in top_confusions][::-1]
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.barh(labels, counts)
        ax.set_xlabel("Number of misclassified images")
        ax.set_title("Top 10 class confusions")
        fig.tight_layout()
        fig.savefig(report_dir / "top_confusions.png", dpi=160)
        plt.close(fig)

    save_gallery(
        high_confidence_errors, classes, report_dir / "misclassified_examples.png",
        mean, std, "20 most confident misclassifications",
    )

    metrics = {
        "run_directory": save_dir,
        "checkpoint": str(checkpoint_path),
        "test_images": total_samples,
        "test_loss": avg_loss,
        "test_accuracy": accuracy,
        "test_accuracy_percent": 100 * accuracy,
        "best_val_accuracy_percent": 100 * float(checkpoint["val_acc"]),
        "parameters": sum(p.numel() for p in model.parameters()),
    }
    (report_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"{report_name or save_dir}: test_acc={accuracy * 100:.2f}% -> {report_dir}")
    return {"metrics": metrics, "target": all_true, "prediction": all_pred}


def save_comparison(results, test_loader):
    """Compare models on identical test samples, not on unmatched batches."""
    required = ("baseline", "kd", "dkd")
    if any(name not in results for name in required):
        return
    true = results["baseline"]["target"]
    for name in required:
        if results[name]["target"] != true:
            raise ValueError(f"Ground-truth order differs for {name}")
    pred_b = results["baseline"]["prediction"]
    pred_k = results["kd"]["prediction"]
    pred_d = results["dkd"]["prediction"]
    classes = test_loader.dataset.classes
    out_dir = REPORTS_ROOT / "model_comparison"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Image-level changes in correctness, not simply aggregate accuracy.
    flags = {
        "baseline_wrong_kd_correct": [],
        "baseline_wrong_dkd_correct": [],
        "kd_wrong_dkd_correct": [],
        "baseline_correct_dkd_wrong": [],
    }
    comparison_rows = []
    for i, (gt, b, k, d) in enumerate(zip(true, pred_b, pred_k, pred_d)):
        b_ok, k_ok, d_ok = b == gt, k == gt, d == gt
        if not b_ok and k_ok:
            flags["baseline_wrong_kd_correct"].append(i)
        if not b_ok and d_ok:
            flags["baseline_wrong_dkd_correct"].append(i)
        if not k_ok and d_ok:
            flags["kd_wrong_dkd_correct"].append(i)
        if b_ok and not d_ok:
            flags["baseline_correct_dkd_wrong"].append(i)
        comparison_rows.append((
            i, classes[gt], classes[b], classes[k], classes[d],
            int(b_ok), int(k_ok), int(d_ok),
        ))
    write_csv(
        out_dir / "predictions_comparison.csv",
        ("test_index", "true_class", "baseline_pred", "kd_pred", "dkd_pred",
         "baseline_correct", "kd_correct", "dkd_correct"),
        comparison_rows,
    )
    (out_dir / "change_counts.json").write_text(
        json.dumps({key: len(value) for key, value in flags.items()}, indent=2),
        encoding="utf-8",
    )

    stats = np.load(STATS_FILE)
    mean = torch.tensor(stats["mean"], dtype=torch.float32).view(3, 1, 1)
    std = torch.tensor(stats["std"], dtype=torch.float32).view(3, 1, 1)

    # Gallery of actual examples for each change category (not shuffled).
    for category, indices in flags.items():
        if not indices:
            continue
        chosen = indices[:15]
        fig, axes = plt.subplots(3, 5, figsize=(16, 10), squeeze=False)
        for ax in axes.flat:
            ax.axis("off")
        for ax, i in zip(axes.flat, chosen):
            image, gt = test_loader.dataset[i]
            ax.imshow(unnormalize(image.cpu(), mean, std))
            ax.set_title(
                f"#{i} GT: {classes[gt]}\n"
                f"Base: {classes[pred_b[i]]}\n"
                f"KD: {classes[pred_k[i]]} / DKD: {classes[pred_d[i]]}",
                fontsize=8,
            )
        fig.suptitle(category.replace("_", " "))
        fig.tight_layout()
        fig.savefig(out_dir / f"{category}.png", dpi=160, bbox_inches="tight")
        plt.close(fig)
    print(f"Cross-model comparison -> {out_dir}")


def main():
    loader = test_dataloader()  # Must have shuffle=False and no random transforms.
    criterion = nn.CrossEntropyLoss()
    results = {}
    for name, (model_class, save_dir) in EXPERIMENTS.items():
        model = model_class(100)
        results[name] = test_model(
            model, loader, criterion, save_dir, report_name=name, device=DEVICE
        )
    summary = []
    for name, result in results.items():
        metrics = result["metrics"]
        summary.append((
            name,
            metrics["best_val_accuracy_percent"],
            metrics["test_accuracy_percent"],
            metrics["parameters"],
            metrics["run_directory"],
        ))
    REPORTS_ROOT.mkdir(parents=True, exist_ok=True)
    write_csv(
        REPORTS_ROOT / "summary.csv",
        ("model", "best_val_accuracy_percent", "test_accuracy_percent",
         "parameters", "run_directory"),
        summary,
    )
    save_comparison(results, loader)


if __name__ == "__main__":
    main()
