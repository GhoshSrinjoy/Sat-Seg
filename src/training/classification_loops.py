"""Minimal classification loops for DataLoaders yielding (images, class_ids)."""

import torch


def run_epoch(model, data_loader, device, optimizer=None) -> dict[str, float]:
    """Train if optimizer is provided, otherwise evaluate; return sample means.

    Uses cross-entropy for one class ID per image. The caller owns data splits,
    scheduling, checkpoints and normalization. This is not a SAM fine-tuner.
    """
    training = optimizer is not None
    model.train(training)
    total_loss, total_correct, total_samples = 0.0, 0, 0
    with torch.set_grad_enabled(training):
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = torch.nn.functional.cross_entropy(logits, labels)
            if training:
                loss.backward()
                optimizer.step()
            count = labels.size(0)
            total_loss += loss.detach().item() * count
            total_correct += (logits.argmax(dim=1) == labels).sum().item()
            total_samples += count
    if not total_samples:
        raise ValueError("The data loader yielded no samples.")
    return {"loss": total_loss / total_samples, "accuracy": total_correct / total_samples}
