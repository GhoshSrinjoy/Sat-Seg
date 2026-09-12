"""An untrained RGB classification baseline; no automatic weight downloads."""

from torch import nn
from torchvision.models import resnet18


def build_classifier(num_classes: int) -> nn.Module:
    """Build ResNet-18 with a task-specific head and random initial weights.

    Choose classes and labeled data before training. SAM 3 is a separate
    promptable segmentation model and does not train this classifier.
    """
    if num_classes < 2:
        raise ValueError("Classification requires at least two classes.")
    model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model
