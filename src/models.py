import torch.nn as nn
from torchvision import models


def build_model(name, num_classes=2, pretrained=True):
    """Model factory for the three architectures in the proposal.

    TODO (your contribution): add the EA/EC edge layers and LTHE preprocessing
    from Bae, Cho and Jung (2025) once the baselines run cleanly.
    """
    name = name.lower()
    if name == "resnet50":
        w = models.ResNet50_Weights.DEFAULT if pretrained else None
        m = models.resnet50(weights=w)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
        return m
    if name == "densenet121":
        w = models.DenseNet121_Weights.DEFAULT if pretrained else None
        m = models.densenet121(weights=w)
        m.classifier = nn.Linear(m.classifier.in_features, num_classes)
        return m
    if name == "vit":
        w = models.ViT_B_16_Weights.DEFAULT if pretrained else None
        m = models.vit_b_16(weights=w)
        m.heads.head = nn.Linear(m.heads.head.in_features, num_classes)
        return m
    raise ValueError(f"Unknown model: {name}")
