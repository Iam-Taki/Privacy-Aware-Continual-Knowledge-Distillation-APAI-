"""
src/models.py
=============
Model definitions for the study.

Teachers (single-head, trained on Task A):
    - resnet50      (CNN)
    - vit_b_16      (Transformer)

Students (multi-head: shared backbone + one head per task):
    - mobilenetv3   (MobileNetV3-Large)
    - efficientnet_b0

A multi-head student is what makes continual learning possible: the backbone
is shared across tasks (so it can forget), while each task keeps its own head.

Public API:
    build_teacher(name, num_classes)         -> nn.Module
    build_student(name, n_a, n_b)            -> MultiHeadStudent
    count_params(model)                      -> float (millions)
"""

import torch
import torch.nn as nn
from torchvision import models

import config


# --------------------------------------------------------------------------- #
#  TEACHERS  (single head, for Task A)
# --------------------------------------------------------------------------- #
def build_teacher(name=None, num_classes=None):
    name = (name or config.MAIN_TEACHER).lower()
    num_classes = num_classes or config.NUM_CLASSES_A

    if name == "resnet50":
        m = models.resnet50(weights="IMAGENET1K_V2")
        m.fc = nn.Linear(m.fc.in_features, num_classes)

    elif name == "vit_b_16":
        m = models.vit_b_16(weights="IMAGENET1K_V1")
        # ViT classification head lives in heads.head
        in_f = m.heads.head.in_features
        m.heads.head = nn.Linear(in_f, num_classes)

    else:
        raise ValueError(f"Unknown teacher: {name}")
    return m


# --------------------------------------------------------------------------- #
#  STUDENT  (multi-head: shared backbone + head_a + head_b)
# --------------------------------------------------------------------------- #
class MultiHeadStudent(nn.Module):
    """Shared backbone + a separate classification head per task.

    forward(x, task="a") routes through the Task-A head;
    forward(x, task="b") routes through the Task-B head.
    """

    def __init__(self, backbone_name="mobilenetv3", n_a=2, n_b=4):
        super().__init__()
        self.backbone_name = backbone_name

        if backbone_name == "mobilenetv3":
            base = models.mobilenet_v3_large(weights="IMAGENET1K_V1")
            self.backbone = base.features
            self.pool = base.avgpool
            feat_dim = base.classifier[0].in_features          # 960

        elif backbone_name == "efficientnet_b0":
            base = models.efficientnet_b0(weights="IMAGENET1K_V1")
            self.backbone = base.features
            self.pool = base.avgpool
            feat_dim = base.classifier[1].in_features          # 1280

        else:
            raise ValueError(f"Unknown student backbone: {backbone_name}")

        self.feat_dim = feat_dim
        self.head_a = self._make_head(feat_dim, n_a)
        self.head_b = self._make_head(feat_dim, n_b)

    @staticmethod
    def _make_head(in_dim, n_classes):
        return nn.Sequential(
            nn.Linear(in_dim, 512), nn.Hardswish(),
            nn.Dropout(0.2), nn.Linear(512, n_classes))

    def features(self, x):
        x = self.backbone(x)
        x = self.pool(x)
        return torch.flatten(x, 1)

    def forward(self, x, task="a"):
        f = self.features(x)
        return self.head_a(f) if task == "a" else self.head_b(f)


def build_student(name=None, n_a=None, n_b=None):
    name = (name or config.MAIN_STUDENT).lower()
    n_a = n_a or config.NUM_CLASSES_A
    n_b = n_b or config.NUM_CLASSES_B
    return MultiHeadStudent(backbone_name=name, n_a=n_a, n_b=n_b)


# --------------------------------------------------------------------------- #
#  UTIL
# --------------------------------------------------------------------------- #
def count_params(model):
    """Number of parameters in millions."""
    return sum(p.numel() for p in model.parameters()) / 1e6


def describe_all():
    """Print param counts for every teacher / student — sanity check."""
    print("Teachers:")
    for t in config.TEACHERS:
        print(f"  {t:16s} {count_params(build_teacher(t)):.2f} M")
    print("Students (multi-head):")
    for s in config.STUDENTS:
        print(f"  {s:16s} {count_params(build_student(s)):.2f} M")
