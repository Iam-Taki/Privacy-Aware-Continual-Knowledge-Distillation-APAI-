"""
src/distillation.py
===================
Knowledge Distillation: transfer the teacher's knowledge (Task A) into the
lightweight multi-head student's Task-A head + shared backbone.

We use logit-based KD (Hinton et al.):
    L = alpha * CE(student, labels) + (1-alpha) * T^2 * KL(soft_teacher || soft_student)

Public API:
    kd_logit_loss(student_logits, teacher_logits, T)
    distill(teacher, student, task_A_data)   -> (student, history)
"""

import copy
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import config
from src import engine


# --------------------------------------------------------------------------- #
#  LOSSES
# --------------------------------------------------------------------------- #
def kd_logit_loss(student_logits, teacher_logits, T=None):
    """Temperature-scaled KL divergence between teacher and student outputs."""
    T = T or config.KD_TEMPERATURE
    p_teacher = F.softmax(teacher_logits / T, dim=1)
    logp_student = F.log_softmax(student_logits / T, dim=1)
    return F.kl_div(logp_student, p_teacher, reduction="batchmean") * (T * T)


# --------------------------------------------------------------------------- #
#  DISTILLATION TRAINING
# --------------------------------------------------------------------------- #
def distill(teacher, student, task_A_data, epochs=None, lr=None,
            T=None, alpha=None, verbose=True):
    """Distill the teacher into the student's Task-A head (+ shared backbone)."""
    device = config.get_device()
    epochs = epochs or config.DISTILL_EPOCHS
    T = T or config.KD_TEMPERATURE
    alpha = config.KD_ALPHA if alpha is None else alpha

    teacher = teacher.to(device).eval()
    student = student.to(device)

    # only the Task-A head + backbone are trained at this stage
    params = list(student.backbone.parameters()) + list(student.head_a.parameters())
    opt = engine.make_optimizer(params, lr)
    ce = nn.CrossEntropyLoss(weight=task_A_data["class_weights"])

    train_loader = task_A_data["train_loader"]
    val_loader = task_A_data["val_loader"]

    history = {"train_loss": [], "val_acc": []}
    best_acc, best_state = 0.0, copy.deepcopy(student.state_dict())

    for ep in range(epochs):
        student.train()
        run_loss, total = 0.0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            with torch.no_grad():
                t_logits = teacher(xb)
            s_logits = student(xb, task="a")
            loss = alpha * ce(s_logits, yb) + (1 - alpha) * kd_logit_loss(
                s_logits, t_logits, T)
            opt.zero_grad()
            loss.backward()
            opt.step()
            run_loss += loss.item() * xb.size(0)
            total += xb.size(0)

        val = engine.evaluate(student, val_loader, task="a")
        history["train_loss"].append(run_loss / total)
        history["val_acc"].append(val["acc"])
        if val["acc"] >= best_acc:
            best_acc = val["acc"]
            best_state = copy.deepcopy(student.state_dict())
        if verbose:
            print(f"  epoch {ep+1}/{epochs}  "
                  f"loss={run_loss/total:.3f}  val_acc(A)={val['acc']:.3f}")

    student.load_state_dict(best_state)
    return student, history
