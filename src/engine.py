"""
src/engine.py
=============
The training / evaluation engine — task-agnostic and reused by every stage
(teacher training, distillation, continual learning, privacy).

Public API:
    train_supervised(model, train_loader, val_loader, weights, task, epochs)
        -> (model, history)            # generic supervised training

    evaluate(model, loader, task)
        -> dict(y_true, y_pred, y_prob, acc)

    full_report(model, loader, task, class_names)
        -> dict of metrics (acc, macro precision/recall/f1, auc, confusion)

    save_checkpoint(model, name)  /  load_checkpoint(model, name)
"""

import os
import copy
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score)

import config


# --------------------------------------------------------------------------- #
#  OPTIMIZER
# --------------------------------------------------------------------------- #
def make_optimizer(params, lr=None):
    lr = lr or config.LR
    if config.OPTIMIZER.lower() == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=config.MOMENTUM)
    return torch.optim.Adam(params, lr=lr)


# --------------------------------------------------------------------------- #
#  EVALUATION
# --------------------------------------------------------------------------- #
@torch.no_grad()
def evaluate(model, loader, task=None):
    """Run the model over a loader. task=None for single-head teacher,
    task='a'/'b' for the multi-head student."""
    device = config.get_device()
    model.eval()
    ys, preds, probs = [], [], []
    for xb, yb in loader:
        xb = xb.to(device)
        out = model(xb) if task is None else model(xb, task)
        probs.append(F.softmax(out, dim=1).cpu().numpy())
        preds.extend(out.argmax(1).cpu().numpy())
        ys.extend(yb.numpy())
    ys, preds = np.array(ys), np.array(preds)
    probs = np.concatenate(probs) if probs else np.empty((0,))
    acc = (preds == ys).mean() if len(ys) else 0.0
    return {"y_true": ys, "y_pred": preds, "y_prob": probs, "acc": acc}


def full_report(model, loader, task, class_names):
    """Accuracy + macro P/R/F1 + AUC (binary) + confusion matrix."""
    r = evaluate(model, loader, task)
    yt, yp, pr = r["y_true"], r["y_pred"], r["y_prob"]
    rep = classification_report(yt, yp, target_names=class_names,
                                output_dict=True, zero_division=0)
    out = {
        "acc": r["acc"],
        "precision": rep["macro avg"]["precision"],
        "recall": rep["macro avg"]["recall"],
        "f1": rep["macro avg"]["f1-score"],
        "confusion": confusion_matrix(yt, yp),
        "report_text": classification_report(yt, yp, target_names=class_names,
                                             digits=4, zero_division=0),
        "y_true": yt, "y_pred": yp, "y_prob": pr,
    }
    # ROC-AUC only well-defined for binary here (Task A)
    if pr.shape[1] == 2:
        out["auc"] = roc_auc_score(yt, pr[:, 1])
    else:
        # multi-class: one-vs-rest macro AUC
        try:
            out["auc"] = roc_auc_score(yt, pr, multi_class="ovr", average="macro")
        except Exception:
            out["auc"] = float("nan")
    return out


# --------------------------------------------------------------------------- #
#  TRAINING
# --------------------------------------------------------------------------- #
def train_supervised(model, train_loader, val_loader, weights,
                     task=None, epochs=None, lr=None, verbose=True):
    """Generic supervised training with validation; keeps best-val weights."""
    device = config.get_device()
    epochs = epochs or config.TEACHER_EPOCHS
    model = model.to(device)
    crit = nn.CrossEntropyLoss(weight=weights)
    opt = make_optimizer(model.parameters(), lr)

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_acc, best_state = 0.0, copy.deepcopy(model.state_dict())

    for ep in range(epochs):
        model.train()
        run_loss, correct, total = 0.0, 0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            out = model(xb) if task is None else model(xb, task)
            loss = crit(out, yb)
            loss.backward()
            opt.step()
            run_loss += loss.item() * xb.size(0)
            correct += (out.argmax(1) == yb).sum().item()
            total += xb.size(0)

        tr_loss, tr_acc = run_loss / total, correct / total
        val = evaluate(model, val_loader, task)
        # validation loss (for the curve)
        v_loss = _loss_on(model, val_loader, crit, task)

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(v_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(val["acc"])

        if val["acc"] >= best_acc:
            best_acc = val["acc"]
            best_state = copy.deepcopy(model.state_dict())
        if verbose:
            print(f"  epoch {ep+1}/{epochs}  "
                  f"train_loss={tr_loss:.3f}  val_acc={val['acc']:.3f}")

    model.load_state_dict(best_state)
    return model, history


@torch.no_grad()
def _loss_on(model, loader, crit, task):
    device = config.get_device()
    model.eval()
    s, n = 0.0, 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        out = model(xb) if task is None else model(xb, task)
        s += crit(out, yb).item() * xb.size(0)
        n += xb.size(0)
    return s / max(n, 1)


# --------------------------------------------------------------------------- #
#  CONTINUAL-LEARNING METRIC: retention
# --------------------------------------------------------------------------- #
def retention(acc_after, acc_before):
    """Task-A retention (%) = how much of the original accuracy survived."""
    if acc_before <= 0:
        return 0.0
    return 100.0 * acc_after / acc_before


# --------------------------------------------------------------------------- #
#  CHECKPOINTS
# --------------------------------------------------------------------------- #
def save_checkpoint(model, name):
    path = os.path.join(config.CHECKPOINT_DIR, name)
    if not path.endswith(".pth"):
        path += ".pth"
    torch.save(model.state_dict(), path)
    print("saved:", path)
    return path


def load_checkpoint(model, name):
    path = os.path.join(config.CHECKPOINT_DIR, name)
    if not path.endswith(".pth"):
        path += ".pth"
    model.load_state_dict(torch.load(path, map_location=config.get_device()))
    model.to(config.get_device())
    print("loaded:", path)
    return model


def checkpoint_exists(name):
    path = os.path.join(config.CHECKPOINT_DIR, name)
    if not path.endswith(".pth"):
        path += ".pth"
    return os.path.exists(path)
