"""
src/plots.py
============
All figures for the paper. Each function saves a PNG into results/figures/
and also shows it inline.

Public API:
    plot_confusion(y_true, y_pred, class_names, title, fname)
    plot_roc(y_true, y_prob, title, fname)                    # binary (Task A)
    plot_loss_curve(history, title, fname)
    plot_retention_bar(cl_df, fname)
    plot_stability_plasticity(cl_df, fname)
    plot_privacy_tradeoff(priv_df, fname)
    plot_lambda_sweep(sweep_df, fname)
    plot_pairs_grouped(all_pairs_df, fname)
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc

import config


def _save(fig, fname):
    path = os.path.join(config.FIGURE_DIR, fname)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print("saved:", path)


# --------------------------------------------------------------------------- #
#  CONFUSION MATRIX
# --------------------------------------------------------------------------- #
def plot_confusion(y_true, y_pred, class_names, title="Confusion", fname="confusion.png"):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_title(title); ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    plt.tight_layout(); _save(fig, fname); plt.show()


# --------------------------------------------------------------------------- #
#  ROC  (binary — Task A)
# --------------------------------------------------------------------------- #
def plot_roc(y_true, y_prob, title="ROC", fname="roc.png"):
    # y_prob: probability of the positive class (column 1)
    p1 = y_prob[:, 1] if y_prob.ndim == 2 else y_prob
    fpr, tpr, _ = roc_curve(y_true, p1)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(fpr, tpr, label=f"AUC = {auc(fpr, tpr):.3f}")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title(title); ax.legend(loc="lower right"); ax.grid(alpha=0.3)
    plt.tight_layout(); _save(fig, fname); plt.show()


# --------------------------------------------------------------------------- #
#  LOSS / ACC CURVE
# --------------------------------------------------------------------------- #
def plot_loss_curve(history, title="Training curve", fname="loss_curve.png"):
    ep = range(1, len(history["train_loss"]) + 1)
    fig, ax1 = plt.subplots(figsize=(6, 4))
    ax1.plot(ep, history["train_loss"], "o-", label="train loss")
    if history.get("val_loss"):
        ax1.plot(ep, history["val_loss"], "o-", label="val loss")
    ax1.set_xlabel("epoch"); ax1.set_ylabel("loss"); ax1.legend(loc="upper right")
    if history.get("val_acc"):
        ax2 = ax1.twinx()
        ax2.plot(ep, history["val_acc"], "g--s", label="val acc")
        ax2.set_ylabel("val acc")
    ax1.set_title(title)
    plt.tight_layout(); _save(fig, fname); plt.show()


# --------------------------------------------------------------------------- #
#  RETENTION BAR  (continual comparison)
# --------------------------------------------------------------------------- #
def plot_retention_bar(cl_df, fname="retention_bar.png"):
    d = cl_df.sort_values("TaskA_retention_%", ascending=False)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(d["method"], d["TaskA_retention_%"], color="tab:red", alpha=0.8)
    ax.set_ylabel("Task A retention (%)")
    ax.set_title("Catastrophic forgetting (higher = better retention)")
    for i, v in enumerate(d["TaskA_retention_%"]):
        ax.text(i, v + 1, f"{v:.0f}", ha="center", fontsize=9)
    plt.tight_layout(); _save(fig, fname); plt.show()


# --------------------------------------------------------------------------- #
#  STABILITY vs PLASTICITY
# --------------------------------------------------------------------------- #
def plot_stability_plasticity(cl_df, fname="stability_plasticity.png"):
    d = cl_df.sort_values("TaskA_retention_%", ascending=False)
    x = d["method"].tolist()
    fig, ax1 = plt.subplots(figsize=(7, 4.5))
    ax1.bar(x, d["TaskA_retention_%"], color="tab:red", alpha=0.7,
            label="Task A retention %")
    ax1.set_ylabel("Task A retention (%)", color="tab:red")
    ax2 = ax1.twinx()
    ax2.plot(x, d["TaskB_acc"], "ko--", label="Task B acc")
    ax2.set_ylabel("Task B accuracy"); ax2.set_ylim(0, 1)
    ax1.set_title("Stability (Task A) vs Plasticity (Task B)")
    plt.tight_layout(); _save(fig, fname); plt.show()


# --------------------------------------------------------------------------- #
#  PRIVACY-UTILITY TRADE-OFF
# --------------------------------------------------------------------------- #
def plot_privacy_tradeoff(priv_df, fname="privacy_tradeoff.png"):
    x = [str(e) for e in priv_df["epsilon"]]
    fig, ax1 = plt.subplots(figsize=(7, 5))
    ax1.plot(x, priv_df["TaskA_acc"], "o-", color="tab:blue", label="Task A accuracy")
    ax1.set_xlabel("Privacy budget  ε"); ax1.set_ylabel("Task A accuracy", color="tab:blue")
    ax2 = ax1.twinx()
    ax2.plot(x, priv_df["MIA_AUC"], "s--", color="tab:red", label="MIA AUC")
    ax2.axhline(0.5, color="gray", ls=":", alpha=0.6)
    ax2.set_ylabel("MIA AUC (attack success)", color="tab:red")
    ax1.set_title("Privacy–Utility Trade-off (smaller ε = more privacy)")
    plt.tight_layout(); _save(fig, fname); plt.show()


# --------------------------------------------------------------------------- #
#  LAMBDA SWEEP  (ablation)
# --------------------------------------------------------------------------- #
def plot_lambda_sweep(sweep_df, fname="lambda_sweep.png"):
    x = sweep_df["lambda"].astype(str)
    fig, ax1 = plt.subplots(figsize=(7, 4.5))
    ax1.plot(x, sweep_df["TaskA_retention_%"], "ro-", label="Task A retention %")
    ax1.set_xlabel("EWC λ"); ax1.set_ylabel("Task A retention %", color="tab:red")
    ax2 = ax1.twinx()
    ax2.plot(x, sweep_df["TaskB_acc"], "bs--", label="Task B acc")
    ax2.set_ylabel("Task B accuracy", color="tab:blue")
    ax1.set_title("EWC λ ablation: retention vs plasticity")
    plt.tight_layout(); _save(fig, fname); plt.show()


# --------------------------------------------------------------------------- #
#  GROUPED BAR  (all teacher x student pairs)
# --------------------------------------------------------------------------- #
def plot_pairs_grouped(all_pairs_df, fname="all_pairs_retention.png"):
    """Grouped bar: retention per method, grouped by (teacher,student) pair."""
    df = all_pairs_df.copy()
    df["pair"] = df["teacher"] + "+" + df["student"]
    pivot = df.pivot_table(index="pair", columns="method",
                           values="TaskA_retention_%")
    fig, ax = plt.subplots(figsize=(10, 5))
    pivot.plot(kind="bar", ax=ax)
    ax.set_ylabel("Task A retention (%)")
    ax.set_title("Retention by method across teacher×student pairs")
    ax.legend(title="method", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout(); _save(fig, fname); plt.show()
