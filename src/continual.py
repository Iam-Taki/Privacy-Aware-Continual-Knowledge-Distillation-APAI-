"""
src/continual.py
================
Stage 3 — Continual Learning. Adapt the distilled student to Task B (brain MRI)
WITHOUT access to Task A data, using four strategies, then measure how much of
Task A is forgotten.

Strategies
----------
  naive      : just fine-tune on Task B            (baseline / lower bound)
  ewc        : Elastic Weight Consolidation        (Fisher-weighted penalty)
  lwf        : Learning without Forgetting          (distill old Task-A outputs)
  bn_freeze  : freeze the ENTIRE backbone (incl. BN running stats + affine
               params) and only train the Task-B head (architectural
               stabilization / upper bound on retention)

Each method starts from the SAME distilled student (a fresh copy), so the
comparison is fair.

Public API:
    continual_train(student_state, method, task_A_data, task_B_data, student_name)
        -> trained model
    run_all_methods(student_state, task_A_data, task_B_data, ref_acc, student_name)
        -> pandas.DataFrame of results
"""

import copy
import numpy as np
import pandas as pd

import torch
import torch.nn as nn

import config
from src import models, engine
from src.distillation import kd_logit_loss


# --------------------------------------------------------------------------- #
#  BN FREEZE
# --------------------------------------------------------------------------- #
def _freeze_bn(model):
    """Put all BatchNorm layers in eval mode (stops running-stat updates) and
    stop their affine gradients — locks Task-A feature statistics."""
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
            m.eval()
            if m.weight is not None:
                m.weight.requires_grad_(False)
            if m.bias is not None:
                m.bias.requires_grad_(False)


def _freeze_backbone(model):
    """FULL backbone freeze (architectural stabilization).

    This is stronger than _freeze_bn alone: it also stops gradient updates
    on every conv/linear weight in the shared backbone, not just the BN
    affine params. Without this, the backbone's conv weights keep shifting
    during Task B training while BN's running statistics stay locked to
    Task A's distribution — that mismatch is what was causing bn_freeze to
    score WORSE than naive fine-tuning (BN stats frozen but the features
    feeding into them no longer match those stats).

    Freezing the whole backbone removes that mismatch entirely: nothing in
    the shared trunk changes after Task A, so Task A retention should be
    near the teacher/distillation reference accuracy. Only the Task-B head
    is trained, so Task B accuracy may be a bit lower than the other
    methods — this trade-off is exactly what the stability-vs-plasticity
    plot is meant to show.
    """
    _freeze_bn(model)  # eval-mode BN + stop BN affine grads (belt-and-braces)
    for p in model.backbone.parameters():
        p.requires_grad_(False)


# --------------------------------------------------------------------------- #
#  EWC: Fisher information + penalty
# --------------------------------------------------------------------------- #
def _compute_fisher(model, loader, n_batches=None):
    """Estimate the diagonal Fisher information on Task A (importance of each
    parameter for the old task)."""
    n_batches = n_batches or config.EWC_FISHER_BATCHES
    device = config.get_device()
    fisher = {n: torch.zeros_like(p) for n, p in model.named_parameters()}
    ce = nn.CrossEntropyLoss()
    model.eval()
    count = 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        model.zero_grad()
        loss = ce(model(xb, task="a"), yb)
        loss.backward()
        for n, p in model.named_parameters():
            if p.grad is not None:
                fisher[n] += p.grad.detach() ** 2
        count += 1
        if count >= n_batches:
            break
    for n in fisher:
        fisher[n] /= max(count, 1)
    theta_star = {n: p.detach().clone() for n, p in model.named_parameters()}
    return fisher, theta_star


def _ewc_penalty(model, fisher, theta_star):
    loss = 0.0
    for n, p in model.named_parameters():
        loss = loss + (fisher[n] * (p - theta_star[n]) ** 2).sum()
    return loss


# --------------------------------------------------------------------------- #
#  MAIN: train one continual method
# --------------------------------------------------------------------------- #
def continual_train(student_state, method, task_A_data, task_B_data,
                    student_name=None, epochs=None, lr=None,
                    ewc_lambda=None, lwf_alpha=None, verbose=True):
    """Adapt a fresh copy of the distilled student to Task B with `method`."""
    device = config.get_device()
    epochs = epochs or config.CONTINUAL_EPOCHS
    ewc_lambda = config.EWC_LAMBDA if ewc_lambda is None else ewc_lambda
    lwf_alpha = config.LWF_ALPHA if lwf_alpha is None else lwf_alpha
    student_name = student_name or config.MAIN_STUDENT

    # fresh student loaded from the distilled snapshot
    model = models.build_student(student_name).to(device)
    model.load_state_dict(student_state)

    # ----------------------------------------------------------------- #
    # Build the optimizer's parameter list.
    #
    # For bn_freeze we freeze the WHOLE backbone (see _freeze_backbone
    # docstring above) and only optimize the Task-B head. For every other
    # method, the backbone + Task-B head are both trainable, as before.
    # ----------------------------------------------------------------- #
    if method == "bn_freeze":
        _freeze_backbone(model)
        opt = engine.make_optimizer(list(model.head_b.parameters()), lr)
    else:
        opt = engine.make_optimizer(
            list(model.backbone.parameters()) + list(model.head_b.parameters()), lr)

    crit = nn.CrossEntropyLoss(weight=task_B_data["class_weights"])

    # method-specific setup
    fisher = theta_star = old_model = None
    if method == "ewc":
        fisher, theta_star = _compute_fisher(model, task_A_data["train_eval_loader"])
    if method == "lwf":
        old_model = models.build_student(student_name).to(device)
        old_model.load_state_dict(student_state)
        old_model.eval()

    for ep in range(epochs):
        model.train()
        if method == "bn_freeze":
            _freeze_backbone(model)    # re-apply each epoch (train() resets BN
                                        # to train-mode and would otherwise also
                                        # re-enable requires_grad via .train())
        run_loss, total = 0.0, 0
        for xb, yb in task_B_data["train_loader"]:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = crit(model(xb, task="b"), yb)

            if method == "ewc":
                loss = loss + (ewc_lambda / 2.0) * _ewc_penalty(model, fisher, theta_star)
            if method == "lwf":
                with torch.no_grad():
                    old_a = old_model(xb, task="a")
                new_a = model(xb, task="a")
                loss = loss + lwf_alpha * kd_logit_loss(new_a, old_a,
                                                        config.LWF_TEMPERATURE)

            loss.backward()
            opt.step()
            run_loss += loss.item() * xb.size(0)
            total += xb.size(0)

        if verbose:
            accB = engine.evaluate(model, task_B_data["val_loader"], "b")["acc"]
            print(f"  [{method}] epoch {ep+1}/{epochs}  "
                  f"loss={run_loss/total:.3f}  val_acc(B)={accB:.3f}")

    return model


# --------------------------------------------------------------------------- #
#  RUN ALL FOUR METHODS + COMPARE
# --------------------------------------------------------------------------- #
def run_all_methods(student_state, task_A_data, task_B_data, ref_acc,
                    student_name=None, save=True):
    """Train every CL method, measure Task-A retention vs Task-B accuracy."""
    student_name = student_name or config.MAIN_STUDENT
    rows, trained = [], {}

    for method in config.CL_METHODS:
        print("=" * 50, f"{method}  ({student_name})")
        model = continual_train(student_state, method, task_A_data, task_B_data,
                                student_name=student_name)
        trained[method] = model

        accA = engine.evaluate(model, task_A_data["test_loader"], "a")["acc"]
        accB = engine.evaluate(model, task_B_data["test_loader"], "b")["acc"]
        rows.append({
            "student": student_name,
            "method": method,
            "TaskA_retention_%": round(engine.retention(accA, ref_acc), 2),
            "TaskA_acc": round(accA, 4),
            "TaskB_acc": round(accB, 4),
        })
        print(f"  -> Task A acc={accA:.3f}  Task B acc={accB:.3f}  "
              f"retention={engine.retention(accA, ref_acc):.1f}%")
        if save:
            engine.save_checkpoint(model, f"cl_{student_name}_{method}")

    df = pd.DataFrame(rows).sort_values("TaskA_retention_%", ascending=False)
    return df, trained
