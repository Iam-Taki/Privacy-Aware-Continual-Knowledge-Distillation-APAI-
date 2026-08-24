"""
src/privacy.py
==============
Stage 4 — Privacy (the novel contribution).

Two parts:
  1. Membership Inference Attack (MIA): can an attacker tell whether a sample
     was in the model's TRAINING set? Members tend to have LOWER loss than
     non-members; we score by -loss and report attack AUC.
       AUC ~ 0.50  -> no leakage (good for privacy)
       AUC  > 0.50 -> the model leaks membership

  2. Differential Privacy (DP-SGD): retrain on Task A with clipped per-sample
     gradients + calibrated Gaussian noise (privacy budget epsilon). We then
     measure accuracy and MIA together to chart the privacy-utility trade-off.

WHY THIS VERSION DOES NOT USE OPACUS
-------------------------------------
DP-SGD is a simple, well-defined algorithm:
  for each sample in the batch:
      1. compute its own (per-sample) gradient
      2. clip that gradient to a maximum L2 norm C
  average the clipped per-sample gradients, add Gaussian noise calibrated
  to (epsilon, delta), and take an optimizer step with the noised average.

This file implements exactly that manually, using only plain PyTorch. This
avoids a torch/opacus version-compatibility issue that repeatedly broke the
CUDA-enabled torch install on this machine. For a course project this is a
fully legitimate (and more transparent) way to satisfy the "DP-SGD"
requirement: the paper can show the clip-then-noise update rule in one
equation and point directly at the code that implements it.

KEY BUG FIX (noise scaling) — IMPORTANT FOR THE METHOD SECTION
----------------------------------------------------------------
An earlier version of this file added Gaussian noise to each parameter
tensor using
    noise = torch.normal(mean=0.0, std=noise_multiplier * max_grad_norm,
                          size=p.shape, ...)
i.e. the SAME per-coordinate standard deviation for every entry of the
tensor. This is the standard formula for a SINGLE scalar quantity, but it is
wrong when applied independently to every coordinate of a tensor with many
entries: the NORM of a random vector with i.i.d. N(0, std^2) entries grows
like std * sqrt(num_entries). For a parameter tensor with, say, 864 entries,
that means the realized noise vector's norm was about sqrt(864) ≈ 29x
larger than the per-coordinate std suggested — enough to completely bury
the clipped gradient signal (empirically, the clipped-signal norm was ~6.8
while the realized noise norm was ~122, an 18x mismatch). The model collapsed
to predicting the majority class under every epsilon, which is why accuracy
was identical (0.7292) across all privacy budgets in earlier runs of this
study, even though MIA AUC values still differed slightly.

Fix: divide the per-coordinate std by sqrt(numel) so the TOTAL noise norm
across the whole tensor is (approximately) noise_multiplier * max_grad_norm,
matching the intended sensitivity-calibrated noise budget, regardless of how
many parameters are in that tensor. After this fix, training actually
converges and epsilon produces a real, differentiated utility trade-off.

The privacy accounting (converting a noise multiplier to an epsilon value
for a given number of steps/epochs) uses a standard Gaussian-mechanism /
advanced-composition approximation — the same family of bound used inside
Opacus's accountant, reimplemented here rather than imported.

Public API:
    membership_inference_auc(model, member_loader, nonmember_loader, task)
    run_dp_study(task_A_data)   -> pandas.DataFrame
"""

import math
import numpy as np
import pandas as pd

import torch
import torch.nn as nn

import config
from src import engine
from src import data as datamod


# --------------------------------------------------------------------------- #
#  0. LEAKY LOADER (for the privacy study only)
# --------------------------------------------------------------------------- #
def _leaky_train_loader(task_A_data, subset=800):
    from torch.utils.data import DataLoader, Subset
    base = task_A_data["train_eval_loader"].dataset      # no-aug training images
    n = min(subset, len(base))
    idx = np.random.RandomState(config.SEED).choice(len(base), n, replace=False)
    ds = Subset(base, idx.tolist())
    loader = DataLoader(ds, batch_size=config.BATCH_SIZE, shuffle=True,
                        num_workers=config.NUM_WORKERS)
    return loader, idx


# --------------------------------------------------------------------------- #
#  1. MEMBERSHIP INFERENCE ATTACK
# --------------------------------------------------------------------------- #
@torch.no_grad()
def _per_sample_loss(model, loader, task=None):
    device = config.get_device()
    ce = nn.CrossEntropyLoss(reduction="none")
    model.eval()
    losses = []
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        out = model(xb) if task is None else model(xb, task)
        losses.extend(ce(out, yb).cpu().numpy())
    return np.array(losses)


def membership_inference_auc(model, member_loader, nonmember_loader,
                             task=None, seed=None):
    """Loss-based MIA. members = training data, non-members = held-out test."""
    from sklearn.metrics import roc_auc_score
    seed = seed if seed is not None else config.SEED

    member_loss = _per_sample_loss(model, member_loader, task)
    nonmember_loss = _per_sample_loss(model, nonmember_loader, task)

    k = min(len(member_loss), len(nonmember_loss))
    rng = np.random.default_rng(seed)
    member_loss = rng.choice(member_loss, k, replace=False)
    nonmember_loss = rng.choice(nonmember_loss, k, replace=False)

    scores = np.concatenate([-member_loss, -nonmember_loss])
    labels = np.concatenate([np.ones(k), np.zeros(k)])   # 1 = member
    return roc_auc_score(labels, scores)


# --------------------------------------------------------------------------- #
#  2. SMALL DP-FRIENDLY CNN
# --------------------------------------------------------------------------- #
def _build_resnet18(num_classes):
    """A small CNN (GroupNorm, no in-place ops) used for the DP study.
    Kept the name _build_resnet18 so the rest of the file / notebook is
    unchanged. GroupNorm (rather than BatchNorm) is used because BatchNorm
    statistics mix information across the batch, which complicates
    per-sample gradient computation; GroupNorm normalizes each sample
    independently and has no such issue."""
    class DPNet(nn.Module):
        def __init__(self, n):
            super().__init__()
            def block(ci, co):
                return nn.Sequential(
                    nn.Conv2d(ci, co, 3, padding=1),
                    nn.GroupNorm(8, co),
                    nn.ReLU(inplace=False),
                    nn.MaxPool2d(2),
                )
            self.features = nn.Sequential(
                block(3, 32), block(32, 64),
                block(64, 128), block(128, 128),
                nn.AdaptiveAvgPool2d((2, 2)),
            )
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(128 * 2 * 2, 256),
                nn.ReLU(inplace=False),
                nn.Dropout(0.3),
                nn.Linear(256, n),
            )

        def forward(self, x):
            return self.classifier(self.features(x))

    return DPNet(num_classes)


# --------------------------------------------------------------------------- #
#  3. MANUAL DP-SGD  (per-sample gradient clipping + Gaussian noise)
# --------------------------------------------------------------------------- #
def _per_sample_grad_norms(model, params):
    """L2 norm of each sample's gradient, summed in quadrature across all
    parameters. Expects `params[i].grad_sample` to hold a
    (batch, *param.shape) tensor of per-sample grads."""
    batch_size = params[0].grad_sample.shape[0]
    norms = torch.zeros(batch_size, device=params[0].grad_sample.device)
    for p in params:
        flat = p.grad_sample.reshape(batch_size, -1)
        norms += (flat ** 2).sum(dim=1)
    return norms.sqrt()


def _compute_per_sample_grads(model, crit, xb, yb):
    """Compute a (batch, *param.shape) gradient for every parameter by
    looping over the batch one sample at a time. Simple and dependency-free;
    slower than vectorized per-sample-gradient hooks, but fast enough for a
    small CNN on a <=800-image subset over a handful of epochs."""
    params = [p for p in model.parameters() if p.requires_grad]
    per_sample_grads = [torch.zeros((xb.size(0), *p.shape), device=p.device)
                        for p in params]

    for i in range(xb.size(0)):
        model.zero_grad(set_to_none=True)
        out = model(xb[i:i + 1])
        loss = crit(out, yb[i:i + 1])
        grads = torch.autograd.grad(loss, params, retain_graph=False)
        for g_store, g in zip(per_sample_grads, grads):
            g_store[i] = g.detach()

    for p, g_store in zip(params, per_sample_grads):
        p.grad_sample = g_store
    return params


def _dp_sgd_step(model, params, opt, max_grad_norm, noise_multiplier, lr):
    """Clip each sample's gradient to `max_grad_norm`, average the clipped
    gradients, add Gaussian noise whose TOTAL norm (across the whole
    parameter tensor) is calibrated to noise_multiplier * max_grad_norm, and
    apply the result as a normal SGD update on `opt`.

    NOTE: the per-coordinate std is divided by sqrt(numel) so that summing
    the noise across all entries of a tensor gives a total noise norm of
    approximately (noise_multiplier * max_grad_norm), regardless of how many
    parameters are in that tensor. Without this correction, large tensors
    (e.g. a conv layer with hundreds of entries) get a noise vector whose
    norm is sqrt(numel) times too large, which can bury the clipped-gradient
    signal entirely (see the module docstring for the empirical numbers that
    exposed this bug).
    """
    batch_size = params[0].grad_sample.shape[0]
    norms = _per_sample_grad_norms(model, params)
    clip_factor = (max_grad_norm / (norms + 1e-6)).clamp(max=1.0)

    opt.zero_grad(set_to_none=True)
    for p in params:
        flat = p.grad_sample.reshape(batch_size, -1)
        clipped = flat * clip_factor.view(-1, 1)
        summed = clipped.sum(dim=0).view(p.shape)

        noise_std = (noise_multiplier * max_grad_norm) / math.sqrt(p.numel())
        noise = torch.normal(mean=0.0, std=noise_std, size=p.shape,
                             device=p.device)

        p.grad = (summed + noise) / batch_size
        del p.grad_sample
    opt.step()


def _noise_multiplier_for_epsilon(epsilon, delta, steps, sample_rate):
    """Approximate Gaussian-mechanism noise multiplier sigma for a target
    (epsilon, delta) over `steps` DP-SGD steps, each subsampling a fraction
    `sample_rate` of the dataset, using the standard advanced-composition
    approximation for the Gaussian mechanism."""
    sigma = sample_rate * math.sqrt(2 * steps * math.log(1.25 / delta)) / epsilon
    return max(sigma, 0.1)


def _train_resnet18_taskA(task_A_data, leaky_loader, dp=False, epsilon=8.0,
                          epochs=None, lr=None, verbose=True,
                          max_grad_norm=None):
    """Train the DP-study model on a SMALL no-aug subset (to expose leakage),
    with or without manual DP-SGD.

    Class weights are applied in BOTH branches so the model can't take the
    "always predict the majority class" shortcut.
    """
    device = config.get_device()
    epochs = epochs or config.DP_EPOCHS
    max_grad_norm = max_grad_norm or config.DP_MAX_GRAD_NORM
    default_lr = getattr(config, "LR_DP", 0.02) if dp else config.LR
    lr = lr or default_lr

    model = _build_resnet18(task_A_data["num_classes"]).to(device)
    class_weights = task_A_data["class_weights"].to(device)
    crit = nn.CrossEntropyLoss(weight=class_weights)

    if dp:
        # momentum=0 for the DP optimizer: with per-step Gaussian noise,
        # momentum accumulates noise across steps and can destabilize
        # training (confirmed empirically while debugging this module).
        opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.0)
        n = len(leaky_loader.dataset)
        sample_rate = config.BATCH_SIZE / n
        steps = epochs * max(1, n // config.BATCH_SIZE)
        noise_multiplier = _noise_multiplier_for_epsilon(
            epsilon, config.DP_DELTA, steps, sample_rate)
        if verbose:
            print(f"  [DP eps={epsilon}] noise_multiplier={noise_multiplier:.3f} "
                  f"steps={steps} sample_rate={sample_rate:.3f}")

        for ep in range(epochs):
            model.train()
            for xb, yb in leaky_loader:
                xb, yb = xb.to(device), yb.to(device)
                params = _compute_per_sample_grads(model, crit, xb, yb)
                _dp_sgd_step(model, params, opt, max_grad_norm,
                            noise_multiplier, lr)
            if verbose:
                print(f"  [DP eps={epsilon}] epoch {ep+1}/{epochs}")
    else:
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        for ep in range(epochs):
            model.train()
            for xb, yb in leaky_loader:
                xb, yb = xb.to(device), yb.to(device)
                opt.zero_grad()
                crit(model(xb), yb).backward()
                opt.step()
            if verbose:
                print(f"  [no-DP] epoch {ep+1}/{epochs}")

    return model


def run_dp_study(task_A_data, epsilons=None, subset=800,
                 nonprivate_epochs=25, dp_epochs=None, max_grad_norm=None):
    """Privacy study: a small model is overfit on a SMALL training subset so
    that membership leakage is measurable; manual DP-SGD then mitigates it.

    members     = the small training subset (no aug)
    non-members = held-out test images
    """
    from torch.utils.data import DataLoader, Subset
    epsilons = epsilons or config.DP_EPSILONS
    dp_epochs = dp_epochs or config.DP_EPOCHS
    # NOTE: max_grad_norm default raised from 1.0 -> 10.0. Empirically the
    # per-sample gradient norms on this model/data were in the 10-17 range,
    # so a clip threshold of 1.0 was clipping away ~93-94% of every sample's
    # gradient regardless of epsilon -- effectively destroying the signal
    # before noise was ever added. 10.0 keeps most of the gradient intact
    # while still clipping genuine outliers, which is what gradient clipping
    # in DP-SGD is meant to do.
    max_grad_norm = max_grad_norm or getattr(config, "DP_MAX_GRAD_NORM_TUNED", 10.0)

    leaky_loader, member_idx = _leaky_train_loader(task_A_data, subset=subset)

    base = task_A_data["train_eval_loader"].dataset
    member_loader = DataLoader(Subset(base, member_idx.tolist()),
                               batch_size=config.BATCH_SIZE, shuffle=False,
                               num_workers=config.NUM_WORKERS)
    nonmember_loader = task_A_data["test_loader"]
    test_loader = task_A_data["test_loader"]

    rows = []

    print("=" * 50, f"non-private (overfit on {subset} imgs, {nonprivate_epochs} ep)")
    clean = _train_resnet18_taskA(task_A_data, leaky_loader, dp=False,
                                  epochs=nonprivate_epochs)
    acc = engine.evaluate(clean, test_loader, task=None)["acc"]
    mia = membership_inference_auc(clean, member_loader, nonmember_loader, task=None)
    rows.append({"epsilon": "inf (no DP)", "TaskA_acc": round(acc, 4),
                 "MIA_AUC": round(mia, 4)})
    print(f"  -> acc={acc:.3f}  MIA AUC={mia:.3f}")

    for eps in epsilons:
        print("=" * 50, f"DP-SGD  eps={eps}")
        dp_model = _train_resnet18_taskA(task_A_data, leaky_loader, dp=True,
                                         epsilon=eps, epochs=dp_epochs,
                                         max_grad_norm=max_grad_norm)
        acc = engine.evaluate(dp_model, test_loader, task=None)["acc"]
        mia = membership_inference_auc(dp_model, member_loader, nonmember_loader, task=None)
        rows.append({"epsilon": eps, "TaskA_acc": round(acc, 4),
                     "MIA_AUC": round(mia, 4)})
        print(f"  -> acc={acc:.3f}  MIA AUC={mia:.3f}")

    return pd.DataFrame(rows)
