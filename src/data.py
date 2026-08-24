"""
src/data.py
===========
Everything about data:
  - download both Kaggle datasets (via kagglehub)
  - collect image paths + labels (skipping macOS junk files)
  - stratified train / val / test split
  - PyTorch Dataset + DataLoader factory
  - class weights for imbalance

Public API (used from the notebook):
    download_datasets()           -> (path_A, path_B)
    build_task(task="A")          -> dict of loaders + metadata
    show_samples(task_data)       -> quick visual sanity check
"""

import os
import numpy as np
from PIL import Image, ImageFile

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split

import config

ImageFile.LOAD_TRUNCATED_IMAGES = True   # tolerate slightly corrupt files


# --------------------------------------------------------------------------- #
#  1. DOWNLOAD
# --------------------------------------------------------------------------- #
def download_datasets():
    """Download both datasets with kagglehub. Returns their local folders."""
    import kagglehub
    path_a = kagglehub.dataset_download(config.DATASET_A_SLUG)
    path_b = kagglehub.dataset_download(config.DATASET_B_SLUG)
    print("Task A folder:", path_a)
    print("Task B folder:", path_b)
    return path_a, path_b


# --------------------------------------------------------------------------- #
#  2. COLLECT IMAGE PATHS + LABELS
# --------------------------------------------------------------------------- #
def collect_images(root, class_names):
    """Walk `root`, map subfolders named like class_names to integer labels.

    Skips __MACOSX folders and ._ resource-fork stubs; dedups by filename
    (removes any nested-duplicate folders that some Kaggle zips contain).
    """
    cn = [c.lower() for c in class_names]
    paths, labels, seen = [], [], set()
    for r, _, files in os.walk(root):
        if "__MACOSX" in r:
            continue
        folder = os.path.basename(r).lower()
        if folder in cn:
            label = cn.index(folder)
            for f in files:
                if f.startswith("._"):
                    continue
                if f.lower().endswith((".jpg", ".jpeg", ".png")) and f not in seen:
                    seen.add(f)
                    paths.append(os.path.join(r, f))
                    labels.append(label)
    return np.array(paths), np.array(labels)


# --------------------------------------------------------------------------- #
#  3. SPLIT
# --------------------------------------------------------------------------- #
def stratified_split(paths, labels):
    """Split into train / val / test keeping class proportions."""
    test_size = config.VAL_RATIO + config.TEST_RATIO
    x_tr, x_tmp, y_tr, y_tmp = train_test_split(
        paths, labels, test_size=test_size,
        stratify=labels, random_state=config.SEED)
    # split the remainder into val and test
    rel_test = config.TEST_RATIO / test_size
    x_val, x_te, y_val, y_te = train_test_split(
        x_tmp, y_tmp, test_size=rel_test,
        stratify=y_tmp, random_state=config.SEED)
    return (x_tr, y_tr), (x_val, y_val), (x_te, y_te)


def _cap_train(x, y, n):
    """Randomly keep at most n training images (speed control)."""
    if n is None or len(x) <= n:
        return x, y
    idx = np.random.RandomState(config.SEED).choice(len(x), n, replace=False)
    return x[idx], y[idx]


# --------------------------------------------------------------------------- #
#  4. TRANSFORMS
# --------------------------------------------------------------------------- #
def get_transforms(train: bool):
    base = [transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE))]
    if train:
        base += [transforms.RandomHorizontalFlip(),
                 transforms.RandomRotation(10)]
    base += [transforms.ToTensor(),
             transforms.Normalize(config.IMAGENET_MEAN, config.IMAGENET_STD)]
    return transforms.Compose(base)


# --------------------------------------------------------------------------- #
#  5. DATASET
# --------------------------------------------------------------------------- #
class ImageDataset(Dataset):
    def __init__(self, paths, labels, tf):
        self.paths, self.labels, self.tf = paths, labels, tf

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert("RGB")
        return self.tf(img), int(self.labels[i])


def _loader(paths, labels, tf, shuffle):
    ds = ImageDataset(paths, labels, tf)
    return DataLoader(ds, batch_size=config.BATCH_SIZE, shuffle=shuffle,
                      num_workers=config.NUM_WORKERS)


def class_weights(labels, n_classes, device):
    counts = np.bincount(labels, minlength=n_classes)
    counts = np.clip(counts, 1, None)  # avoid divide-by-zero
    w = counts.sum() / (n_classes * counts)
    return torch.tensor(w, dtype=torch.float, device=device)


# --------------------------------------------------------------------------- #
#  6. PUBLIC: build everything for one task
# --------------------------------------------------------------------------- #
def build_task(task, dataset_path):
    """Build all loaders + metadata for Task 'A' or 'B'.

    Returns a dict:
        train_loader, train_eval_loader (no aug, for the attack),
        val_loader, test_loader, class_weights, num_classes, class_names, sizes
    """
    if task.upper() == "A":
        classes = config.TASK_A_CLASSES
        n = config.NUM_CLASSES_A
    else:
        classes = config.TASK_B_CLASSES
        n = config.NUM_CLASSES_B

    paths, labels = collect_images(dataset_path, classes)
    if len(paths) == 0:
        raise RuntimeError(f"No images found for Task {task} in {dataset_path}. "
                           f"Check the class folder names: {classes}")

    (x_tr, y_tr), (x_val, y_val), (x_te, y_te) = stratified_split(paths, labels)
    x_tr, y_tr = _cap_train(x_tr, y_tr, config.TRAIN_CAP)

    device = config.get_device()
    train_tf, eval_tf = get_transforms(True), get_transforms(False)

    data = {
        "train_loader":      _loader(x_tr,  y_tr,  train_tf, True),
        "train_eval_loader": _loader(x_tr,  y_tr,  eval_tf,  False),  # no aug
        "val_loader":        _loader(x_val, y_val, eval_tf,  False),
        "test_loader":       _loader(x_te,  y_te,  eval_tf,  False),
        "class_weights":     class_weights(y_tr, n, device),
        "num_classes":       n,
        "class_names":       classes,
        "sizes":             {"train": len(x_tr), "val": len(x_val), "test": len(x_te)},
    }
    print(f"[Task {task}] classes={classes}  "
          f"train={data['sizes']['train']} val={data['sizes']['val']} "
          f"test={data['sizes']['test']}")
    return data


# --------------------------------------------------------------------------- #
#  7. PUBLIC: quick visual check
# --------------------------------------------------------------------------- #
def show_samples(task_data, n=5):
    import matplotlib.pyplot as plt
    inv = transforms.Normalize(
        [-m / s for m, s in zip(config.IMAGENET_MEAN, config.IMAGENET_STD)],
        [1 / s for s in config.IMAGENET_STD])
    xb, yb = next(iter(task_data["val_loader"]))
    names = task_data["class_names"]
    fig, ax = plt.subplots(1, n, figsize=(3 * n, 3))
    for i in range(n):
        ax[i].imshow(inv(xb[i]).permute(1, 2, 0).clip(0, 1))
        ax[i].set_title(names[yb[i]], fontsize=9)
        ax[i].axis("off")
    plt.tight_layout()
    plt.show()
