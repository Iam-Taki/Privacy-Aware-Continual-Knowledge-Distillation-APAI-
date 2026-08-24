"""
config.py
=========
Central configuration for the whole project. NOTHING is hard-coded elsewhere —
every path, hyper-parameter, model name and privacy budget lives here.

Works on BOTH Google Colab and a local machine without any code change:
the PROJECT path is auto-detected.
"""

import os


# --------------------------------------------------------------------------- #
#  PATHS  (auto-detect Colab vs local)
# --------------------------------------------------------------------------- #
def _detect_project_root() -> str:
    """Return the project root folder on Colab or local automatically."""
    colab_path = "/content/drive/MyDrive/APAI_Project"
    if os.path.isdir(colab_path):
        return colab_path
    # local: the folder that contains this config.py
    return os.path.dirname(os.path.abspath(__file__))


PROJECT_ROOT = _detect_project_root()
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "checkpoints")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
FIGURE_DIR = os.path.join(RESULTS_DIR, "figures")

# create output folders if they do not exist yet
for _d in (CHECKPOINT_DIR, RESULTS_DIR, FIGURE_DIR):
    os.makedirs(_d, exist_ok=True)


# --------------------------------------------------------------------------- #
#  REPRODUCIBILITY
# --------------------------------------------------------------------------- #
SEED = 42


# --------------------------------------------------------------------------- #
#  DATA
# --------------------------------------------------------------------------- #
# Kaggle dataset slugs (downloaded via kagglehub)
DATASET_A_SLUG = "paultimothymooney/chest-xray-pneumonia"   # Task A
DATASET_B_SLUG = "masoudnickparvar/brain-tumor-mri-dataset"  # Task B

# Folder names inside each dataset -> the ORDER defines the integer labels
TASK_A_CLASSES = ["NORMAL", "PNEUMONIA"]                       # 2 classes
TASK_B_CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]  # 4 classes

NUM_CLASSES_A = len(TASK_A_CLASSES)
NUM_CLASSES_B = len(TASK_B_CLASSES)

IMAGE_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# stratified split ratios (train / val / test) -> val+test share the remainder
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Cap the number of TRAIN images per task to keep Colab fast.
# Set to None for the final run (uses all images).
TRAIN_CAP = 3000

BATCH_SIZE = 32
NUM_WORKERS = 0


# --------------------------------------------------------------------------- #
#  MODELS  (this is what makes the study "big": 2 teachers x 2 students)
# --------------------------------------------------------------------------- #
TEACHERS = ["resnet50", "vit_b_16"]            # CNN vs Transformer teacher
STUDENTS = ["mobilenetv3", "efficientnet_b0"]  # two lightweight students

# the "main" pair used for the core pipeline (others are run as ablation)
MAIN_TEACHER = "resnet50"
MAIN_STUDENT = "mobilenetv3"


# --------------------------------------------------------------------------- #
#  TRAINING
# --------------------------------------------------------------------------- #
LR = 1e-3
MOMENTUM = 0.9            # used if an SGD optimizer is selected
OPTIMIZER = "adam"       # "adam" or "sgd"

TEACHER_EPOCHS = 5
DISTILL_EPOCHS = 5
CONTINUAL_EPOCHS = 8


# --------------------------------------------------------------------------- #
#  KNOWLEDGE DISTILLATION
# --------------------------------------------------------------------------- #
KD_TEMPERATURE = 3.0     # softmax temperature T
KD_ALPHA = 0.5           # weight: alpha*CE + (1-alpha)*KD


# --------------------------------------------------------------------------- #
#  CONTINUAL LEARNING
# --------------------------------------------------------------------------- #
CL_METHODS = ["naive", "ewc", "lwf", "bn_freeze"]

EWC_LAMBDA = 50000.0     # EWC regularization strength
EWC_FISHER_BATCHES = 25  # how many batches to estimate the Fisher matrix
LWF_ALPHA = 1.0          # LwF distillation weight
LWF_TEMPERATURE = 3.0

# ablation sweep for EWC lambda
EWC_LAMBDA_SWEEP = [0, 500, 2000, 8000]


# --------------------------------------------------------------------------- #
#  PRIVACY
# --------------------------------------------------------------------------- #
DP_EPSILONS = [8.0, 3.0, 1.0]   # privacy budgets to test (smaller = stronger)
DP_DELTA = 1e-5
DP_MAX_GRAD_NORM = 1.0
DP_EPOCHS = 15
DP_MAX_PHYSICAL_BATCH = 8        # kept for backward-compat (unused by the
                                  # manual/no-Opacus DP-SGD implementation)

# Learning rate specifically for DP-SGD training (src/privacy.py).
# DP-SGD's gradient clipping + noise make a "normal" SGD lr (e.g. 0.05) far
# too large for this small CNN -- empirically, loss simply doesn't decrease
# at lr=0.05 or lr=0.005 (the model collapses to predicting the majority
# class). A small-lr sweep (0.001 / 0.01 / 0.02) found 0.02 gives the best
# accuracy on the non-DP sanity check, so that's used as the DP learning
# rate too. Kept separate from LR (used for the main teacher/student/
# continual-learning pipeline) since that pipeline is unaffected and uses
# Adam at LR=1e-3, a very different optimizer/scale.
LR_DP = 0.02


# --------------------------------------------------------------------------- #
#  DEVICE
# --------------------------------------------------------------------------- #
def get_device():
    import torch
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def summary():
    """Print the key settings — handy sanity check at the top of the notebook."""
    print("PROJECT_ROOT :", PROJECT_ROOT)
    print("Device       :", get_device())
    print("Task A        :", TASK_A_CLASSES, f"({NUM_CLASSES_A} classes)")
    print("Task B        :", TASK_B_CLASSES, f"({NUM_CLASSES_B} classes)")
    print("Teachers      :", TEACHERS)
    print("Students      :", STUDENTS)
    print("CL methods    :", CL_METHODS)
    print("Train cap     :", TRAIN_CAP)
