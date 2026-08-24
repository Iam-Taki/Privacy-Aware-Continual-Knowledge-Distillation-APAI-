"""
src/experiments.py
==================
Option 3 — the "big" study that matches the proposal:
  - 2 teachers  (ResNet-50, ViT-B/16)
  - 2 students  (MobileNetV3, EfficientNet-B0)
  - 4 continual methods (naive, EWC, LwF, BN-freeze)
  - EWC lambda sweep (ablation)

Everything is checkpoint-based, so you can run ONE (teacher, student) pair at a
time and resume later — important for Colab timeouts. Heavy pairs (e.g. ViT)
can be run on a stronger machine and the checkpoints copied back.

Public API:
    run_pair(teacher_name, student_name, task_A, task_B)      -> DataFrame
    run_all_pairs(task_A, task_B, pairs=None)                 -> DataFrame
    run_lambda_sweep(student_state, task_A, task_B, ref_acc)  -> DataFrame
"""

import pandas as pd

import config
from src import models, engine, distillation, continual


# --------------------------------------------------------------------------- #
#  ONE (teacher, student) PAIR  ->  full pipeline
# --------------------------------------------------------------------------- #
def run_pair(teacher_name, student_name, task_A, task_B,
             force_retrain=False, save=True):
    """Teacher -> distill -> 4 continual methods, for one (teacher, student)."""
    tag = f"{teacher_name}__{student_name}"
    print("#" * 60)
    print(f"#  PAIR: teacher={teacher_name}   student={student_name}")
    print("#" * 60)

    # ---- 1. teacher (shared across students -> keyed by teacher only) ----
    teacher = models.build_teacher(teacher_name)
    tkey = f"teacher_{teacher_name}"
    if engine.checkpoint_exists(tkey) and not force_retrain:
        teacher = engine.load_checkpoint(teacher, tkey)
    else:
        print(f"[{tag}] training teacher {teacher_name} ...")
        teacher, _ = engine.train_supervised(
            teacher, task_A["train_loader"], task_A["val_loader"],
            weights=task_A["class_weights"], task=None,
            epochs=config.TEACHER_EPOCHS)
        if save:
            engine.save_checkpoint(teacher, tkey)

    # ---- 2. distill into this student ----
    student = models.build_student(student_name)
    skey = f"student_{tag}"
    if engine.checkpoint_exists(skey) and not force_retrain:
        student = engine.load_checkpoint(student, skey)
    else:
        print(f"[{tag}] distilling -> {student_name} ...")
        student, _ = distillation.distill(teacher, student, task_A)
        if save:
            engine.save_checkpoint(student, skey)

    # reference Task-A accuracy (for retention)
    ref_acc = engine.evaluate(student, task_A["test_loader"], "a")["acc"]
    print(f"[{tag}] reference Task A acc = {ref_acc:.4f}")

    # ---- 3. continual learning, 4 methods ----
    rows = []
    for method in config.CL_METHODS:
        ckey = f"cl_{tag}_{method}"
        if engine.checkpoint_exists(ckey) and not force_retrain:
            m = engine.load_checkpoint(models.build_student(student_name), ckey)
        else:
            print(f"[{tag}] continual: {method}")
            m = continual.continual_train(
                student.state_dict(), method, task_A, task_B,
                student_name=student_name, verbose=False)
            if save:
                engine.save_checkpoint(m, ckey)

        accA = engine.evaluate(m, task_A["test_loader"], "a")["acc"]
        accB = engine.evaluate(m, task_B["test_loader"], "b")["acc"]
        rows.append({
            "teacher": teacher_name, "student": student_name, "method": method,
            "TaskA_retention_%": round(engine.retention(accA, ref_acc), 2),
            "TaskA_acc": round(accA, 4), "TaskB_acc": round(accB, 4),
        })
        print(f"   -> A={accA:.3f}  B={accB:.3f}  "
              f"retention={engine.retention(accA, ref_acc):.1f}%")

    df = pd.DataFrame(rows)
    if save:
        df.to_csv(f"{config.RESULTS_DIR}/pair_{tag}.csv", index=False)
    return df


# --------------------------------------------------------------------------- #
#  ALL PAIRS  (2 x 2 = 4 combinations)
# --------------------------------------------------------------------------- #
def run_all_pairs(task_A, task_B, pairs=None, force_retrain=False):
    """Run every (teacher, student) combination and concatenate the results.

    pairs : optional list of (teacher, student) tuples to run a subset
            (handy for splitting heavy runs). Default = full 2x2 cross-product.
    """
    if pairs is None:
        pairs = [(t, s) for t in config.TEACHERS for s in config.STUDENTS]

    all_rows = []
    for (t, s) in pairs:
        df = run_pair(t, s, task_A, task_B, force_retrain=force_retrain)
        all_rows.append(df)

    full = pd.concat(all_rows, ignore_index=True)
    full = full.sort_values(["teacher", "student", "TaskA_retention_%"],
                            ascending=[True, True, False])
    full.to_csv(f"{config.RESULTS_DIR}/all_pairs.csv", index=False)
    print("\nSaved combined results -> results/all_pairs.csv")
    return full


# --------------------------------------------------------------------------- #
#  EWC LAMBDA SWEEP  (ablation)
# --------------------------------------------------------------------------- #
def run_lambda_sweep(student_state, task_A, task_B, ref_acc,
                     student_name=None, lambdas=None):
    """Ablation: vary EWC lambda and watch retention vs Task-B accuracy."""
    student_name = student_name or config.MAIN_STUDENT
    lambdas = lambdas or config.EWC_LAMBDA_SWEEP

    rows = []
    for lam in lambdas:
        print(f"[lambda sweep] lambda={lam}")
        if lam <= 0:
            m = continual.continual_train(student_state, "naive", task_A, task_B,
                                          student_name=student_name, verbose=False)
        else:
            m = continual.continual_train(student_state, "ewc", task_A, task_B,
                                          student_name=student_name,
                                          ewc_lambda=lam, verbose=False)
        accA = engine.evaluate(m, task_A["test_loader"], "a")["acc"]
        accB = engine.evaluate(m, task_B["test_loader"], "b")["acc"]
        rows.append({"lambda": lam,
                     "TaskA_retention_%": round(engine.retention(accA, ref_acc), 2),
                     "TaskB_acc": round(accB, 4)})
        print(f"   -> retention={engine.retention(accA, ref_acc):.1f}%  B={accB:.3f}")

    df = pd.DataFrame(rows)
    df.to_csv(f"{config.RESULTS_DIR}/lambda_sweep.csv", index=False)
    return df
