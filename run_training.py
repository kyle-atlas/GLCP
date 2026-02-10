import os
import subprocess
import threading

# -------------------- CONFIG --------------------
os.environ["nnUNet_results"] = "/home/kyle/Documents/nnUNet_results"
os.environ["nnUNet_preprocessed"] = "/home/kyle/Documents/nnUNet_preprocessed"
os.environ["nnUNet_raw"] = "/home/kyle/Documents/nnUNet_raw"

task_id = "700"
planner = "nnUNetPlanner"
plans_name = "nnUNetPlans"
trainer_name = "nnUNetTrainer_3DUnet_NoDeepSupervision_CE_DC_crit_ske_refine"


# Plan & preprocess -----
"""
subprocess.run([
    "/home/kyle/PycharmProjects/heartlung-calcium-training/.venv/bin/nnUNetv2_plan_and_preprocess",
    "-d", task_id,
    "--verify_dataset_integrity",
    "-np", "12",
], check=True)

"""
# Train x folds
for fold in map(str, range(0, 1)):
    subprocess.run([
        "/home/kyle/PycharmProjects/heartlung-calcium-training/.venv/bin/nnUNetv2_train",
        task_id,
        "3d_fullres",
        fold,
        "-tr", trainer_name,
        "-p", plans_name,
        "--npz",
        "--c"
    ], check=True)

