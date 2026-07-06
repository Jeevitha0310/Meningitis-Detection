import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)

# ===========================
# FUSION FUNCTION
# ===========================
def fuse_predictions(clinical_pred, image_pred):
    """
    Simple late fusion rule:
    - If both models agree → return that class
    - If not → mark as uncertain
    """
    if clinical_pred == image_pred:
        return clinical_pred
    else:
        return "Uncertain"

# ===========================
# DEVICE SIMULATION DATA
# ===========================
# Replace this with your real model outputs in loop
# Format: (clinical_pred, image_pred, true_label)

test_data = [
    (0, 0, 0),
    (1, 1, 1),
    (0, 1, 1),
    (1, 0, 0),
    (1, 1, 1),
    (0, 0, 0),
]

# ===========================
# STORAGE
# ===========================
all_preds = []
all_labels = []

# ===========================
# APPLY FUSION
# ===========================
for clinical_pred, image_pred, true_label in test_data:

    fused_pred = fuse_predictions(clinical_pred, image_pred)

    all_preds.append(fused_pred)
    all_labels.append(true_label)

# ===========================
# REMOVE UNCERTAIN CASES
# ===========================
filtered_preds = []
filtered_labels = []

for pred, label in zip(all_preds, all_labels):
    if pred != "Uncertain":
        filtered_preds.append(pred)
        filtered_labels.append(label)

# ===========================
# METRICS
# ===========================
accuracy = accuracy_score(filtered_labels, filtered_preds)
precision = precision_score(filtered_labels, filtered_preds, average="weighted", zero_division=0)
recall = recall_score(filtered_labels, filtered_preds, average="weighted", zero_division=0)
f1 = f1_score(filtered_labels, filtered_preds, average="weighted", zero_division=0)

# ===========================
# RESULTS
# ===========================
print("\n📊 FUSION MODEL RESULTS")
print("----------------------------")

print(f"Accuracy  : {accuracy * 100:.2f}%")
print(f"Precision : {precision * 100:.2f}%")
print(f"Recall    : {recall * 100:.2f}%")
print(f"F1 Score  : {f1 * 100:.2f}%")

print("\n📄 Classification Report:\n")
print(classification_report(filtered_labels, filtered_preds, zero_division=0))