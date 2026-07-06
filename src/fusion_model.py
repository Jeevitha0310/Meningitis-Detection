import numpy as np

def fuse_predictions(clinical_pred, image_pred):
    # Simple fusion (you can improve later)
    if clinical_pred == image_pred:
        return clinical_pred
    else:
        return "Uncertain"