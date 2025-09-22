import torch

def sensitivity_at_fixed_fpr(y_true, y_score, fpr_target=0.05):
    from sklearn.metrics import roc_curve
    fpr, tpr, thr = roc_curve(y_true, y_score)
    idx = (fpr <= fpr_target).nonzero()[0].max()
    return tpr[idx], thr[idx]
