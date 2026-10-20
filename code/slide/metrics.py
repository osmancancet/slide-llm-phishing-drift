import numpy as np
from sklearn.metrics import (precision_score, recall_score, f1_score, matthews_corrcoef,
                             average_precision_score, roc_auc_score, confusion_matrix)


def metrics(y, s, thr=0.5):
    p = (s >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, p, labels=[0, 1]).ravel()
    out = dict(precision=precision_score(y, p, zero_division=0), recall=recall_score(y, p, zero_division=0),
               f1=f1_score(y, p, zero_division=0), f1_macro=f1_score(y, p, average="macro", zero_division=0),
               mcc=matthews_corrcoef(y, p), fpr=fp / max(fp + tn, 1), acc=(tp + tn) / len(y),
               tp=int(tp), fp=int(fp), tn=int(tn), fn=int(fn))
    out["brier"] = float(np.mean((np.asarray(s, float) - y) ** 2))
    both = len(np.unique(y)) == 2
    out["pr_auc"] = average_precision_score(y, s) if both else np.nan
    out["roc_auc"] = roc_auc_score(y, s) if both else np.nan
    return out
