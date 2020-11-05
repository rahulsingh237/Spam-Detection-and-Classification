import numpy as np


# Operations on confusion matrix
def matrixcm(cm):
    matrix = []
    for ix,iy in np.ndindex(cm.shape):
        matrix.append(cm[ix,iy])
    return matrix

# calculating precision using confusion matrix
def precision(TP,FP):
    pr = (TP)/(TP+FP)
    return pr

# Calculating recall using confusion matrix
def recal(TP,FN):
    re = (TP)/(TP+FN)
    return re

# Calculating F1 score using confusion matrix
def f1scre(pr,re):
    f1 = 2*(re * pr) / (re + pr)  # 2*(Recall * Precision) / (Recall + Precision)
    return f1