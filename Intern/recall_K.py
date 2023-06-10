from typing import List
import numpy as np


def recall_at_k(labels: List[int], scores: List[float], k=5) -> float:
    inp = []
    for i in range(len(labels)):
        inp.append([labels[i], scores[i]])
    sorted_inp = sorted(inp, key = lambda x: x[1], reverse=True)
    labels_k = 0
    for i in range(k):
        labels_k += sorted_inp[i][0]
    return labels_k / (sum(labels) + np.finfo(float).eps)


def precision_at_k(labels: List[int], scores: List[float], k=5) -> float:
    inp = []
    for i in range(len(labels)):
        inp.append([labels[i], scores[i]])
    sorted_inp = sorted(inp, key = lambda x: x[1], reverse=True)
    labels_k = 0
    for i in range(k):
        labels_k += sorted_inp[i][0]
    return labels_k / (k + np.finfo(float).eps)


def specificity_at_k(labels: List[int], scores: List[float], k=5) -> float:
    ''' TN - то что не в топ-к и то что имеет labels = 0
        FP - то что в топ-к и имеет labels = 0
        sp = TN / (TN + FP)'''
    inp = []
    for i in range(len(labels)):
        inp.append([labels[i], scores[i]])
    sorted_inp = sorted(inp, key = lambda x: x[1], reverse=True)
    TN = 0 
    for i in range(k, len(labels)):
        if sorted_inp[i][0] == 0:
            TN += 1
    FP = 0
    for i in range(k):
        if sorted_inp[i][0] == 0:
            FP += 1
    return TN / (TN + FP + np.finfo(float).eps)


def f1_at_k(labels: List[int], scores: List[float], k=5) -> float:
    return 2  / (1 / precision_at_k(labels, scores, k) + 1 / recall_at_k(labels, scores, k) + np.finfo(float).eps)
