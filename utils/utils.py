import numpy as np
import warnings
import os
from sklearn.preprocessing import minmax_scale, normalize
from scipy import sparse
import torch
import random
import logging
from logging import handlers
from torch.utils.data import Dataset
from xclib.evaluation.xc_metrics import precision, ndcg, recall, psprecision, psndcg, psrecall
from xclib.evaluation.xc_metrics import compute_inv_propesity
from scipy.sparse import csr_matrix


def set_seed(seed=0):
    random.seed(seed)  # Python random module.
    np.random.seed(seed)  # Numpy module.
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def evaluation_xmc(y_true, y_pred, inv_propesity):
    y_true = csr_matrix(y_true)
    y_pred = csr_matrix(y_pred)
    out_score = {}
    out_score['precision'] = precision(y_pred, y_true) * 100
    out_score['ndcg'] = ndcg(y_pred, y_true) * 100
    out_score['recall'] = recall(y_pred, y_true) * 100
    out_score['psprecision'] = psprecision(y_pred, y_true, inv_propesity) * 100
    out_score['psndcg'] = psndcg(y_pred, y_true, inv_propesity) * 100
    out_score['psrecall'] = psrecall(y_pred, y_true, inv_propesity) * 100
    return out_score


def get_inv_propesity(dataset_name, y_train):
    # y_train:[N,c], sparse matrix
    if 'Wikipedia' in dataset_name or 'WikiTitles' in dataset_name:
        A, B = 0.5, 0.4
    elif 'Amazon' in dataset_name:
        A, B = 0.6, 2.6
    else:
        A, B = 0.55, 1.5
    print(f'dataset_name: {dataset_name}, inv_propesity: A-{A}, B-{B}')
    inv_propesity = compute_inv_propesity(y_train, A, B)
    return inv_propesity
