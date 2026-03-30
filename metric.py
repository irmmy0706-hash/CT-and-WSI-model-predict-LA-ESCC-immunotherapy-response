import torch
from sklearn.metrics import f1_score, precision_score, recall_score,roc_auc_score


def acc_(y_label, y_predict):
    correct = torch.sum((y_label == y_predict).float())
    total = y_label.size(0)
    return correct / total

def auc_(y_true, y_pred):
    # y_true = y_true.cpu().numpy()
    y_pred = y_pred.detach().cpu().numpy()
    return roc_auc_score(y_true, y_pred, average='macro')


def precision_(y_true, y_pred):
    y_true = y_true.cpu().numpy()
    y_pred = y_pred.cpu().numpy()
    return precision_score(y_true, y_pred, average='macro')


def recall_(y_true, y_pred):
    y_true = y_true.cpu().numpy()
    y_pred = y_pred.cpu().numpy()
    return recall_score(y_true, y_pred, average='macro')


def f1_score_(y_true, y_pred):
    y_true = y_true.cpu().numpy()
    y_pred = y_pred.cpu().numpy()
    return f1_score(y_true, y_pred, average='macro')


