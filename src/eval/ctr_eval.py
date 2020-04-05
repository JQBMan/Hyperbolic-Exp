'''AUC Evaluation'''
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score

########################
# ctr evaluation
########################
def ctr_eval(test_loader, device, model, graph):
    # auc_list, f1_list, recall_list = [], [], []
    # compute auc
    #TODO: test
    labels = []
    scores = []
    for _, [user, item, label, _] in enumerate(test_loader):
        u, i, l = user.to(device), item.to(device), label.to(device)
        # print(u, i, l)
        labels += l.tolist()
        scores += (model(u, i, graph)).tolist()
        # auc
        # auc_list.append(roc_auc_score(y_true=labels, y_score=scores))
        #
    auc = roc_auc_score(y_true=labels, y_score=scores)
    scores = np.array(scores)
    scores[scores >= 0.5] = 1
    scores[scores < 0.5] = 0
    # recall_list.append(recall_score(y_true=labels, y_pred=scores, average='micro', zero_division=0))
    recall = recall_score(y_true=labels, y_pred=scores, average='micro', zero_division=0)
    # f1_list.append(f1_score(y_true=labels, y_pred=scores, average='macro', zero_division=0))
    f1 = f1_score(y_true=labels, y_pred=scores, average='macro', zero_division=0)
        # print(float(np.mean(auc_list)), float(np.mean(f1_list)))
    return auc, recall, f1
