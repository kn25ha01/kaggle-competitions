import numpy as np
from sklearn.metrics import accuracy_score
#from sklearn.metrics import f1_score


def get_cate_acc(true: list, pred: list, logger=None):
    if len(true) != len(pred):
        return 0.0

    labels = len(set(true))

    acc_n = np.zeros([labels])
    acc_d = np.zeros([labels])

    for i, t in enumerate(true):
        acc_d[t] += 1
        if t == pred[i]:
            acc_n[t] += 1

    cate_acc = acc_n / acc_d # shape [5]
    if logger:
        logger.debug(f'cate_acc : {cate_acc}')
    #acc = acc_n.sum() / acc_d.sum() # accuracy_score(true, pred)
    return cate_acc


def get_acc(true: list, pred: list, logger=None):
    acc = accuracy_score(true, pred)
    if logger:
        logger.debug(f'acc : {acc}')
    return acc


#def get_f1_score(true: list, pred: list):
#    return f1_score(true, pred, average='macro')


if __name__ == "__main__":
    true = [0,1,1,1,1,1]
    pred = [0,1,2,1,1,3]
    acc = get_acc_score(true, pred)
    print(acc)
    #f1 = get_f1_score(true, pred)
    #print(f1)
