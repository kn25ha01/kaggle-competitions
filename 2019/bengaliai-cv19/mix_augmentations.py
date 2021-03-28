import numpy as np
import torch
from torch import nn
from sklearn.metrics import recall_score


# reference : 
# https://www.kaggle.com/kaushal2896/pytorch-bengali-graphemes-densenet121


def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


def cutmix(data, targets1, targets2, targets3, alpha):
    indices = torch.randperm(data.size(0))
    shuffled_data = data[indices]
    shuffled_targets1 = targets1[indices]
    shuffled_targets2 = targets2[indices]
    shuffled_targets3 = targets3[indices]

    lam = np.random.beta(alpha, alpha)
    bbx1, bby1, bbx2, bby2 = rand_bbox(data.size(), lam)
    #data[:, :, bbx1:bbx2, bby1:bby2] = data[indices, :, bbx1:bbx2, bby1:bby2]
    data[:, :, bbx1:bbx2, bby1:bby2] = shuffled_data[indices, :, bbx1:bbx2, bby1:bby2]
    # adjust lambda to exactly match pixel ratio
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (data.size()[-1] * data.size()[-2]))

    #targets = [targets1, shuffled_targets1, targets2, shuffled_targets2, targets3, shuffled_targets3, lam]

    #return data, targets
    return data, shuffled_targets1, shuffled_targets2, shuffled_targets3, lam


def cutmix_criterion(preds1,preds2,preds3, targets):
    targets1, targets2, targets3, targets4, targets5, targets6, lam = targets[0], targets[1], targets[2], targets[3], targets[4], targets[5], targets[6]
    criterion = nn.CrossEntropyLoss(reduction='mean')
    return 0.8 * (lam * criterion(preds1, targets1) + (1 - lam) * criterion(preds1, targets2)) + 0.1 * (lam * criterion(preds2, targets3) + (1 - lam) * criterion(preds2, targets4)) + 0.1 * (lam * criterion(preds3, targets5) + (1 - lam) * criterion(preds3, targets6))


def mixup(data, targets1, targets2, targets3, alpha):
    indices = torch.randperm(data.size(0))
    shuffled_data = data[indices]
    shuffled_targets1 = targets1[indices]
    shuffled_targets2 = targets2[indices]
    shuffled_targets3 = targets3[indices]

    lam = np.random.beta(alpha, alpha)
    data = data * lam + shuffled_data * (1 - lam)
    #targets = [targets1, shuffled_targets1, targets2, shuffled_targets2, targets3, shuffled_targets3, lam]

    #return data, targets
    return data, shuffled_targets1, shuffled_targets2, shuffled_targets3, lam


def mixup_criterion(preds1, preds2, preds3, targets):
    targets1, targets2, targets3, targets4, targets5, targets6, lam = targets[0], targets[1], targets[2], targets[3], targets[4], targets[5], targets[6]
    criterion = nn.CrossEntropyLoss(reduction='mean')
    return 0.8 * (lam * criterion(preds1, targets1) + (1 - lam) * criterion(preds1, targets2)) + 0.1 * (lam * criterion(preds2, targets3) + (1 - lam) * criterion(preds2, targets4)) + 0.1 * (lam * criterion(preds3, targets5) + (1 - lam) * criterion(preds3, targets6))


def mix_recall_multi(preds1, targets1, shuffled_targets1, preds2, targets2, shuffled_targets2, preds3, targets3, shuffled_targets3, lam):
    preds1_label = torch.argmax(preds1, dim=1).cpu().numpy()
    preds2_label = torch.argmax(preds2, dim=1).cpu().numpy()
    preds3_label = torch.argmax(preds3, dim=1).cpu().numpy()
    targets1_label = targets1.cpu().numpy()
    targets2_label = targets2.cpu().numpy()
    targets3_label = targets3.cpu().numpy()
    sh_targets1_label = shuffled_targets1.cpu().numpy()
    sh_targets2_label = shuffled_targets2.cpu().numpy()
    sh_targets3_label = shuffled_targets3.cpu().numpy()

    #recall1 = lam * recall_score(preds1_label, targets1_label, average='macro') + (1 - lam) * recall_score(preds1_label, sh_targets1_label, average='macro')
    #recall2 = lam * recall_score(preds2_label, targets2_label, average='macro') + (1 - lam) * recall_score(preds2_label, sh_targets2_label, average='macro')
    #recall3 = lam * recall_score(preds3_label, targets3_label, average='macro') + (1 - lam) * recall_score(preds3_label, sh_targets3_label, average='macro')
    recall1 = lam * recall_score(targets1_label, preds1_label, average='macro') + (1 - lam) * recall_score(sh_targets1_label, preds1_label, average='macro')
    recall2 = lam * recall_score(targets2_label, preds2_label, average='macro') + (1 - lam) * recall_score(sh_targets2_label, preds2_label, average='macro')
    recall3 = lam * recall_score(targets3_label, preds3_label, average='macro') + (1 - lam) * recall_score(sh_targets3_label, preds3_label, average='macro')
    scores = [recall1, recall2, recall3]
    final_score = np.average(scores, weights=[2, 1, 1])
    return final_score

