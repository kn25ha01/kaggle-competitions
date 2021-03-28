import numpy as np
import sklearn.metrics
import torch


def macro_recall(pred_y, y, n_grapheme=168, n_vowel=11, n_consonant=7):
    pred_y = torch.split(pred_y, [n_grapheme, n_vowel, n_consonant], dim=1)
    pred_labels = [torch.argmax(py, dim=1).cpu().numpy() for py in pred_y]

    y = y.cpu().numpy()
    # pred_y = [p.cpu().numpy() for p in pred_y]

    recall_grapheme = sklearn.metrics.recall_score(pred_labels[0], y[:, 0], average='macro')
    recall_vowel = sklearn.metrics.recall_score(pred_labels[1], y[:, 1], average='macro')
    recall_consonant = sklearn.metrics.recall_score(pred_labels[2], y[:, 2], average='macro')
    final_score = np.average([recall_grapheme, recall_vowel, recall_consonant], weights=[2, 1, 1])
    return final_score


def macro_recall_multi(pred_graphemes, true_graphemes, pred_vowels, true_vowels, pred_consonants, true_consonants, n_grapheme=168, n_vowel=11, n_consonant=7):
    pred_label_graphemes = torch.argmax(pred_graphemes, dim=1).cpu().numpy()
    true_label_graphemes = true_graphemes.cpu().numpy()

    pred_label_vowels = torch.argmax(pred_vowels, dim=1).cpu().numpy()
    true_label_vowels = true_vowels.cpu().numpy()

    pred_label_consonants = torch.argmax(pred_consonants, dim=1).cpu().numpy()
    true_label_consonants = true_consonants.cpu().numpy()    

    recall_grapheme = sklearn.metrics.recall_score(pred_label_graphemes, true_label_graphemes, average='macro')
    recall_vowel = sklearn.metrics.recall_score(pred_label_vowels, true_label_vowels, average='macro')
    recall_consonant = sklearn.metrics.recall_score(pred_label_consonants, true_label_consonants, average='macro')
    final_score = np.average([recall_grapheme, recall_vowel, recall_consonant], weights=[2, 1, 1])
    return final_score


