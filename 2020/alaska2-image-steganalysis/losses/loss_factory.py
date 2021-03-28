import torch


def binary_cross_entropy_loss(**_):
    #print('loss: binary cross entropy loss')
    return torch.nn.BCELoss()


def binary_cross_entropy_with_logits_loss(**_):
    #print('loss: binary cross entropy with logits loss')
    return torch.nn.BCEWithLogitsLoss()


def get_loss(config):
    f = globals().get(config.loss.name)
    return f(**config.loss.params)


