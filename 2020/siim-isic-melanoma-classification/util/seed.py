import os
import random
import numpy as np
import torch


def seed_everything(seed, is_backends=True):
    '''
    https://pytorch.org/docs/stable/notes/randomness.html
    '''
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if is_backends:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

