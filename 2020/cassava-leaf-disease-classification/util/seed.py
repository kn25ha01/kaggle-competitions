import os
import random
import numpy as np
import torch

#https://pytorch.org/docs/stable/notes/randomness.html

def set_seed(seed, logger=None):
    if logger:
        logger.debug(f'seed : {seed}')
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    #torch.cuda.manual_seed_all(seed) #不要
    torch.backends.cudnn.deterministic = True
    # Trueにすると2週目以降早くなる?が、再現性を確保できなくなる
    torch.backends.cudnn.benchmark = False

if __name__ == "__main__":
    # test
    from logging import getLogger, basicConfig
    logger = getLogger(__name__)
    basicConfig(level='DEBUG')
    seed = 20200219
    print(seed)
    set_seed(seed, logger)
