import os
import random
import numpy as np
import torch


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


# モデルを保存するcheckpointフォルダを作成する。
def prepare_train_directories(config):
    os.makedirs(os.path.join(config.train.out_dir, 'checkpoint'), exist_ok=True)


