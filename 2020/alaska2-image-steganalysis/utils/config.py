import yaml
from easydict import EasyDict as edict


def _get_default_config():
    c = edict()

    # dataset
    c.data = edict()
    c.data.name = None
    c.data.cover_dir = None
    c.data.stego_dir = None
    c.data.params = edict()

    # model
    c.model = edict()
    c.model.name = None
    c.model.params = edict()

    # train
    c.train = edict()
    c.train.out_dir = './results/default_out'
    c.train.batch_size = None
    c.train.num_epochs = None
    c.train.num_grad_acc = None

    # evaluation
    c.eval = edict()
    c.eval.batch_size = None

    # optimizer
    c.optimizer = edict()
    c.optimizer.name = None
    c.optimizer.params = edict()

    # scheduler
    c.scheduler = edict()
    c.scheduler.name = None
    c.scheduler.params = edict()

    # transforms
    c.transform = edict()
    c.transform.name = None
    c.transform.num_preprocessor = 0
    c.transform.params = edict()

    # losses
    c.loss = edict()
    c.loss.name = None
    c.loss.params = edict()

    c.seed = 2020

    return c


# _get_default_config()にマージ
def _merge_config(src, dst):
    if not isinstance(src, edict):
        return

    for k, v in src.items():
        if isinstance(v, edict):
            _merge_config(src[k], dst[k])
        else:
            dst[k] = v


# コンフィグ読み込み
def load(config_path):
    with open(config_path, 'r') as fid:
        yaml_config = edict(yaml.load(fid))

    config = _get_default_config()
    _merge_config(yaml_config, config)

    return config
    
