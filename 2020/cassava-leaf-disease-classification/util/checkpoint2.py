import os
import torch

def save(checkpoint: dict, path: str, logger=None):
    torch.save(checkpoint, path)
    if logger:
        logger.debug(f'save {path}')

def get_model_dict(checkpoint):
    model_dict = {}

    for k, v in checkpoint.items():
        if 'num_batches_tracked' in k:
            continue
        if k.startswith('module.'):
            if True:
                model_dict[k[7:]] = v
            else:
                model_dict['feature_extractor.' + k[7:]] = v
        else:
            if True:
                model_dict[k] = v
            else:
                model_dict['feature_extractor.' + k] = v

    return model_dict

def load(model, path: str, logger=None):
    checkpoint = torch.load(path)
    model.load_state_dict(get_model_dict(checkpoint['model']))
