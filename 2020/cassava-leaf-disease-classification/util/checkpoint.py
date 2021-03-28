import os
import torch


def get_last_checkpoint(workdir):
    cps = [cp for cp in os.listdir(workdir) if cp.startswith('epoch_') and cp.endswith('.pth')]
    if len(cps):
        return os.path.join(workdir, list(sorted(cps))[-1])
    return None


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


def load_checkpoint(checkpoint, model, optimizer=None, scheduler=None):
    print('load checkpoint from', checkpoint)
    checkpoint = torch.load(checkpoint)
    #return checkpoint

    model_dict = get_model_dict(checkpoint['model'])
    model.load_state_dict(model_dict) #, strict=False)

    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer'])

    if scheduler:
        scheduler.load_state_dict(checkpoint['scheduler'])

    last_epoch = checkpoint['epoch']

    return last_epoch
