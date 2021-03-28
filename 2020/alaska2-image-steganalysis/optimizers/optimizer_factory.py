import torch


def AdamW(params, lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01, amsgrad=False, **_):
    if isinstance(betas, str):
        betas = eval(betas)
    return torch.optim.AdamW(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, amsgrad=amsgrad)


def get_optimizer(config, parameters):
    f = globals().get(config.optimizer.name)
    return f(parameters, **config.optimizer.params)


