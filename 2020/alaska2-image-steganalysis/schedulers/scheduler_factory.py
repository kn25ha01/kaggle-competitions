from torch.optim import lr_scheduler


def LambdaLR(optimizer, lr_lambda, last_epoch, **_):
    return lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda, last_epoch=last_epoch)


def get_scheduler(config, optimizer, last_epoch, lr_lambda=None):
    func = globals().get(config.scheduler.name)
    if lr_lambda:
        return func(optimizer, lr_lambda, last_epoch, **config.scheduler.params)
    else:
        return func(optimizer, last_epoch, **config.scheduler.params)


