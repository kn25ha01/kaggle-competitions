import os
import torch


# 最終エポックのモデルのチェックポイントを返す。
def get_initial_checkpoint(config):
    checkpoint_dir = os.path.join(config.train.out_dir, 'checkpoint')
    checkpoints = [checkpoint
                   for checkpoint in os.listdir(checkpoint_dir)
                   if checkpoint.startswith('epoch_') and checkpoint.endswith('.pth')]
    if checkpoints:
        return os.path.join(checkpoint_dir, list(sorted(checkpoints))[-1])
    return None


# 詳細はわからないけど、チェックポイントを読み込む。
def load_checkpoint(model, optimizer, checkpoint):
    print('load checkpoint from', checkpoint)
    checkpoint = torch.load(checkpoint)

    checkpoint_dict = {}
    for k, v in checkpoint['state_dict'].items():
        if 'num_batches_tracked' in k:
            continue
        if k.startswith('module.'):
            if True:
                checkpoint_dict[k[7:]] = v
            else:
                checkpoint_dict['feature_extractor.' + k[7:]] = v
        else:
            if True:
                checkpoint_dict[k] = v
            else:
                checkpoint_dict['feature_extractor.' + k] = v

    model.load_state_dict(checkpoint_dict) #, strict=False)

    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_dict'])

    step = checkpoint['step'] if 'step' in checkpoint else -1
    last_epoch = checkpoint['epoch'] if 'epoch' in checkpoint else -1

    return last_epoch, step


# モデルのパラメータとオプティマイザのパラメータとエポック数を保存する
def save_checkpoint(config, model, optimizer, epoch, step, weights_dict=None, name=None):
    checkpoint_dir = os.path.join(config.train.out_dir, 'checkpoint')

    if name:
        checkpoint_path = os.path.join(checkpoint_dir, '{}.pth'.format(name))
    else:
        checkpoint_path = os.path.join(checkpoint_dir, 'epoch_{:04d}.pth'.format(epoch))

    if weights_dict is None:
        weights_dict = {
            'state_dict': model.state_dict(),
            'optimizer_dict' : optimizer.state_dict(),
            'epoch' : epoch,
            'step' : step,
        }
    torch.save(weights_dict, checkpoint_path)


