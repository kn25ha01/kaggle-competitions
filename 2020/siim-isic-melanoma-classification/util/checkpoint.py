# BSD 2-Clause License
# 
# Copyright (c) 2019, pudae
# All rights reserved.
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# 
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
# 
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
# 
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# ==============================================================================

import os
import shutil
import torch


# 対象ディレクトリにある最終チェックポイントを取得する。
def get_last_checkpoint(checkpoint_dir):
    checkpoints = [checkpoint
                   for checkpoint in os.listdir(checkpoint_dir)
                   if checkpoint.startswith('epoch_') and checkpoint.endswith('.pth')]
    if checkpoints:
        return os.path.join(checkpoint_dir, list(sorted(checkpoints))[-1])
    return None
# 最終foldのチェックポイントを取れるように変更
#def get_last_checkpoint(checkpoint_dirs):
#    checkpoints = []
#    for checkpoint_dir in checkpoint_dirs:
#        checkpoints += [os.path.join(checkpoint_dir, checkpoint)
#                       for checkpoint in os.listdir(checkpoint_dir)
#                       if checkpoint.startswith('epoch_') and checkpoint.endswith('.pth')]
#    if checkpoints:
#        return sorted(checkpoints)[-1]
#    return None


# 初期チェックポイントを取得する。
def get_initial_checkpoint(config):
    #checkpoint_dir = os.path.join(config.train.out_dir, 'checkpoint')
    #return get_last_checkpoint(checkpoint_dir)
    #checkpoint_dirs = [os.path.join(config.train.out_dir, f'fold{fold}') for fold in range(config.train.num_folds)]
    #return get_last_checkpoint(checkpoint_dirs)
    return get_last_checkpoint(checkpoint_dir)


# 指定したチェックポイントのパスを取得する。
# 古いまま
def get_checkpoint(config, name):
    checkpoint_dir = os.path.join(config.train.out_dir, 'checkpoint')
    return os.path.join(checkpoint_dir, name)


# 最後からn個のチェックポイントをコピーする。
# 古いまま
def copy_last_n_checkpoints(config, n, name):
    checkpoint_dir = os.path.join(config.train.out_dir, 'checkpoint')
    checkpoints = [checkpoint
                   for checkpoint in os.listdir(checkpoint_dir)
                   if checkpoint.startswith('epoch_') and checkpoint.endswith('.pth')]
    checkpoints = sorted(checkpoints)
    for i, checkpoint in enumerate(checkpoints[-n:]):
        shutil.copyfile(os.path.join(checkpoint_dir, checkpoint),
                        os.path.join(checkpoint_dir, name.format(i)))


# チェックポイントを読み込む。
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
    #last_fold = checkpoint['fold'] if 'fold' in checkpoint else -1

    #return last_fold, last_epoch, step
    return last_epoch, step


# チェックポイントを保存する。
def save_checkpoint(out_dir, model, optimizer, fold, epoch, step, weights_dict=None):
    checkpoint_path = os.path.join(out_dir, 'epoch_{:04d}.pth'.format(epoch))
    if weights_dict is None:
        weights_dict = {
            'state_dict': model.state_dict(),
            'optimizer_dict' : optimizer.state_dict(),
            'fold' : fold,
            'epoch' : epoch,
            'step' : step,
        }
    torch.save(weights_dict, checkpoint_path)


