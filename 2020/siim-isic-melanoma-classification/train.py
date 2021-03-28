import os
import argparse
import pprint
import tqdm

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score

from dataset import get_dataloader
from model import get_model
import util
import util.config
import util.checkpoint
import util.metrics
import util.seed


def valid_single_epoch(config, model, dataloader, epoch, writer):
    model.eval()
    loss_list = []
    target_list = []
    pred_list = []
    tbar = tqdm.tqdm(dataloader, total=len(dataloader))
    tbar.set_description(f'Valid epoch {epoch}')

    with torch.no_grad():
        for inputs, targets in tbar:
            inputs[0] = inputs[0].to('cuda', dtype=torch.float32)
            inputs[1] = inputs[1].to('cuda', dtype=torch.float32)
            targets = targets.view(-1, 1).to('cuda', dtype=torch.float32)
            target_list.extend(targets.cpu().numpy())
            logits = model(inputs)
            loss = nn.BCEWithLogitsLoss()(logits, targets)
            loss_list.append(loss.item())
            pred = F.sigmoid(logits)
            pred_list.extend(pred.cpu().numpy())

    targets = np.array(target_list)
    preds = np.array(pred_list)

    log_dict = {}
    log_dict['valid/loss'] = sum(loss_list) / len(loss_list)
    log_dict['valid/auc'] = roc_auc_score(targets.flatten(), preds.flatten())

    if writer is not None:
        for key, value in log_dict.items():
            writer.add_scalar('valid/{}'.format(key), value, epoch + 1)

    return log_dict


def train_single_epoch(config, model, dataloader, optimizer, epoch, writer):
    model.train()
    loss_list = []
    target_list = []
    pred_list = []
    tbar = tqdm.tqdm(dataloader, total=len(dataloader))
    tbar.set_description(f'Train epoch {epoch}')

    for inputs, targets in tbar:
        inputs[0] = inputs[0].to('cuda', dtype=torch.float32) # size=[batch_size, 3, 128, 128]
        inputs[1] = inputs[1].to('cuda', dtype=torch.float32) # size=[batch_size, 2]
        targets = targets.view(-1, 1).to('cuda', dtype=torch.float32) # size=[batch_size, 1]
        target_list.extend(targets.cpu().numpy())
        optimizer.zero_grad()
        logits = model(inputs)
        #loss = nn.BCEWithLogitsLoss(weight)(logits, targets)
        loss = nn.BCEWithLogitsLoss()(logits, targets)
        loss_list.append(loss.item())
        loss.backward()
        optimizer.step()
        pred = F.sigmoid(logits)
        pred_list.extend(pred.cpu().detach().numpy())

    targets = np.array(target_list)
    preds = np.array(pred_list)

    log_dict = {}
    log_dict['train/loss'] = sum(loss_list) / len(loss_list)
    log_dict['train/auc'] = roc_auc_score(targets.flatten(), preds.flatten())
    log_dict['train/lr'] = optimizer.param_groups[0]['lr']

    if writer is not None:
        for key, value in log_dict.items():
            writer.add_scalar('train/{}'.format(key), value, epoch+1)

    return log_dict


def train(config, out_dir, model, dataloaders, optimizer, fold, start_epoch):
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    model = model.to('cuda', dtype=torch.float32)
    #scheduler = ReduceLROnPlateau(optimizer)
    # 学習率は落ちるとき0.2倍される
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=1, verbose=True, factor=0.2)

    history = pd.DataFrame()
    writer = SummaryWriter(out_dir)

    for epoch in range(start_epoch, config.train.num_epochs):
        train_log = train_single_epoch(config, model, dataloaders['train'], optimizer, epoch, writer)
        valid_log = valid_single_epoch(config, model, dataloaders['valid'], epoch, writer)
        scheduler.step(valid_log['valid/auc'])
        scheduler.step(train_log['train/auc'])
        util.checkpoint.save_checkpoint(out_dir, model, optimizer, fold, epoch, 0)

        # history
        for key, value in train_log.items():
            history.loc[epoch, key] = value
        for key, value in valid_log.items():
            history.loc[epoch, key] = value

    history.to_csv(os.path.join(out_dir, 'history.csv'))
    writer.close()


def run(config):
    # train data
    train_img_dir = config.data.dir + 'train/'
    #train_df = pd.read_csv(config.data.dir + 'train.csv')
    train_df = pd.read_csv('train_new.csv')
    train_df['sex'] = train_df['sex'].map({'male': 1, 'female': 0})
    train_df['age'] /= 100.0
    #train_df = train_df[:10000]

    skf = KFold(n_splits=config.train.num_folds, shuffle=True, random_state=config.seed)

    for fold, (idxT, idxV) in enumerate(skf.split(np.arange(15))):
        # out dir
        out_dir = os.path.join(config.train.out_dir, f'fold{fold}')
        os.makedirs(os.path.join(out_dir), exist_ok=True)

        # model, optimizer
        model = get_model(config).to('cuda', dtype=torch.float32)
        optimizer = Adam(model.parameters())

        # 最終foldを探索するようにしたい。
        checkpoint = util.checkpoint.get_last_checkpoint(out_dir)
        if checkpoint is not None:
            last_epoch, step = util.checkpoint.load_checkpoint(model, optimizer, checkpoint)
            print(f'from checkpoint: {checkpoint} last epoch:{last_epoch}')
            if last_epoch + 1 >= config.train.num_epochs:
                print(f'This fold has finished already. last epoch:{last_epoch} train epochs:{config.train.num_epochs}')
        else:
            last_epoch, step = -1, -1

        print('=' * 20, 'Fold', fold, '=' * 20)
        
        # data loader
        dataloaders = {'train':get_dataloader(config,
                                              train_df.loc[train_df['fold'].isin(idxT)].reset_index(drop=True),
                                              train_img_dir,
                                              'train',
                                              ['sex', 'age']),

                       'valid':get_dataloader(config,
                                              train_df.loc[train_df['fold'].isin(idxV)].reset_index(drop=True),
                                              train_img_dir,
                                              'valid',
                                              ['sex', 'age'])}

        # train
        train(config, out_dir, model, dataloaders, optimizer, fold, last_epoch + 1)


def parse_args():
    parser = argparse.ArgumentParser(description='training')
    parser.add_argument('--config', dest='config_file',
                        help='configuration filename',
                        default=None, type=str)
    return parser.parse_args()


def main():
    import warnings
    warnings.filterwarnings('ignore')

    args = parse_args()
    if args.config_file is None:
        raise Exception('no configuration file')

    config = util.config.load(args.config_file)
    pprint.PrettyPrinter(indent=2).pprint(config)
    util.seed.seed_everything(config.seed)
    run(config)

    print('success!')


if __name__ == '__main__':
    main()


