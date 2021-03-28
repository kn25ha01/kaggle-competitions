import math
import argparse
import pprint
import tqdm

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from datasets import get_dataloader, get_train_valid_data
#from transforms import get_transform
from models import get_model
from losses import get_loss
from optimizers import get_optimizer
from schedulers import get_scheduler
import utils
import utils.config
import utils.checkpoint
import utils.metrics


def valid_single_epoch(config, model, dataloader, criterion, epoch, writer):
    model.eval()
    probability_list = []
    label_list = []
    loss_list = []
    tbar = tqdm.tqdm(enumerate(dataloader), total=len(dataloader))
    tbar.set_description(f'valid epoch {epoch}')

    with torch.no_grad():
        for i, data in tbar:
            images = data['image'].to('cuda:0', dtype=torch.float32)
            labels = data['label'].view(-1, 1).to('cuda:0', dtype=torch.float32)
            logits = model(images)
            loss = criterion(logits, labels)
            probabilities = F.sigmoid(logits)

            loss_list.append(loss.item())
            probability_list.extend(probabilities.cpu().numpy())
            label_list.extend(labels.cpu().numpy())

    probabilities = np.array(probability_list)
    labels = np.array(label_list)

    log_dict = {}
    log_dict['valid/loss'] = sum(loss_list) / len(loss_list)
    log_dict['valid/wauc'] = utils.metrics.weighted_acu(labels, probabilities)

    if writer is not None:
        for key, value in log_dict.items():
            writer.add_scalar('valid/{}'.format(key), value, epoch+1)

    return log_dict


def train_single_epoch(config, model, dataloader, criterion, optimizer,
                       scheduler, epoch, writer):
    model.train()
    probability_list = []
    label_list = []
    loss_list = []
    tbar = tqdm.tqdm(enumerate(dataloader), total=len(dataloader))
    tbar.set_description(f'train epoch {epoch}')

    for i, data in tbar:
        images = data['image'].to('cuda:0', dtype=torch.float32)
        labels = data['label'].view(-1, 1).to('cuda:0', dtype=torch.float32)
        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        probabilities = F.sigmoid(logits)
        
        loss_list.append(loss.item())
        probability_list.extend(probabilities.detach().cpu().numpy())
        label_list.extend(labels.cpu().numpy())
        
        loss.backward()
        optimizer.step()
        scheduler.step()

    probabilities = np.array(probability_list)
    labels = np.array(label_list)

    log_dict = {}
    log_dict['train/loss'] = sum(loss_list) / len(loss_list)
    log_dict['train/wauc'] = utils.metrics.weighted_acu(labels, probabilities)
    log_dict['train/lr'] = optimizer.param_groups[0]['lr']

    if writer is not None:
        for key, value in log_dict.items():
            writer.add_scalar('train/{}'.format(key), value, epoch+1)

    return log_dict


def train(config, model, dataloaders, criterion, optimizer, scheduler, start_epoch):
    history = pd.DataFrame()
    writer = SummaryWriter(config.train.out_dir)

    for epoch in range(start_epoch, config.train.num_epochs):
        # train
        train_log = train_single_epoch(config, model, dataloaders['train'], criterion,
                                       optimizer, scheduler,
                                       epoch, writer)
        # validation
        valid_log = valid_single_epoch(config, model, dataloaders['valid'], criterion,
                                       epoch, writer)
        # save
        utils.checkpoint.save_checkpoint(config, model, optimizer, epoch, 0)

        # history
        for key, value in train_log.items():
            history.loc[epoch, key] = value
        for key, value in valid_log.items():
            history.loc[epoch, key] = value

    history.to_csv(config.train.out_dir + 'history.csv')
    writer.close()


def run(config):
    # model
    model = get_model(config).to('cuda:0', dtype=torch.float32)
    criterion = get_loss(config)
    optimizer = get_optimizer(config, model.parameters())
    checkpoint = utils.checkpoint.get_initial_checkpoint(config)
    if checkpoint is not None:
        last_epoch, step = utils.checkpoint.load_checkpoint(model, optimizer, checkpoint)
    else:
        last_epoch, step = -1, -1
    print(f'from checkpoint: {checkpoint} last epoch:{last_epoch}')

    # split
    x_train, x_valid, y_train, y_valid = get_train_valid_data(config.data.cover_dir,
                                                              config.data.stego_dir)

    #x_train = x_train[67500-72:67500+72]
    #x_valid = x_valid[ 7500-48: 7500+48]
    #y_train = y_train[67500-72:67500+72]
    #y_valid = y_valid[ 7500-48: 7500+48]

    # data loader
    dataloaders = {'train':get_dataloader(config, 'train', x_train, y_train),
                   'valid':get_dataloader(config, 'valid', x_valid, y_valid)}

    # scheduler
    ALL_TRAIN_STEPS = len(dataloaders['train']) * config.train.num_epochs
    WARM_UP_STEP = ALL_TRAIN_STEPS * 0.5

    def warmup_linear_decay(step):
        if step < WARM_UP_STEP:
            return step / WARM_UP_STEP
        else:
            return (ALL_TRAIN_STEPS - step) / (ALL_TRAIN_STEPS - WARM_UP_STEP)

    scheduler = get_scheduler(config, optimizer, last_epoch, warmup_linear_decay)

    # train
    train(config, model, dataloaders, criterion, optimizer, scheduler, last_epoch+1)


def parse_args():
    parser = argparse.ArgumentParser(description='ALASKA2')
    parser.add_argument('--config', dest='config_file',
                        help='configuration filename',
                        default=None, type=str)
    return parser.parse_args()


def main():
    import warnings
    warnings.filterwarnings('ignore')

    print('kaggle ALASKA2 Image Steganalysis.')
    args = parse_args()
    if args.config_file is None:
        raise Exception('no configuration file')

    config = utils.config.load(args.config_file)
    pprint.PrettyPrinter(indent=2).pprint(config)
    utils.prepare_train_directories(config)
    utils.seed_everything(config.seed)
    run(config)

    print('success!')


if __name__ == '__main__':
    main()


