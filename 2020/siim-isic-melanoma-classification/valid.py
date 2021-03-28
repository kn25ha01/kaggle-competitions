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
#from torch.optim.lr_scheduler import ReduceLROnPlateau
#from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score

from dataset import get_dataloader
from model import get_model
import util
import util.config
import util.checkpoint
import util.metrics
import util.seed


def valid_single_epoch(config, out_dir, model, dataloader):
    model.eval()
    target_list = []
    pred_list = []
    tbar = tqdm.tqdm(dataloader, total=len(dataloader))
    tbar.set_description(f'Valid')

    with torch.no_grad():
        for inputs, targets in tbar:
            inputs[0] = inputs[0].to('cuda', dtype=torch.float32)
            inputs[1] = inputs[1].to('cuda', dtype=torch.float32)
            targets = targets.view(-1, 1).to('cuda', dtype=torch.float32)
            target_list.extend(targets.cpu().numpy())
            logits = model(inputs)
            pred = F.sigmoid(logits)
            pred_list.extend(pred.cpu().numpy())

    targets = np.array(target_list)
    preds = np.array(pred_list)

    print(roc_auc_score(targets.flatten(), preds.flatten()))

    df = pd.DataFrame(columns=['target', 'pred'])
    df['target'] = targets.flatten()
    df['pred'] = preds.flatten()
    df.to_csv(os.path.join(out_dir, 'valid.csv'))


def valid(config, out_dir, model, dataloaders):
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    model = model.to('cuda', dtype=torch.float32)

    valid_single_epoch(config, out_dir, model, dataloaders['valid'])


def run(config):
    # train data
    train_img_dir = config.data.dir + 'train/'
    train_df = pd.read_csv('train_new.csv')
    train_df['sex'] = train_df['sex'].map({'male': 1, 'female': 0})
    train_df['age'] /= 100.0

    skf = KFold(n_splits=config.train.num_folds, shuffle=True, random_state=config.seed)

    checkpoints = [config.valid.fold0_cp,
                   config.valid.fold1_cp,
                   config.valid.fold2_cp,
                   config.valid.fold3_cp,
                   config.valid.fold4_cp]

    for fold, (idxT, idxV) in enumerate(skf.split(np.arange(15))):
        print('=' * 20, 'Fold', fold, '=' * 20)
        
        # out dir
        out_dir = os.path.join(config.train.out_dir, f'fold{fold}')
        os.makedirs(os.path.join(out_dir), exist_ok=True)

        # model, optimizer
        model = get_model(config).to('cuda', dtype=torch.float32)
        optimizer = Adam(model.parameters())

        checkpoint = os.path.join(out_dir, checkpoints[fold])
        if checkpoint is not None:
            last_epoch, step = util.checkpoint.load_checkpoint(model, optimizer, checkpoint)
            print(f'from checkpoint: {checkpoint} last epoch:{last_epoch}')
            if last_epoch + 1 >= config.train.num_epochs:
                print(f'This fold has finished already. last epoch:{last_epoch} train epochs:{config.train.num_epochs}')
        else:
            last_epoch, step = -1, -1

        # data loader
        dataloaders = {'valid':get_dataloader(config,
                                              train_df.loc[train_df['fold'].isin(idxV)].reset_index(drop=True),
                                              train_img_dir,
                                              'valid',
                                              ['sex', 'age'])}

        # train
        valid(config, out_dir, model, dataloaders)


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


