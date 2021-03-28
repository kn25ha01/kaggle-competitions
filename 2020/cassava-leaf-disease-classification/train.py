import os
import re
import argparse
import datetime

import pandas as pd
from sklearn.model_selection import StratifiedKFold

from classifier import MultiClassifierModel
from classifier import BinaryClassifierModel
from factory import get_dataset, get_model

from util.seed import set_seed
from util.yml import get_config

from logging import getLogger, DEBUG, INFO, StreamHandler, FileHandler, Formatter
logger = getLogger(__name__)


# いずれはvalidation dataなしで学習させるよー
# fold0フォルダ邪魔

def train_model01_1(args, model_config, dataset_config, train_config, logger=None):
    workdir_base = args.workdir
    df = pd.read_csv(dataset_config['train_csv'])
    #df = df[:200]
    skf = StratifiedKFold(n_splits=train_config['folds'], shuffle=True, random_state=args.seed)

    for fold, (train_index, valid_index) in enumerate(skf.split(df, df['label'])):
        print('=' * 20, f'fold {fold}', '=' * 20)
        workdir = os.path.join(workdir_base, f'fold{fold}')
        workdir = workdir_base
        os.makedirs(workdir, exist_ok=True)

        train_df = df.loc[train_index].reset_index(drop=True)
        valid_df = df.loc[valid_index].reset_index(drop=True)

        train_dataset = get_dataset(train_df, dataset_config, 'train', logger)
        valid_dataset = get_dataset(valid_df, dataset_config, 'valid', logger)

        loss_weight = None
        #if self.loss_weight:
        #    loss_weight = train_df['label'].value_counts().reset_index().sort_values('index')['label'].to_numpy()
        #    logging.debug(f'value counts : {loss_weight}')
        #    loss_weight = loss_weight.min() / loss_weight
        #    logging.debug(f'loss weight : {loss_weight}') # [1.0 0.49628784 0.45521215 0.08254963 0.42163998]
        #    loss_weight = torch.Tensor(loss_weight).to(self.device, dtype=self.dtype)

        model = get_model(model_config, logger)
        mcModel = MultiClassifierModel(workdir, model, logger)
        mcModel.fit(train_dataset, valid_dataset, train_config['batch_size'], train_config['epochs'], loss_weight)

        # only fold0
        break


def train_model02_1(args, model_config, dataset_config, train_config, logger=None):
    workdir_base = args.workdir
    df = pd.read_csv(dataset_config['train_csv'])
    #df = df[:200]
    skf = StratifiedKFold(n_splits=train_config['folds'], shuffle=True, random_state=args.seed)

    for fold, (train_index, valid_index) in enumerate(skf.split(df, df['label'])):
        print('=' * 20, f'fold {fold}', '=' * 20)
        workdir = os.path.join(workdir_base, f'fold{fold}')
        workdir = workdir_base
        os.makedirs(workdir, exist_ok=True)

        train_df = df.loc[train_index].reset_index(drop=True)
        train_df['label'] = train_df['label'].replace([1,2,4,3], [0,0,0,1])

        valid_df = df.loc[valid_index].reset_index(drop=True)
        valid_df['label'] = valid_df['label'].replace([1,2,4,3], [0,0,0,1])

        train_dataset = get_dataset(train_df, dataset_config, 'train', logger)
        valid_dataset = get_dataset(valid_df, dataset_config, 'valid', logger)

        loss_weight = train_df['label'].value_counts().reset_index().sort_values('index')['label'].to_numpy()
        loss_weight = loss_weight.min() / loss_weight
        if logger:
            logger.debug(f'loss weight : {loss_weight}')

        model = get_model(model_config, logger)
        bcModel = BinaryClassifierModel(workdir, model, logger)
        bcModel.fit(train_dataset, valid_dataset, train_config['batch_size'], train_config['epochs'])

        # only fold0
        break


def train_model02_2(args, model_config, dataset_config, train_config, logger=None):
    workdir_base = args.workdir
    df = pd.read_csv(dataset_config['train_csv'])
    #df = df[:200]
    skf = StratifiedKFold(n_splits=train_config['folds'], shuffle=True, random_state=args.seed)

    for fold, (train_index, valid_index) in enumerate(skf.split(df, df['label'])):
        print('=' * 20, f'fold {fold}', '=' * 20)
        workdir = os.path.join(workdir_base, f'fold{fold}')
        workdir = workdir_base
        os.makedirs(workdir, exist_ok=True)

        train_df = df.loc[train_index].reset_index(drop=True)
        train_df = train_df[train_df['label']!=3].reset_index(drop=True)
        train_df['label'] = train_df['label'].replace(4, 3)

        valid_df = df.loc[valid_index].reset_index(drop=True)
        valid_df = valid_df[valid_df['label']!=3].reset_index(drop=True)
        valid_df['label'] = valid_df['label'].replace(4, 3)

        train_dataset = get_dataset(train_df, dataset_config, 'train', logger)
        valid_dataset = get_dataset(valid_df, dataset_config, 'valid', logger)

        loss_weight = None
        model = get_model(model_config, logger)
        mcModel = MultiClassifierModel(workdir, model, logger)
        mcModel.fit(train_dataset, valid_dataset, train_config['batch_size'], train_config['epochs'], loss_weight)

        # only fold0
        break


def main(args):
    # make work directory
    split = re.split('[/.]', args.yml)
    args.workdir = os.path.join(args.workdir, split[3])
    args.workdir = os.path.join(args.workdir, split[4])
    args.workdir = os.path.join(args.workdir, get_date())
    os.makedirs(args.workdir, exist_ok=True)

    # set logger
    set_logger(args.workdir, args.level)

    # set seed
    set_seed(args.seed, logger)

    # get config
    yml = os.path.join(args.yml)
    config = get_config(yml, logger)
    model_config = config['model']
    dataset_config = config['dataset']
    train_config = config['train']

    # train
    train_model = globals().get('train_' + split[3])
    train_model(args, model_config, dataset_config, train_config, logger)


def get_date():
    timezone = datetime.timezone(datetime.timedelta(hours=9))
    return datetime.datetime.now(timezone).strftime('%Y%m%d_%H%M%S')


def date_converter(*args):
    timezone = datetime.timezone(datetime.timedelta(hours=9))
    return datetime.datetime.now(timezone).timetuple()


def set_logger(workdir, level):
    # make file handler
    filename = os.path.join(workdir, 'logger.log')
    f_formatter = Formatter('%(asctime)s - %(levelname)s - %(filename)s - %(funcName)s - %(message)s')
    f_handler = FileHandler(filename)
    f_handler.setLevel(DEBUG)
    f_handler.setFormatter(f_formatter)

    # make stream handler
    s_formatter = Formatter('%(asctime)s - %(levelname)s - %(filename)s - %(funcName)s - %(message)s')
    s_formatter.converter = date_converter
    s_handler = StreamHandler()
    s_handler.setLevel(level)
    s_handler.setFormatter(s_formatter)

    # set logger
    logger.setLevel(level)
    logger.addHandler(f_handler)
    logger.addHandler(s_handler)

    # log
    logger.info(f'work dir : {workdir}')
    logger.info(f'log file : {filename}')
    logger.info(f'log level : {level}')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=20200219, type=int)
    parser.add_argument('--workdir', default='./output', type=str)
    parser.add_argument('--level', default='DEBUG', type=str)
    parser.add_argument('--yml', default='', type=str)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    main(args)
