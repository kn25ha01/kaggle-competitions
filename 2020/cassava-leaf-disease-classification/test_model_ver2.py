import os
import argparse
import pandas as pd

from classifier import MultiClassifierModel
from classifier import BinaryClassifierModel
from factory import get_dataset, get_model

from util.yml import get_config


def test_model_v1(args):
    workdir_base = args.workdir
    dataset_config = get_config(args.dataset_yml)
    model_config = get_config(args.model_yml)
    batch_size = args.batch_size

    df = pd.read_csv(dataset_config['train_csv'])
    classes = 5
    #df = df[:200]

    test_df = df
    test_dataset = get_dataset(test_df, dataset_config, 'test')

    last_checkpoint_path = get_last_checkpoint(workdir)
    mcModel = MultiClassifierModel(workdir, model_config, classes, epochs, last_checkpoint_path)
    mcModel.predict(train_dataset, valid_dataset, batch_size, loss_weight)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workdir', default='./output/work', type=str)
    parser.add_argument('--model_version', default=1, type=int)
    parser.add_argument('--dataset_yml', default='./yml/dataset/200x150.yml', type=str)
    parser.add_argument('--model_yml', default='./yml/model/efficientnet-b0.yml', type=str)
    parser.add_argument('--batch_size', default=64, type=int)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    if args.model_version == 1:
        test_model_v1(args)

    elif args.model_version == 2:
        test_model_v2(args)


if __name__ == '__main__':
    main()
