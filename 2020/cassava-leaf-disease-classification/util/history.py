import os
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

def savefig1(data1, label1, path):
    plt.figure()
    plt.plot(data1, label=label1)
    plt.grid()
    plt.legend()
    plt.savefig(path)

def savefig2(data1, data2, label1, label2, path):
    plt.figure()
    plt.plot(data1, label=label1)
    plt.plot(data2, label=label2)
    plt.grid()
    plt.legend()
    plt.savefig(path)

class History:
    def __init__(self, workdir: str, logger=None, cate=False):
        self._workdir = workdir
        self._history = pd.DataFrame()
        self._writer = SummaryWriter(self._workdir)
        self._logger = logger
        self._write_list = ['loss', 'acc']
        if cate:
            self._savefig_list1 = ['loss', 'acc', 'acc0', 'acc1', 'acc2', 'acc3', 'acc4']
        else:
            self._savefig_list1 = ['loss', 'acc']
        self._savefig_list2 = ['lr']

    def write(self, epoch: int, train_log: dict, valid_log: dict):
        # train history
        for key, value in train_log.items():
            self._history.loc[epoch, f'{key}/train'] = value

        # valid history
        for key, value in valid_log.items():
            self._history.loc[epoch, f'{key}/valid'] = value

        # tensorboard
        for key in self._write_list:
            self._writer.add_scalar(f'{key}/train', train_log[f'{key}'], epoch)
            self._writer.add_scalar(f'{key}/valid', valid_log[f'{key}'], epoch)
        self._writer.add_scalar(f'lr/train', train_log['lr'], epoch)

    def save(self):
        # hostory
        path = os.path.join(self._workdir, 'history.csv')
        self._history.to_csv(path)
        if self._logger:
            self._logger.info(f'save {path}')

        # tensorboard
        self._writer.close()

        # figure
        self._savefig()

    def _savefig(self):
        for key in self._savefig_list1:
            data1 = self._history[f'{key}/train']
            data2 = self._history[f'{key}/valid']
            label1 = f'{key}/train'
            label2 = f'{key}/valid'
            path = os.path.join(self._workdir, f'{key}.png')
            savefig2(data1, data2, label1, label2, path)
            if self._logger:
                self._logger.info(f'save {path}')

        for key in self._savefig_list2:
            data1 = self._history[f'{key}/train']
            label1 = key
            path = os.path.join(self._workdir, f'{key}.png')
            savefig1(data1, label1, path)
            if self._logger:
                self._logger.info(f'save {path}')
