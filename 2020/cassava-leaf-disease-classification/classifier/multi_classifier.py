import os
import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from util.checkpoint2 import save
from util.early_stopping import EarlyStopping
from util.history import History
from util.metrics import get_cate_acc, get_acc

class MultiClassifierModel():

    def __init__(self, workdir, model, logger=None):
        self.workdir = workdir
        self.logger = logger
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dtype = torch.float32
        self._model = model.to(self.device, dtype=self.dtype)

        if self.logger:
            self.logger.info(f'Using device {self.device}')


    def fit(self, train_dataset, valid_dataset, batch_size, epochs, loss_weight=None):
        if torch.cuda.device_count() > 1:
            self.logger.info('Train with multi GPUs.')
            self._model = torch.nn.DataParallel(self._model)

        self._epochs = epochs

        lr = 0.001 * 2
        self._optimizer = AdamW(self._model.parameters(), lr=lr)

        #t_max = self._epochs
        t_max = 20
        self._scheduler = CosineAnnealingLR(self._optimizer, T_max=t_max)
        self._criterion = CrossEntropyLoss(loss_weight)

        train_dataloader = DataLoader(
            train_dataset,
            shuffle=True,
            batch_size=batch_size,
            num_workers=2,
            pin_memory=True,
            drop_last=True
        )
        valid_dataloader = DataLoader(
            valid_dataset,
            shuffle=True,
            batch_size=batch_size,
            num_workers=2,
            pin_memory=True,
            drop_last=True
        )

        history = History(self.workdir)
        early_stopping = EarlyStopping(logger=self.logger)

        for epoch in range(self._epochs):
            train_log = self._train(train_dataloader, epoch)
            valid_log = self._valid(valid_dataloader, epoch)
            history.write(epoch, train_log, valid_log)
            self._save(epoch)

            if early_stopping(valid_log['loss']):
                break

        history.save()


    def _train(self, dataloader, epoch):
        self._model.train()
        tbar = tqdm(dataloader, total=len(dataloader))
        tbar.set_description(f'train {epoch + 1}/{self._epochs}')
        true_list = []
        pred_list = []
        loss_list = []

        for images, labels in tbar:
            images = images.to(self.device, dtype=self.dtype, non_blocking=True) # [batch_size, 3, , ]
            true_list.extend(labels.numpy())
            labels = labels.to(self.device, non_blocking=True) # [batch_size]

            self._optimizer.zero_grad()
            logits = self._model(images)
            pred_list.extend(torch.argmax(logits, dim=1).cpu().numpy())

            loss = self._criterion(logits, labels)
            loss_list.append(loss.item())
            loss.backward()

            self._optimizer.step()
        self._scheduler.step()

        log = {}
        log['loss'] = sum(loss_list) / len(loss_list)
        self._score(true_list, pred_list, log)
        log['lr'] = self._optimizer.param_groups[0]['lr']
        return log


    def _valid(self, dataloader, epoch):
        self._model.eval()
        tbar = tqdm(dataloader, total=len(dataloader))
        tbar.set_description(f'valid {epoch + 1}/{self._epochs}')
        true_list = []
        pred_list = []
        loss_list = []

        with torch.no_grad():
            for images, labels in tbar:
                images = images.to(self.device, dtype=self.dtype, non_blocking=True) # [batch_size, 3, , ]
                true_list.extend(labels.numpy())
                labels = labels.to(self.device, non_blocking=True) # [batch_size]

                logits = self._model(images)
                pred_list.extend(torch.argmax(logits, dim=1).cpu().numpy())

                loss = self._criterion(logits, labels)
                loss_list.append(loss.item())

        log = {}
        log['loss'] = sum(loss_list) / len(loss_list)
        self._score(true_list, pred_list, log)
        return log


    def predict(self, test_dataset, batch_size):
        test_dataloader = DataLoader(
            test_dataset,
            shuffle=False,
            batch_size=batch_size,
            num_workers=2,
            pin_memory=True
        )
        score = self._test(test_dataloader)
        return score


    def _test(self, dataloader):
        self._model.eval()
        tbar = tqdm(dataloader, total=len(dataloader))
        true_list = []
        pred_list = []

        with torch.no_grad():
            for images, labels in tbar:
                images = images.to(self.device, dtype=self.dtype, non_blocking=True) # [batch_size, 3, , ]
                true_list.extend(labels.numpy())
                labels = labels.to(self.device, non_blocking=True) # [batch_size]

                logits = self._model(images)
                pred_list.extend(torch.argmax(logits, dim=1).cpu().numpy())

        cate_acc, acc = self._score(true_list, pred_list)
        return cate_acc, acc


    def _score(self, true: list, pred: list, log=None):
        acc = get_acc(true, pred, self.logger)
        cate_acc = get_cate_acc(true, pred, self.logger)

        # train, valid
        if log:
            log['acc'] = acc
            for i in range(cate_acc.shape[0]):
                log[f'acc{i}'] = cate_acc[i]
        # test
        else:
            return cate_acc, acc


    # この関数はもう1つｎモデルでも使いたいので、checkpointに移していいと思う
    def _get_checkpoint(self, epoch: int):
        return {
            'epoch': epoch,
            'model': self._model.state_dict(),
            'gpus': torch.cuda.device_count(),
        }


    def _save(self, epoch: int):
        checkpoint = self._get_checkpoint(epoch)
        path = os.path.join(self.workdir, f'epoch_{epoch:03d}.pth')
        save(checkpoint, path, self.logger)
