import numpy as np

#https://github.com/Bjarten/early-stopping-pytorch

class EarlyStopping:
    def __init__(self, patience=10, delta=0, logger=None):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_loss = None
        self._logger = logger

    def __call__(self, valid_loss):
        if self.best_loss is None:
            self.best_loss = valid_loss
            return False

        elif valid_loss <= self.best_loss + self.delta:
            self.best_loss = valid_loss
            self.counter = 0
            return False

        else:
            self.counter += 1
            if self._logger:
                self._logger.debug(f'Early stopping count {self.counter}')
            if self.counter >= self.patience:
                if self._logger:
                    self._logger.debug('Early stopped.')
                return True
            return False

if __name__ == "__main__":
    # test
    from logging import getLogger, basicConfig
    logger = getLogger(__name__)
    basicConfig(level='DEBUG')

    early_stopping = EarlyStopping(logger=logger)

    losses = [i for i in range(20)]
    for loss in losses:
        print(loss)
        if early_stopping(loss):
            break
