import copy
import pathlib
import tempfile
from typing import Sequence

import numpy as np
from scipy import signal

import torch
from torch.optim.lr_scheduler import _LRScheduler
from pytorch_lightning import Trainer

from callbacks import PandasLogger


class LRFinder():
    """Some Information about LRFinder"""

    def __init__(self, hparams, model_class, end_lr=1, mode='exponential'):
        super(LRFinder, self).__init__()
        self.hparams = copy.deepcopy(hparams)
        self.model_class = model_class
        self.end_lr = end_lr
        self.mode = mode
        self.metrics = None

        self.setup_hparams()
        self.setup_model()
        # self.setup_optimizers()

        self.trainer = Trainer(
            weights_summary=None,
            default_root_dir=tempfile.gettempdir(),
            gpus=self.hparams.gpus,
            max_epochs=self.hparams.lr_epochs,
            precision=self.hparams.precision,
            logger=False,
            checkpoint_callback=False,
            early_stop_callback=False,
            limit_train_batches=self.hparams.train_subset,
            limit_val_batches=self.hparams.val_subset,
            callbacks=[PandasLogger()],
            benchmark=True,  # optimized CUDA convolution algorithm
            progress_bar_refresh_rate=int(not self.hparams.silent),
        )

    def setup_hparams(self):
        self.hparams.end_lr = self.end_lr
        if self.mode == 'linear':
            self.hparams.scheduler = 'sweep_lin'
        elif self.mode == 'exponential':
            self.hparams.scheduler = 'sweep_exp'
        else:
            m = f'Option mode={self.mode} not supported.\
                  Possible values: [linear, exponential].'
            raise Exception(m)

    def setup_model(self):
        self.model = self.model_class(self.hparams)

        # def log_lr_decorator(training_step):
        #     def wrapper(*args):
        #         output = training_step(*args)
        #         lr = self.trainer.lr_schedulers[0]['scheduler']._last_lr[0]
        #         output['log']['lr'] = lr
        #         return output
        #     return wrapper
        # self.model.training_step = log_lr_decorator(self.model.training_step)

    # def setup_optimizers(self):
    #     optimizers, _ = self.model.configure_optimizers()

    #     if len(optimizers) != 1:
    #         m = f'`model.configure_optimizers()` returned {len(optimizers)}, \
    #                but learning rate finder only works with single optimizer'
    #         raise Exception(m)
    #     else:
    #         optimizer = optimizers[0]

    #     args = dict(
    #         optimizer=optimizer, end_lr=self.end_lr,
    #         num_iter=self.hparams.lr_epochs*self.model.batches_per_epoch
    #     )

    #     scheduler = dict(scheduler=_ExponentialLR(**args), interval='step')

    #     def configure_optimizers():
    #         return [optimizer], [scheduler]

    #     self.model.configure_optimizers = configure_optimizers

    def fit(self):
        self.trainer.fit(self.model)
        self.metrics = self.trainer.callbacks[0].batch_metrics

    def suggestion(self, metric_name='loss', mode='auto',
                   filter_size=None, skip_begin=10, skip_end=1):
        if self.metrics is None:
            raise Warning('Run .fit() first.')
            return None, None, None

        if filter_size is None:
            filter_size = self.model.batches_per_epoch

        if mode == 'auto':
            if 'loss' in metric_name:
                mode = 'min'
            elif 'acc' in metric_name:
                mode = 'max'
            else:
                mode = 'min'

        if mode == 'min':
            check_op = np.argmin
        elif mode == 'max':
            check_op = np.argmin

        metric = self.metrics[metric_name].astype(float)
        lrs = self.metrics['lr'].astype(float)

        # Moving average before calculating the "gradient"
        coef = np.ones(filter_size) / filter_size
        metric = signal.filtfilt(coef, 1, metric)

        index = np.gradient(metric[skip_begin:-skip_end])
        index = check_op(index) + skip_begin

        return index, lrs, metric

    def plot(self, metric_name='loss', suggestion_args=None,
             save_path=None, format='png'):
        import matplotlib.pyplot as plt
        if self.metrics is None:
            raise Warning('Run .fit() first.')
            return None, None

        if suggestion_args is None:
            suggestion_args = dict(metric_name=metric_name)

        metric_label = metric_name.replace('_', ' ').title()
        scale = self.mode if self.mode == 'linear' else 'log'

        index, lrs, metric = self.suggestion(**suggestion_args)

        fig, ax = plt.subplots()
        ax.plot(self.metrics['lr'], self.metrics[metric_name], ':')
        ax.plot(lrs, metric)
        ax.plot(lrs[index], metric[index], 'ro')
        ax.set_xscale(scale)
        ax.set_xlabel('Learning Rate')
        ax.set_ylabel(metric_label)
        ax.legend(['Per Batch', 'Filtered', 'LR Suggestion'])
        fig.tight_layout()
        if save_path is not None:
            save_path = pathlib.Path(save_path)
            fig.savefig(save_path.with_suffix(f'.{format}'),
                        dpi=150, format=format)

        return fig, ax


"""
The following code is available at
https://github.com/PyTorchLightning/pytorch-lightning/blob/0.8.5/pytorch_lightning/trainer/lr_finder.py
"""


class _LinearLR(_LRScheduler):
    """Linearly increases the learning rate between two boundaries
    over a number of iterations.
    Arguments:

        optimizer: wrapped optimizer.

        end_lr: the final learning rate.

        num_iter: the number of iterations over which the test occurs.

        last_epoch: the index of last epoch. Default: -1.
    """
    last_epoch: int
    base_lrs: Sequence

    def __init__(self,
                 optimizer: torch.optim.Optimizer,
                 end_lr: float,
                 num_iter: int,
                 last_epoch: int = -1):
        self.end_lr = end_lr
        self.num_iter = num_iter
        super(_LinearLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        curr_iter = self.last_epoch + 1
        r = curr_iter / self.num_iter

        if self.last_epoch > 0:
            val = [base_lr + r * (self.end_lr - base_lr)
                   for base_lr in self.base_lrs]
        else:
            val = [base_lr for base_lr in self.base_lrs]
        self._lr = val
        return val

    @property
    def lr(self):
        return self._lr


class _ExponentialLR(_LRScheduler):
    """Exponentially increases the learning rate between two boundaries
    over a number of iterations.

    Arguments:

        optimizer: wrapped optimizer.

        end_lr: the final learning rate.

        num_iter: the number of iterations over which the test occurs.

        last_epoch: the index of last epoch. Default: -1.
    """
    last_epoch: int
    base_lrs: Sequence

    def __init__(self,
                 optimizer: torch.optim.Optimizer,
                 end_lr: float,
                 num_iter: int,
                 last_epoch: int = -1):
        self.end_lr = end_lr
        self.num_iter = num_iter
        super(_ExponentialLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        curr_iter = self.last_epoch + 1
        r = curr_iter / self.num_iter

        if self.last_epoch > 0:
            val = [base_lr * (self.end_lr / base_lr) **
                   r for base_lr in self.base_lrs]
        else:
            val = [base_lr for base_lr in self.base_lrs]
        self._lr = val
        return val

    @property
    def lr(self):
        return self._lr
