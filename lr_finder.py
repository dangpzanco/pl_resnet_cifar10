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
from collections import namedtuple

import yaml


class LRFinder():
    """Some Information about LRFinder"""

    def __init__(self, hparams, model_class, num_epochs=10,
                 min_lr=1e-8, max_lr=1, mode='exponential'):
        super(LRFinder, self).__init__()
        self.hparams = copy.deepcopy(hparams)
        self.model_class = model_class
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.mode = mode
        self.metrics = None
        self.results = None

        self.setup_hparams()
        self.setup_model()

        self.trainer = Trainer(
            weights_summary=None,
            default_root_dir=tempfile.gettempdir(),
            gpus=self.hparams.gpus,
            max_epochs=num_epochs,
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
        self.hparams.learning_rate = self.min_lr
        self.hparams.max_lr = self.max_lr
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

    def fit(self):
        self.trainer.fit(self.model)
        self.metrics = self.trainer.callbacks[0].batch_metrics

    def save(self, save_path=None):
        if self.metrics is None:
            raise Warning('Run .fit() first.')
            return

        if self.results is None:
            self.suggestion()

        if save_path is None:
            save_path = pathlib.Path('./lr_finder/')

        self.metrics.to_csv(save_path / 'lr_metrics.csv', header=True)
        with open(save_path / 'result.yml', 'w') as yaml_file:
            yaml.dump(self.results, yaml_file)

    def _filtered_metric(self, metric_name='loss', filter_size=None):
        if self.metrics is None:
            raise Warning('Run .fit() first.')
            return None

        if filter_size is None:
            filter_size = self.model.batches_per_epoch

        # Moving average
        metric = self.metrics[metric_name].astype(float).values
        coef = np.ones(filter_size) / filter_size
        metric = signal.filtfilt(coef, 1, metric)

        return metric

    def suggestion(self, metric_name='loss', mode='auto',
                   filter_size=None, skip_begin=10, skip_end=1):
        if self.metrics is None:
            raise Warning('Run .fit() first.')
            return None, None, None

        if mode == 'auto':
            if 'loss' in metric_name:
                mode = 'min'
            elif 'acc' in metric_name:
                mode = 'max'
            else:
                mode = 'min'

        if mode == 'min':
            check_op = np.argmin
            compare_op = np.less
        elif mode == 'max':
            check_op = np.argmax
            compare_op = np.greater

        metric = self._filtered_metric(metric_name, filter_size)
        lrs = self.metrics['lr'].astype(float).values
        grad = np.gradient(metric)[skip_begin:-skip_end]

        best_index = check_op(grad)
        min_index = check_op(compare_op(grad, 0)[:best_index+1][::-1])
        max_index = check_op(compare_op(grad, 0)[best_index:])

        best_index += skip_begin
        min_index = best_index - min_index
        max_index = best_index + max_index

        results = dict(
            best_index=int(best_index),
            best_lr=float(lrs[best_index]),
            min_index=int(min_index),
            min_lr=float(lrs[min_index]),
            max_index=int(max_index),
            max_lr=float(lrs[max_index])
        )
        self.results = results
        LRSuggestion = namedtuple('LRSuggestion', results)
        results = LRSuggestion(**results)

        return results, lrs, metric

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

        res, lrs, metric = self.suggestion(**suggestion_args)

        fig, ax = plt.subplots()
        ax.plot(lrs, self.metrics[metric_name], ':')
        ax.plot(lrs, metric)
        ax.plot(res.best_lr, metric[res.best_index], 'ro')
        ax.plot(res.min_lr, metric[res.min_index], 'go')
        ax.plot(res.max_lr, metric[res.max_index], 'bo')
        ax.set_xscale(scale)
        ax.set_xlabel('Learning Rate')
        ax.set_ylabel(metric_label)
        ax.grid(True)
        ax.legend(['Per Batch', 'Filtered',
                   f"Sugg. (Best LR = {res.best_lr:.3e})",
                   f"Sugg. (Min. LR = {res.min_lr:.3e})",
                   f"Sugg. (Max. LR = {res.max_lr:.3e})"])
        fig.tight_layout()
        if save_path is not None:
            save_path = pathlib.Path(save_path)
            save_path = save_path / f'lr_finder.{format}'
            fig.savefig(save_path, dpi=150, format=format)

        return fig, ax

    def plot_grad(self, metric_name='loss', suggestion_args=None,
                  save_path=None, format='png'):
        import matplotlib.pyplot as plt
        if self.metrics is None:
            raise Warning('Run .fit() first.')
            return None, None

        if suggestion_args is None:
            suggestion_args = dict(metric_name=metric_name)

        metric_label = metric_name.replace('_', ' ').title()
        scale = self.mode if self.mode == 'linear' else 'log'

        res, lrs, metric = self.suggestion(**suggestion_args)

        grad = np.gradient(self.metrics[metric_name])
        filt_grad = np.gradient(metric)

        fig, ax = plt.subplots()
        ax.plot(lrs, grad, ':')
        ax.plot(lrs, filt_grad)
        ax.plot(res.best_lr, filt_grad[res.best_index], 'ro')
        ax.plot(res.min_lr, filt_grad[res.min_index], 'go')
        ax.plot(res.max_lr, filt_grad[res.max_index], 'bo')
        ax.set_xscale(scale)
        ax.set_xlabel('Learning Rate')
        ax.set_ylabel(f'Grad. of {metric_label}')
        ax.grid(True)
        ax.legend(['Per Batch', 'Filtered',
                   f"Sugg. (Best LR = {res.best_lr:.3e})",
                   f"Sugg. (Min. LR = {res.min_lr:.3e})",
                   f"Sugg. (Max. LR = {res.max_lr:.3e})"])
        fig.tight_layout()
        if save_path is not None:
            save_path = pathlib.Path(save_path)
            save_path = save_path / f'lr_finder_grad.{format}'
            fig.savefig(save_path, dpi=150, format=format)

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
