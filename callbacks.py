import torch
import pathlib
import pandas as pd

import pytorch_lightning as pl

from datetime import datetime
from collections import OrderedDict


class CSVLogger(pl.Callback):
    """Custom metric logger and model checkpoint."""

    def __init__(self, output_path=None):
        super(CSVLogger, self).__init__()
        self._epoch = None

        if output_path is None:
            self.logger_path = None
        else:
            self.logger_path = pathlib.Path(output_path)
            self.logger_path.mkdir(parents=True, exist_ok=True)

    def metrics(self, interval):
        if interval == 'epoch':
            return self.epoch_metrics
        elif interval in ['step', 'batch']:
            return self.batch_metrics

    @property
    def batch_metrics(self):
        metrics_path = self.logger_path / 'metrics_batch.csv'
        return pd.read_csv(metrics_path)

    @property
    def epoch_metrics(self):
        metrics_path = self.logger_path / 'metrics_epoch.csv'
        return pd.read_csv(metrics_path)

    def _extract_metrics(self, trainer, interval):
        metrics = trainer.callback_metrics

        metric_keys = list(metrics.keys())

        data_dict = OrderedDict()
        if interval == 'epoch':
            metric_keys.remove('epoch')
            data_dict['epoch'] = metrics['epoch']
            data_dict['time'] = str(datetime.now())
        elif interval in ['step', 'batch']:
            remove_list = ['train', 'val', 'epoch']
            for m in metrics.keys():
                if any(sub in m for sub in remove_list):
                    metric_keys.remove(m)
            data_dict[interval] = trainer.global_step

        for k in metric_keys:
            if isinstance(metrics[k], dict):
                for j in metrics[k].keys():
                    data_dict[j] = metrics[k][j]
            else:
                data_dict[k] = metrics[k]

        # cleanup
        for k in data_dict.keys():
            try:
                data_dict[k] = float(data_dict[k].cpu())
            except Exception:
                pass

        return data_dict

    def _log_csv(self, trainer, metrics_path, interval):
        data_dict = self._extract_metrics(trainer, interval)

        new_metrics = pd.DataFrame.from_records([data_dict], index=interval)
        if metrics_path.exists():
            config = dict(header=False, mode='a')
            old_metrics = self.metrics(interval).set_index(interval)
            if not new_metrics.columns.equals(old_metrics.columns):
                new_metrics = pd.concat([old_metrics, new_metrics])
                config = dict(header=True, mode='w')
        else:
            config = dict(header=True, mode='w')

        new_metrics.to_csv(metrics_path, **config)

    def on_init_start(self, trainer):
        """Called when the trainer initialization begins, model has not yet been set."""
        pass

    def on_init_end(self, trainer):
        """Called when the trainer initialization ends, model has not yet been set."""
        if self.logger_path is None:
            checkpoint_path = trainer.checkpoint_callback.dirpath
            # checkpoint_path = trainer.logger.log_dir
            self.logger_path = checkpoint_path.parent / 'logging'
        self.logger_path.mkdir(parents=True, exist_ok=True)

    def on_batch_end(self, trainer, pl_module):
        """Called when the training batch ends."""
        if trainer.global_step > 1:
            metrics_path = self.logger_path / 'metrics_batch.csv'
            self._log_csv(trainer, metrics_path, interval='batch')

    def on_epoch_end(self, trainer, pl_module):
        """Called when the epoch ends."""
        metrics_path = self.logger_path / 'metrics_epoch.csv'
        self._log_csv(trainer, metrics_path, interval='epoch')

    def on_sanity_check_start(self, trainer, pl_module):
        """Called when the validation sanity check starts."""
        pass

    def on_sanity_check_end(self, trainer, pl_module):
        """Called when the validation sanity check ends."""
        pass

    def on_epoch_start(self, trainer, pl_module):
        """Called when the epoch begins."""
        pass

    def on_batch_start(self, trainer, pl_module):
        """Called when the training batch begins."""
        pass

    def on_validation_batch_start(self, trainer, pl_module):
        """Called when the validation batch begins."""
        pass

    def on_validation_batch_end(self, trainer, pl_module):
        """Called when the validation batch ends."""
        pass

    def on_test_batch_start(self, trainer, pl_module):
        """Called when the test batch begins."""
        pass

    def on_test_batch_end(self, trainer, pl_module):
        """Called when the test batch ends."""
        pass

    def on_train_start(self, trainer, pl_module):
        """Called when the train begins."""
        pass

    def on_train_end(self, trainer, pl_module):
        """Called when the train ends."""
        pass

    def on_validation_start(self, trainer, pl_module):
        """Called when the validation loop begins."""
        pass

    def on_validation_end(self, trainer, pl_module):
        """Called when the validation loop ends."""
        pass

    def on_test_start(self, trainer, pl_module):
        """Called when the test begins."""
        pass

    def on_test_end(self, trainer, pl_module):
        """Called when the test ends."""
        pass


class PandasLogger(pl.Callback):
    """PandasLogger metric logger and model checkpoint."""

    def __init__(self, save_path=None):
        super(PandasLogger, self).__init__()

        self.batch_metrics = pd.DataFrame()
        self.epoch_metrics = pd.DataFrame()
        self._epoch = 0

    def _extract_metrics(self, trainer, interval):
        metrics = trainer.callback_metrics
        metric_keys = list(metrics.keys())
        data_dict = OrderedDict()

        # setup required metrics depending on interval
        if interval == 'epoch':
            if interval in metric_keys:
                metric_keys.remove('epoch')
                data_dict['epoch'] = metrics['epoch']
            else:
                data_dict['epoch'] = self._epoch
            data_dict['time'] = str(datetime.now())
            self._epoch += 1
        elif interval in ['step', 'batch']:
            remove_list = ['train', 'val', 'epoch']
            for m in metrics.keys():
                if any(sub in m for sub in remove_list):
                    metric_keys.remove(m)
            data_dict[interval] = trainer.global_step

        # populate ordered dictionary
        for k in metric_keys:
            if isinstance(metrics[k], dict):
                continue
            else:
                data_dict[k] = float(metrics[k])

        # dataframe with a single row (one interval)
        metrics = pd.DataFrame.from_records([data_dict], index=interval)

        return metrics

    def on_batch_end(self, trainer, pl_module):
        """Called when the training batch ends."""
        if trainer.global_step > 0:
            new_metrics = self._extract_metrics(trainer, 'batch')
            self.batch_metrics = pd.concat([self.batch_metrics, new_metrics])

    def on_epoch_end(self, trainer, pl_module):
        """Called when the epoch ends."""
        new_metrics = self._extract_metrics(trainer, 'epoch')
        self.epoch_metrics = pd.concat([self.epoch_metrics, new_metrics])

    def on_train_end(self, trainer, pl_module):
        """Called when the train ends."""
        pass
