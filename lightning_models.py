"""
Example template for defining a system
"""
import pathlib
import logging
from argparse import ArgumentParser
from collections import OrderedDict

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader

import torchvision.transforms as transforms
import torchvision.datasets as torchsets

import pytorch_lightning as pl
import pytorch_lightning.metrics as plm

from resnet_cifar10 import resnet
from lr_finder import _LinearLR, _ExponentialLR


class LightningModel(pl.LightningModule):
    """
    Sample model to show how to define a template
    """

    def __init__(self, hparams):
        """
        Pass in parsed HyperOptArgumentParser to the model
        :param hparams:
        """
        # init superclass
        super(LightningModel, self).__init__()
        self.hparams = hparams

        self.batch_size = hparams.batch_size

        # if you specify an example input,
        # the summary will show input/output for each layer
        self.input_shape = (3, 32, 32)
        self.example_input_array = torch.rand(1, *self.input_shape)

        # Define loss
        self.loss = nn.CrossEntropyLoss()

        # Define metric (has bugs on pl 0.8.5)
        # self.accuracy = plm.Accuracy(num_classes=10)

        # Setup dataset
        self.prepare_data()

        # Initialize model
        self.setup()

    # ---------------------
    # MODEL SETUP
    # ---------------------

    def setup(self, stage=None):
        resnet_module = getattr(resnet, self.hparams.arch)
        self.conv_net = resnet_module()

    # ---------------------
    # TRAINING
    # ---------------------
    def forward(self, x):
        """
        No special modification required for lightning,
        define as you normally would

        :param x:
        :return:
        """

        logits = self.conv_net(x)

        return logits

    def training_step(self, batch, batch_idx):
        """
        Lightning calls this inside the training loop
        :param batch:
        :return:
        """
        # forward pass
        x, y = batch
        y_hat = self.forward(x)

        # calculate loss
        loss_val = self.loss(y_hat, y)

        # acc
        labels_hat = torch.argmax(y_hat, dim=1)
        acc = torch.sum(y == labels_hat).item() / y.shape[0]
        acc = torch.tensor(acc)
        # acc = self.accuracy(y_hat, y)

        # lr
        lr = self.trainer.lr_schedulers[0]['scheduler']._last_lr[0]

        tqdm_dict = {'acc': acc, 'lr': lr}
        # tqdm_dict = {'acc': acc}
        output = OrderedDict({
            'loss': loss_val,
            'progress_bar': tqdm_dict,
            'log': tqdm_dict
        })

        # can also return just a scalar instead of a dict (return loss_val)
        return output

    def validation_step(self, batch, batch_idx):
        """
        Lightning calls this inside the validation loop
        :param batch:
        :return:
        """
        # forward pass
        x, y = batch
        y_hat = self.forward(x)

        # calculate loss
        loss_val = self.loss(y_hat, y)

        # acc
        labels_hat = torch.argmax(y_hat, dim=1)
        val_acc = torch.sum(y == labels_hat).item() / y.shape[0]
        val_acc = torch.tensor(val_acc)
        # val_acc = self.accuracy(y_hat, y)

        if self.on_gpu:
            val_acc = val_acc.cuda(loss_val.device.index)

        output = OrderedDict({
            'val_loss': loss_val,
            'val_acc': val_acc,
        })

        # can also return just a scalar instead of a dict (return loss_val)
        return output

    # Validation is test
    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def training_epoch_end(self, outputs):
        """
        Called at the end of the training epoch
        with the outputs of all training steps.

        Args:
            outputs: List of outputs you defined in :meth:`training_step`, or
                if there are multiple dataloaders, a list containing a list of
                outputs for each dataloader.

        Return:
            Dict or OrderedDict.
            May contain the following optional keys:

            - log (metrics to be added to the logger; only tensors)
            - progress_bar (dict for progress bar display)
            - any metric used in a callback (e.g. early stopping).
        """
        loss_mean = 0
        acc_mean = 0
        for output in outputs:
            loss = output['loss']
            acc = output['log']['acc']

            loss_mean += loss
            acc_mean += acc
        loss_mean /= len(outputs)
        acc_mean /= len(outputs)

        tqdm_dict = {'train_loss': loss_mean, 'train_acc': acc_mean}
        result = {'progress_bar': tqdm_dict,
                  'log': tqdm_dict, 'train_loss': loss_mean}
        return result

    def validation_epoch_end(self, outputs):
        """
        Called at the end of validation to aggregate outputs
        :param outputs: list of individual outputs of each validation step
        :return:
        """
        # if returned a scalar from validation_step,
        # outputs is a list of tensor scalars
        # we return just the average in this case (if we want)
        # return torch.stack(outputs).mean()

        val_loss_mean = 0
        val_acc_mean = 0
        for output in outputs:
            val_loss = output['val_loss']
            val_acc = output['val_acc']

            val_loss_mean += val_loss
            val_acc_mean += val_acc
        val_loss_mean /= len(outputs)
        val_acc_mean /= len(outputs)

        # Progress bar metrics
        tqdm_dict = {'val_loss': val_loss_mean, 'val_acc': val_acc_mean}

        # Logger metrics
        log_dict = {}
        log_dict.update(tqdm_dict)

        result = {'progress_bar': tqdm_dict,
                  'log': log_dict,
                  'val_loss': val_loss_mean}
        return result

    # ---------------------
    # TRAINING SETUP
    # ---------------------
    def configure_optimizers(self):
        """
        return whatever optimizers we want here
        :return: list of optimizers
        """

        if self.hparams.optimizer == 'sgd':
            optimizer = optim.SGD(self.parameters(),
                                  self.hparams.learning_rate,
                                  momentum=self.hparams.momentum,
                                  weight_decay=self.hparams.weight_decay)
        elif self.hparams.optimizer == 'adam':
            optimizer = optim.Adam(self.parameters(),
                                   weight_decay=self.hparams.weight_decay)

        if self.hparams.scheduler == 'clr':
            last_iteration = self.hparams.last_epoch
            if self.hparams.last_epoch >= 0:
                last_iteration *= self.batches_per_epoch
            clr = optim.lr_scheduler.CyclicLR(
                optimizer,
                base_lr=self.hparams.learning_rate,
                max_lr=100*self.hparams.learning_rate,
                step_size_up=4*self.batches_per_epoch,
                # mode='triangular',
                mode='triangular2',
                # mode='exp_range',
                cycle_momentum=True,
                base_momentum=0.8,
                max_momentum=0.9,
                last_epoch=last_iteration
            )
            scheduler = dict(scheduler=clr,
                             interval='step')
        elif self.hparams.scheduler == 'sweep_lin':
            sweep = _LinearLR(
                optimizer, end_lr=self.hparams.end_lr,
                num_iter=self.hparams.lr_epochs*self.batches_per_epoch)
            scheduler = dict(scheduler=sweep,
                             interval='step')
        elif self.hparams.scheduler == 'sweep_exp':
            sweep = _ExponentialLR(
                optimizer, end_lr=self.hparams.end_lr,
                num_iter=self.hparams.lr_epochs*self.batches_per_epoch)
            scheduler = dict(scheduler=sweep,
                             interval='step')
        else:
            scheduler = optim.lr_scheduler.MultiStepLR(
                optimizer, milestones=[100, 150], gamma=0.1,
                last_epoch=self.hparams.last_epoch)

        return [optimizer], [scheduler]

    @property
    def norm_factors(self):
        return self._norm

    @property
    def batches_per_epoch(self):
        try:
            value = self.hparams.train_subset
            value *= len(self.train_dataloader())
            return int(value)
        except Exception:
            self.prepare_data()
            return len(self.train_dataloader())

    def __dataloader(self, train):

        # Define normalization transform
        normalize = transforms.Normalize(**self.norm_factors)

        # Define transforms for train/valid/test set
        if train:
            transform = transforms.Compose([
                transforms.RandomHorizontalFlip(0.5),
                transforms.RandomCrop(32, padding=4),
                transforms.ToTensor(),
                normalize,
            ])
            batch_size = self.hparams.batch_size
        else:
            transform = transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ])
            batch_size = 128

        # Initialize dataset
        dataset = torchsets.CIFAR10(
            root=self.hparams.data_root, train=train,
            transform=transform)

        # Create data loader
        loader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=train,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
        )

        return loader

    def prepare_data(self):
        # Dowload train dataset
        dataset = torchsets.CIFAR10(root=self.hparams.data_root,
                                    train=True, download=True)

        # Dowload test dataset
        torchsets.CIFAR10(root=self.hparams.data_root,
                          train=False, download=True)

        # Compute normalization factors
        self._norm = dict(mean=None, std=None)
        self._norm['mean'] = (dataset.data/255).mean(axis=(0, 1, 2))
        self._norm['std'] = (dataset.data/255).std(axis=(0, 1, 2))

    def train_dataloader(self):
        logging.info('training data loader called')
        return self.__dataloader(train=True)

    def val_dataloader(self):
        logging.info('val data loader called')
        return self.__dataloader(train=False)

    def test_dataloader(self):
        logging.info('test data loader called')
        return self.__dataloader(train=False)

    @staticmethod
    def add_model_specific_args(parent_parser, root_dir):  # pragma: no cover
        """
        Parameters you define here will be available
        to your model through self.hparams

        :param parent_parser:
        :param root_dir:
        :return:
        """
        parser = ArgumentParser(parents=[parent_parser])

        # param overwrites
        parser.set_defaults(gradient_clip_val=5.0)

        # Network params
        parser.add_argument('--arch', type=str,
                            default='resnet20',
                            help='Network architecture. Possible values: \
                                  [resnet20, resnet32, resnet44, \
                                   resnet56, resnet110, resnet1202].')

        # Dataset params
        data_root = str(pathlib.Path(root_dir) / 'dataset')
        parser.add_argument('--data_root', default=data_root, type=str,
                            help='Path to the dataset \
                            (default: [root_dir]/dataset).')
        parser.add_argument('--num_workers', default=0, type=int,
                            help='How many subprocesses to use for data loading. \
                                  0 means that the data will be loaded in \
                                  the main process (default: 0).')

        # Training params (opt)
        parser.add_argument('--batch_size', default=128, type=int,
                            help='Batch size (default: 128).')
        parser.add_argument('--epochs', default=200, type=int,
                            help='Total number of epochs to run \
                                  (default: 200).')
        parser.add_argument('--patience', default=None, type=int,
                            help='Early stopping patience in number of epochs \
                                  (default: epochs).')
        parser.add_argument('--last_epoch', default=-1, type=int,
                            help='Index of the last trained epoch \
                                  (default: -1).')

        # Optimizer params
        parser.add_argument('--optimizer', default='sgd', type=str,
                            help='Optimization algorithm \
                                  [sgd (default), adam].')
        parser.add_argument('--scheduler', default='default', type=str,
                            help='Scheduler profile. \
                                Possible values: [default, clr].')
        parser.add_argument('--learning_rate', default=0.1, type=float,
                            help='Initial learning rate (default: 0.1).')
        parser.add_argument('--momentum', default=0.9, type=float,
                            help='Momentum (default: 0.9).')
        parser.add_argument('--weight_decay', default=1e-4, type=float,
                            help='Weight decay (default: 1e-4).')

        return parser


if __name__ == "__main__":
    pass
