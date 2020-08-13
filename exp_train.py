"""
Runs a model on a single node across N-gpus.
"""
import os
from argparse import ArgumentParser
import pathlib

import numpy as np
import torch

import pytorch_lightning as pl
from lightning_models import LightningModel
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.profiler import SimpleProfiler, AdvancedProfiler

# from lr_logger import LearningRateLogger

from notifications import notify
from callbacks import CSVLogger
from lr_finder import LRFinder


def get_last_version(exp_path):
    exp_path = pathlib.Path(exp_path)

    exp_list = list(exp_path.glob('version*'))

    if len(exp_list) == 0:
        return None

    def sort_func(path):
        return int(path.name.split('version_')[-1])

    path = sorted(exp_list, key=sort_func)[-1]
    version = int(path.name.split('version_')[-1])

    return version


def get_last_epoch(ckpt_path):
    ckpt_path = pathlib.Path(ckpt_path)

    ckpt_list = list(ckpt_path.glob('*.ckpt'))

    if len(ckpt_list) == 0:
        return None

    def sort_func(path):
        return int(path.stem.split('=')[-1])

    path = sorted(ckpt_list, key=sort_func)[-1]
    epoch = int(path.stem.split('=')[-1])

    return epoch


def get_prof_index(prof_path):
    prof_path = pathlib.Path(prof_path)

    prof_list = list(prof_path.glob('*.log'))

    if len(prof_list) == 0:
        return 0

    def sort_func(path):
        return int(path.stem.split('_')[-1])

    path = sorted(prof_list, key=sort_func)[-1]
    index = int(path.stem.split('_')[-1]) + 1

    return index


def parse_range(value):
    value_list = value.strip('[]()').split(',')

    start = float(value_list[0])
    stop = float(value_list[1])
    num = int(value_list[2])

    return start, stop, num


def parse_list(value):
    import ast
    value_list = []
    for value in value.strip('[]()').split(','):
        try:
            val = ast.literal_eval(value)
        except ValueError:
            val = value
        value_list.append(val)

    return value_list


def main(hparams):
    """
    Main training routine specific for this project
    :param hparams:
    """

    # Set random seed before anything starts
    hparams.seed = pl.seed_everything(hparams.seed)

    save_dir = pathlib.Path(hparams.model_path) / hparams.exp_name
    exp_path = save_dir / hparams.arch
    exp_path.mkdir(parents=True, exist_ok=True)

    if hparams.cont is not None:
        if type(hparams.cont) == int:
            version = hparams.cont
        elif type(hparams.cont) == str:
            version = get_last_version(exp_path)

        if version is not None:
            ckpt_path = exp_path / f'version_{version}' / 'checkpoints'
            epoch = get_last_epoch(ckpt_path)
            resume_path = ckpt_path / f'epoch={epoch}.ckpt'
            hparams.last_epoch = epoch
    else:
        version = None

    if version is None:
        epoch = None
        resume_path = None
        hparams.last_epoch = -1

    # ------------------------
    # 1 INIT TRAINER
    # ------------------------

    # ---- Early Stopping ----

    if hparams.patience is None:
        hparams.patience = hparams.epochs

    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        min_delta=0.0,
        patience=hparams.patience,
        verbose=True,
        mode='min'
    )

    # # ----     Logger     ----

    logger = TensorBoardLogger(
        save_dir=save_dir,
        name=hparams.arch,
        version=None
    )
    actual_path = save_dir / logger.name / f'version_{logger.version}'

    # # ---- Optional Profiler ----

    if hparams.profiler is not None:
        prof_path = actual_path / 'profiles'
        prof_path.mkdir(parents=True, exist_ok=True)
        prof_index = get_prof_index(prof_path)
        prof_filename = prof_path / f'profile_{prof_index}.log'

        if hparams.profiler == 'simple' or hparams.profiler == '':
            profiler = SimpleProfiler(output_filename=prof_filename)
        elif hparams.profiler == 'advanced':
            profiler = AdvancedProfiler(output_filename=prof_filename)
        else:
            message = f'Wrong profiler choice [{hparams.profiler}]. \
                        Supported profilers include \
                        [simple (True/empty), advanced]'
            raise ValueError(message)
    else:
        profiler = False

    # ---- Model Checkpoint ----

    ckpt_path = actual_path / 'checkpoints'
    ckpt_path.mkdir(parents=True, exist_ok=True)

    checkpoint_callback = ModelCheckpoint(
        filepath=ckpt_path, monitor='val_loss',
        verbose=False, save_last=True,
        save_top_k=5, save_weights_only=False,
        mode='auto', period=1, prefix=''
    )

    # -------- Custom Callbacks --------

    csv_logger = CSVLogger()
    callback_list = [csv_logger]

    # -------- Trainer --------

    trainer = Trainer(
        weights_summary='full',
        weights_save_path=save_dir,
        gpus=hparams.gpus,
        max_epochs=hparams.epochs,
        precision=hparams.precision,
        fast_dev_run=hparams.debug,
        logger=logger,
        checkpoint_callback=checkpoint_callback,
        early_stop_callback=early_stop_callback,
        profiler=profiler,
        limit_train_batches=hparams.train_subset,
        limit_val_batches=hparams.val_subset,
        resume_from_checkpoint=resume_path,
        callbacks=callback_list,
        benchmark=True,  # optimized CUDA convolution algorithm
        progress_bar_refresh_rate=int(not hparams.silent),
    )

    # -----------------------------
    # 2 FIND INITIAL LEARNING RATE
    # -----------------------------

    if hparams.lr_finder or hparams.scheduler == 'clr':
        mode = 'exponential'
        finder = LRFinder(hparams, LightningModel,
                          num_epochs=hparams.lr_epochs, mode=mode,
                          min_lr=hparams.lr_min, max_lr=hparams.lr_max)
        finder.fit()

        # Set learning rate bounds
        res, _, _ = finder.suggestion()
        hparams.learning_rate = res.best_lr
        hparams.base_lr = res.best_lr
        hparams.max_lr = res.max_lr

        print('######### Learning Rate Finder #########')
        print(f"LR range = ({finder.min_lr:.3e}, {finder.max_lr:.3e})")
        print(f"Sugg. (Best LR = {res.best_lr:.3e})")
        print(f"Sugg. (Min. LR = {res.min_lr:.3e})")
        print(f"Sugg. (Max. LR = {res.max_lr:.3e})")
        print('######### Learning Rate Finder #########')

        # Save LR Finder results
        finder_path = actual_path / 'lr_finder'
        finder.save(finder_path)

        if hparams.lr_plot:
            # Plot
            import matplotlib.pyplot as plt
            finder.plot(save_path=finder_path)
            finder.plot_grad(save_path=finder_path)
            if hparams.lr_show_plot:
                plt.show()

        # Try to release memory
        del finder

    # ------------------------
    # 3 INIT LIGHTNING MODEL
    # ------------------------

    if version is None:
        model = LightningModel(hparams)
    else:
        model = LightningModel.load_from_checkpoint(resume_path)

    # ------------------------
    # 4 START TRAINING
    # ------------------------

    trainer.fit(model)

    return hparams


if __name__ == '__main__':
    # ------------------------
    # TRAINING ARGUMENTS
    # ------------------------
    # these are project-wide arguments

    root_dir = os.path.dirname(os.path.realpath(__file__))
    parent_parser = ArgumentParser(add_help=False)

    # Experiment params
    parent_parser.add_argument(
        '--model_path',
        dest='model_path',
        default='saved_models',
        help='Where the trained model will be saved \
              (Default: ./saved_models).'
    )
    parent_parser.add_argument(
        '--exp_name',
        dest='exp_name',
        default='resnet_cifar10',
        help='Experiment name (Default: resnet_cifar10).'
    )
    parent_parser.add_argument(
        '--seed',
        dest='seed',
        type=int,
        help='Random seed (Default: None).'
    )
    parent_parser.add_argument(
        '--continue_from',
        dest='cont',
        help='Continue training from a previous checkpoint. \
              Possible values: [last (str), a version number (int)].'
    )

    # GPU params
    parent_parser.add_argument(
        '--gpus', '-g',
        dest='gpus',
        type=int,
        default=0,
        help='Which (or how many) GPUs to train on. \
              Possible value types: [list, str, int].'
    )
    parent_parser.add_argument(
        '--precision',
        dest='precision',
        type=int,
        default=32,
        help='Choose float precision. Possible values: \
              [32 (default), 16] bits.'
    )

    # Boolean params
    parent_parser.add_argument(
        '--debug',
        dest='debug',
        action='store_true',
        help='Runs 1 batch of train, test and val to find any bugs \
             (ie: a sort of unit test).'
    )
    parent_parser.add_argument(
        '--profiler',
        dest='profiler',
        nargs='?', const='',
        help='Activate profiler. Possible values: \
              [None (default), simple, advanced].'
    )
    parent_parser.add_argument(
        '--silent',
        dest='silent',
        action='store_true',
        help='Silence the progress bar output. This is useful \
              when running on Google Colab, since it freezes \
              your web browser when too much information is printed.'
    )
    parent_parser.add_argument(
        '--notify',
        dest='telegram',
        action='store_true',
        help='Notify start and stop of training via Telegram. \
              A telegram.json file must be provided on this folder.'
    )

    # Dataset params
    parent_parser.add_argument(
        '--train_subset',
        dest='train_subset',
        type=float, default=1.0,
        help='Fraction of training dataset to use. \
              (Default: 1.0).'
    )
    parent_parser.add_argument(
        '--val_subset',
        dest='val_subset',
        type=float, default=1.0,
        help='Fraction of validation dataset to use. \
              (Default: 1.0).'
    )

    # Learning Rate Finder params
    parent_parser.add_argument(
        '--lr_finder',
        dest='lr_finder',
        action='store_true',
        help='Get initial learning rate via a LR Finder.'
    )
    parent_parser.add_argument(
        '--lr_min',
        dest='lr_min',
        type=float, default=1e-7,
        help='Minimum learning rate value for the LR Finder.'
    )
    parent_parser.add_argument(
        '--lr_max',
        dest='lr_max',
        type=float, default=1,
        help='Maximum learning rate value for the LR Finder.'
    )
    parent_parser.add_argument(
        '--lr_epochs',
        dest='lr_epochs',
        type=int, default=10,
        help='Number of epochs to run the with LR Finder. \
              (Default: 10).'
    )
    parent_parser.add_argument(
        '--lr_plot',
        dest='lr_plot',
        action='store_true',
        help='Plot LR Finder results.'
    )
    parent_parser.add_argument(
        '--lr_show_plot',
        dest='lr_show_plot',
        action='store_true',
        help='Show LR Finder results plot.'
    )

    # Grid search experiment params
    parent_parser.add_argument(
        '--param-name',
        dest='param_name',
        default='learning_rate',
        help="Name of the experiment's hyperparameter."
    )
    parent_parser.add_argument(
        '--param-range',
        dest='param_range',
        type=str,
        default='[0.001,0.15,5]',
        help='Hyperparameter linspace range in \
              format [start,stop,num_samples]. \
              Ignored if param_list is set.'
    )
    parent_parser.add_argument(
        '--param-list',
        dest='param_list',
        type=str,
        help='Hyperparameter value list (overides param-range).'
    )
    parent_parser.add_argument(
        '--param-index',
        dest='param_index',
        type=int,
        default=0,
        help='Hyperparameter initial index \
              (continue from a partial experiment).'
    )

    # each LightningModule defines arguments relevant to it
    parser = LightningModel.add_model_specific_args(parent_parser, root_dir)
    hyperparams = parser.parse_args()

    # ---------------------
    # RUN TRAINING
    # ---------------------

    # Helper function for the telegram notifier
    @notify()
    def call_main(hparams):
        return main(hparams)

    # Initialize experiment list
    if hyperparams.param_list is None:
        start, stop, num = parse_range(hyperparams.param_range)
        params = np.linspace(start, stop, num)
    else:
        params = parse_list(hyperparams.param_list)
        num = len(params)

    seed = hyperparams.seed

    start_index = hyperparams.param_index
    hyperdict = vars(hyperparams)
    for i in range(start_index, num):
        # Reset seed
        hyperparams.seed = seed

        # Change hyperparameter for the ith exp.
        hyperparams.param_index = i
        if hyperparams.param_name in hyperdict.keys():
            hyperdict[hyperparams.param_name] = params[i]

        # Run trainer
        if hyperparams.telegram:
            call_main(hyperparams)
        else:
            main(hyperparams)
