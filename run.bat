python exp_train.py --gpus=1 --epochs=10 ^
--train_subset=0.1 --val_subset=0.1 ^
--scheduler=clr --learning_rate=1e-4 ^
--exp_name=resnet_cifar10 --param-name=arch ^
--param-list=[resnet20] ^
--lr_finder --lr_epochs=10 --lr_plot --silent
REM --param-list=[resnet20,resnet32,resnet44,resnet56,resnet110,resnet1202]
