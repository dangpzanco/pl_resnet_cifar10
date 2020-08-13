python exp_train.py --gpus=1 --epochs=64 ^
--train_subset=0.1 --val_subset=0.1 ^
--scheduler=clr --learning_rate=1e-7 ^
--exp_name=resnet_cifar10 --arch=resnet20 ^
--param-name=epochs_up --param-list=[2,4,8] ^
--lr_finder --lr_epochs=10 --lr_plot --num_workers=0

