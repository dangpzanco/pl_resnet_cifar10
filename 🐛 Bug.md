## LR finder broken 2: not sure why (and other tiny bugs)

## 🐛 Bug

<!-- A clear and concise description of what the bug is. -->

LR finder doesn't seem to work. The model doesn't train when `trainer.lr_find(model)` is running (the loss metric oscillates around its initial value). When looking at the figure from `lr_finder.plot()`, I suspected the learning rate wasn't being changed somehow, but internally it does. So I rebuilt a custom LR finder to check if it wasn't the rest of my code. It seems `lr_find` is broken, but I'm not sure why, since the implementation is kinda complex for me to debug. People might get wrong results if they don't check `lr_finder.plot()` before using `lr_find.suggestion()`.

### To Reproduce

Steps to reproduce the behavior:

1. Clone this [test repository](https://github.com/dangpzanco/pl_resnet_cifar10)
2. Run the corresponding script (`run.bat` or `run.sh`)
3. Compare plot results for LR finder and a custom LR finder ([`lr_finder.png`](https://github.com/dangpzanco/pl_resnet_cifar10/blob/e1f45365a3c29c0e49ee8162d50be80c7985b467/lr_finder.png) and [`custom_lr_finder.png`](https://github.com/dangpzanco/pl_resnet_cifar10/blob/e1f45365a3c29c0e49ee8162d50be80c7985b467/custom_lr_finder.png))

<!-- If you have a code sample, error messages, stack traces, please provide it here as well -->


#### Code sample
<!-- Ideally attach a minimal code sample to reproduce the decried issue. 
Minimal means having the shortest code but still preserving the bug. -->

The sample code is available [on this repo](https://github.com/dangpzanco/pl_resnet_cifar10). It trains [ResNet-s on CIFAR10](https://github.com/akamaster/pytorch_resnet_cifar10) with 10% of the train/val set for 10 `epochs` with initial `learning_rate=1e-5` and `end_lr=1`.

Following is a stripped-down version of it:

```python
# -----------------------------
# 3 FIND INITIAL LEARNING RATE
# -----------------------------

# Run learning rate finder
lr_finder = trainer.lr_find(
    model,
    num_training=hparams.epochs*model.batches_per_epoch,
    min_lr=hparams.learning_rate,
    mode='exponential')

# Plot
import matplotlib.pyplot as plt
fig = lr_finder.plot(suggest=True)
fig.tight_layout()
fig.savefig('lr_finder.png', dpi=300, format='png')

# Pick point based on plot, or get suggestion
new_lr = lr_finder.suggestion()

# -------------------------------------
# 4 FIND INITIAL LEARNING RATE (CUSTOM)
# -------------------------------------

# the scheduler is already configured as a LR sweeper
trainer.fit(model)

# get metrics from a custom CSV logger callback
metrics = trainer.callbacks[1].batch_metrics
loss = metrics['loss'].values

# Same as lr_finder.suggestion(), but with a moving average filter
index, lrs, loss = lr_suggestion(metrics, model.batches_per_epoch)
custom_lr = lrs[index]

# Plot
import matplotlib.pyplot as plt
fig, ax = plt.subplots()
ax.plot(metrics['lr'], metrics['loss'], ':', label='Per Batch')
ax.plot(lrs, loss, label='Filtered ("Per Epoch")')
ax.plot(lrs[index], loss[index], 'ro', label='Suggestion')
ax.set_xscale('log')
ax.set_xlabel('Learning Rate')
ax.set_ylabel('Loss')
ax.legend()
fig.tight_layout()
fig.savefig('custom_lr_finder.png', dpi=300, format='png')

```

The "custom" learning rate finder is supposed to replicate `lr_finder`, it's just the same scheduler (`lr_finder._ExponentialLR`) and a custom CSV logger callback which logs `lr` collected from inside the training loop:

```python
def training_step(self, batch, batch_idx):
    # forward pass
    x, y = batch
    y_hat = self.forward(x)

    # calculate loss
    loss_val = self.loss(y_hat, y)

    # acc
    acc = ...

    # lr
    lr = self.trainer.lr_schedulers[0]['scheduler']._last_lr[0]

    tqdm_dict = {'acc': acc, 'lr': lr}
    output = OrderedDict({
        'loss': loss_val,
        'progress_bar': tqdm_dict,
        'log': tqdm_dict
    })

    # can also return just a scalar instead of a dict (return loss_val)
    return output

def configure_optimizers(self):
        optimizer = optim.SGD(self.parameters(),
                              self.hparams.learning_rate,
                              momentum=self.hparams.momentum,
                              weight_decay=self.hparams.weight_decay)

        customlr = _ExponentialLR
        # customlr = _LinearLR
        clr = customlr(
            optimizer,
            end_lr=1,
            num_iter=self.hparams.epochs*self.batches_per_epoch,
            last_epoch=-1
        )
        scheduler = dict(scheduler=clr,
                         interval='step')

        return [optimizer], [scheduler]
```

When calculating the learning rate suggestion, a moving average filter was applied (with size `batches_per_epoch`). This prevents amplifying the noise with `np.gradient()` and getting wrong results from a "lucky batch". `scipy.signal.filtfilt` is necessary to avoid a delay in the loss array. I removed the line with  `loss = loss[np.isfinite(loss)]` for simplicity (and because of a potential bug when `loss` contains `NaNs`).

```python
def lr_suggestion(metrics, filter_size=100, skip_begin=10, skip_end=1):
    loss = metrics['loss'].values
    lrs = metrics['lr'].values
	
    # if loss has any NaN values, lrs.size != loss.size,
    # which would result in the wrong index for lrs
    # this code assumes there's no NaNs in loss
    # loss = loss[np.isfinite(loss)]
    
    # Moving average before calculating the "gradient"
    from scipy import signal
    coef = np.ones(filter_size) / filter_size
    loss = signal.filtfilt(coef, 1, loss)

    index = np.gradient(loss[skip_begin:-skip_end]).argmin() + skip_begin

    return index, lrs, loss
```

### Expected behavior

<!-- A clear and concise description of what you expected to happen. -->

##### LR finder plot results (not expected):

![](https://github.com/dangpzanco/pl_resnet_cifar10/blob/e1f45365a3c29c0e49ee8162d50be80c7985b467/lr_finder.png?raw=true)

##### Custom LR finder (blue line is the expected behavior):![](https://github.com/dangpzanco/pl_resnet_cifar10/blob/e1f45365a3c29c0e49ee8162d50be80c7985b467/custom_lr_finder.png?raw=true)

### Environment

* CUDA:
        - GPU:
                - GeForce GTX 950M
        - available:         True
        - version:           10.2
* Packages:
        - numpy:             1.19.1
        - pyTorch_debug:     False
        - pyTorch_version:   1.6.0
        - pytorch-lightning: 0.8.5
        - tensorboard:       2.2.1
        - tqdm:              4.47.0
* System:
        - OS:                Windows
        - architecture:
                - 64bit
                - WindowsPE
        - processor:         Intel64 Family 6 Model 142 Stepping 9, GenuineIntel
        - python:            3.7.7
        - version:           10.0.19041

### Additional context

<!-- Add any other context about the problem here. -->

PS: When debugging this problem, I noticed that `LearningRateLogger` only supports `'steps'` and `'epoch'` as an interval, not logging the lr when `interval == 'batch'`. The [sample code](https://github.com/dangpzanco/pl_resnet_cifar10/blob/e1f45365a3c29c0e49ee8162d50be80c7985b467/lr_logger.py) has a simple fix which changes 2 lines of code ([L68](https://github.com/PyTorchLightning/pytorch-lightning/blob/0.8.5/pytorch_lightning/callbacks/lr_logger.py#L68) and [L82](https://github.com/PyTorchLightning/pytorch-lightning/blob/0.8.5/pytorch_lightning/callbacks/lr_logger.py#L82)) to `latest_stat = self._extract_lr(trainer, ['step', 'batch'])` and `if scheduler['interval'] in interval:`, respectively.

