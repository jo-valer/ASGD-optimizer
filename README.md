# Averaged Stochastic Gradient Descent

I found that PyTorch ASGD behaves differently than SGD before the averaging starting point, even though it is not supposed to. In particular SGD converges more rapidly (see image below). For this reason, here is an implementation of ASGD that behaves like SGD before the averaging. The averaging technique has been proposed by [Polyak and Juditsky, 1992](https://dl.acm.org/doi/10.1137/0330046).

In ASGD, the point at which to start averaging is a hyper-parameter. For this reason I also implemented the Non-monotonically Triggered ASGD proposed by [Merity et al., 2017](https://arxiv.org/abs/1708.02182). This method triggers the averaging when the validation loss does not decrease for a certain number of epochs, avoiding the need to set a fixed averaging starting point.

**Pytorch ASGD (green) vs SGD (purple):**
<br><img src="https://github.com/jo-valer/ASGD-optimizer/blob/main/img/pytorch_asgd.png" width="60%" height="60%"><br>

**My ASGD (green) vs SGD (red):**
<br><img src="https://github.com/jo-valer/ASGD-optimizer/blob/main/img/my_asgd.png" width="60%" height="60%"><br>

## Usage
The code can be downloaded and used as a module. The optimizers can be imported as follows:
```python
from asgd import ASGD
from ntasgd import NTASGD
```

### Averaged SGD
The ASGD optimizer can be used as any other PyTorch optimizer. The following example shows how to initialize it, with a given PyTorch model:
```python
optimizer = ASGD(model.parameters(), lr=0.1, t0=100)
```
The parameter `t0` is the number of epochs before starting the averaging. The averaged parameters are not used for optimization, but only for evaluation. Therefore, the model should be evaluated using the averaged parameters, which can be set as follows:
```python
optimizer.average()
```
The model can be set back to the non-averaged parameters using:
```python
optimizer.revert()
```

For instance, if you have a validation set, you can evaluate the model during training as follows:
```python
optimizer.average()
eval_fn(dev_loader, criterion_eval, model) # Your evaluation function
optimizer.revert()
```

### Non-monotonically Triggered ASGD
The NTASGD optimizer can be used as follows:
```python
optimizer = NTASGD(model, dev_loader, train_loader, train_batch_size, criterion_eval, eval_fn, lr=0.1)
```
Note that the NTASGD optimizer requires the model, **not the parameters**, and the following additional parameters:
- `dev_loader`: the PyTorch DataLoader for the validation set.
- `train_loader`: the PyTorch DataLoader for the training set.
- `train_batch_size`: the batch size used for training.
- `criterion_eval`: the loss function used for evaluation.
- `eval_fn`: the evaluation function used for evaluation.

This because the averaging is triggered by the validation loss, not by a fixed number of epochs.

As for the ASGD optimizer, the averaged parameters can be set using:
```python
optimizer.average()
```
and reverted, if needed, using:
```python
optimizer.revert()
```

## Implementation details
- For now, the averaging is applied to the first parameter group only.
- The NTASGD optimizer expects the `eval_fn` to take `dev_loader`, `criterion_eval`, and `model` as input; and return the validation metric. Thus it assumes this metric has to be minimized. If you use a metric that has to be maximized, you can multiply it by -1.

