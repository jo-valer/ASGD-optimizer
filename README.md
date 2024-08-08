# Averaged Stochastic Gradient Descent

I found that PyTorch ASGD behaves differently than SGD before the averaging starting point, even though it is not supposed to. In particular SGD converges more rapidly (see image below). For this reason, here is an implementation of ASGD that behaves like SGD before the averaging. The averaging technique has been proposed by [Polyak and Juditsky, 1992](https://dl.acm.org/doi/10.1137/0330046).

In ASGD, the point at which to start averaging is a hyper-parameter. For this reason I also implemented the Non-monotonically Triggered ASGD proposed by [Merity et al., 2017](https://arxiv.org/abs/1708.02182). This method triggers the averaging when the validation loss does not decrease for a certain number of epochs, avoiding the need to set a fixed averaging starting point.

### Pytorch ASGD (green) vs SGD (purple)
<br><img src="https://github.com/jo-valer/ASGD-optimizer/blob/main/img/pytorch_asgd.png" width="60%" height="60%"><br>

### My ASGD (green) vs SGD (red)
<br><img src="https://github.com/jo-valer/ASGD-optimizer/blob/main/img/my_asgd.png" width="60%" height="60%"><br>

## Implementation details
- For now, the averaging is applied to the first parameter group only.

