# Averaged Stochastic Gradient Descent

I found that PyTorch ASGD behaves differently than SGD before the averaging triggering point, even though it is not supposed to. In particular SGD converges more rapidly. For this reason, here is an implementation of ASGD that behaves like SGD before the averaging. ASGD has been proposed by [Polyak and Juditsky, 1992](https://dl.acm.org/doi/10.1137/0330046).

I also implemented the Non-monotonically Triggered ASGD proposed by [Merity et al., 2017](https://arxiv.org/abs/1708.02182).
