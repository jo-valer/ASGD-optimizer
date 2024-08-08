from typing import Optional
from math import ceil

import torch

class NTASGD(torch.optim.SGD):
    """
    Implements Non-monotonically Triggered ASGD.
    """

    def __init__(
            self,
            model: torch.nn.Module,
            dev_loader: torch.utils.data.DataLoader,
            train_loader: torch.utils.data.DataLoader,
            train_batch_size: int,
            criterion_eval: callable,
            eval_fn: callable,
            lr: float = 1.0,
            non_monotone_interval: int = 5,
            momentum: float = 0,
            dampening: float = 0,
            weight_decay: float = 0,
            nesterov=False,
            *,
            maximize: bool = False,
            foreach: Optional[bool] = None,
            differentiable: bool = False,
            fused: Optional[bool] = None
            ):

        super(NTASGD, self).__init__(model.parameters(), lr, momentum, dampening, weight_decay, nesterov, maximize=maximize, foreach=foreach, differentiable=differentiable, fused=fused)
        self.t0 = 0
        self.model = model
        self.logging_interval = ceil(len(train_loader.dataset)/train_batch_size)
        self.non_monotone_interval = non_monotone_interval
        self.t = 0
        self.k = 0
        self.logs = []
        self.ax = {} # To store the average of the parameters
        self.mu = 1 # To store the number of times the average has been updated (+1 at the beginning)
        self.last_params = {} # To store the last parameters before averaging
        self.dev_loader = dev_loader
        self.criterion_eval = criterion_eval
        self.eval_fn = eval_fn

    def step(self, closure=None):
        super(NTASGD, self).step(closure)

        with torch.no_grad():
            if self.k % self.logging_interval == 0 and self.t0 == 0: # if mod(k, L) = 0 and T = 0.
                ppl_dev, _ = self.eval_fn(self.dev_loader, self.criterion_eval, self.model) # Compute validation perplexity v
                self.model.train()
                if self.t > self.non_monotone_interval and ppl_dev > min(self.logs[:self.t-self.non_monotone_interval]): # if t > n and v > min l∈{0,··· ,t−n−1}logs[l]
                    self.t0 = self.k
                    print("Averaging triggered at t0 =", self.k)
                self.logs.append(ppl_dev) # Append v to logs
                self.t += 1
            self.k += 1

            if self.t0 > 0:
                for parameter in self.param_groups[0]['params']:
                    if parameter not in self.ax:
                        self.ax[parameter] = parameter.data.clone()
                    else:
                        self.ax[parameter] = self.ax[parameter] + (parameter.data - self.ax[parameter]) / self.mu
                self.mu += 1

    def average(self):
        """
        Set the parameters to the average (computed from the triggering point to the last iteration).
        """
        if self.t0 == 0: # Do nothing
            return
        else:
            for parameter in self.param_groups[0]['params']:
                self.last_params[parameter] = parameter.data.clone() # Store the last parameters before averaging
                parameter.data = self.ax[parameter].clone() # Replace the parameters with the averaged ones

    def revert(self):
        """
        Set the parameters to the last ones before averaging.
        """
        if self.t0 == 0: # Do nothing
            return
        else:
            for parameter in self.param_groups[0]['params']:
                parameter.data = self.last_params[parameter].clone() # Revert the parameters to the last ones before averaging
