import torch
import warnings

# Disable warnings
warnings.filterwarnings("ignore")

class ASGD(torch.optim.SGD):
    """
    Implements Averaged Stochastic Gradient Descent.
    """

    def __init__(
            self,
            params,
            lr: float = 1.0,
            t0: int = 1
            ):

        super(ASGD, self).__init__(params, lr=lr)
        self.t0 = t0
        self.t = 0
        self.ax = {} # To store the average of the parameters
        self.mu = 1 # To store the number of times the average has been updated (+1 at the beginning)
        self.last_params = {} # To store the last parameters before averaging

    def step(self, closure=None):
        super(ASGD, self).step(closure)

        with torch.no_grad():
            self.t += 1
            if self.t >= self.t0:
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
        if self.t < self.t0: # No averaging before the triggering point
            return
        else:
            for parameter in self.param_groups[0]['params']:
                self.last_params[parameter] = parameter.data.clone() # Store the last parameters before averaging
                parameter.data = self.ax[parameter].clone() # Replace the parameters with the averaged ones

    def revert(self):
        """
        Set the parameters to the last ones before averaging.
        """
        if self.t < self.t0: # No averaging before the triggering point
            return
        else:
            for parameter in self.param_groups[0]['params']:
                parameter.data = self.last_params[parameter].clone() # Revert the parameters to the last ones before averaging
