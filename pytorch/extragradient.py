from torch.optim import Optimizer


class ExtraSGD(Optimizer):
    """Base class for optimizers with extrapolation step.

        Arguments:
        params (iterable): an iterable of :class:`torch.Tensor` s or
            :class:`dict` s. Specifies what Tensors should be optimized.
        defaults: (dict): a dict containing default values of optimization
            options (used when a parameter group doesn't specify them).
    """

    def __init__(self, params, lr):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        self.old_iterate = []
        super().__init__(params, {'lr': lr})

    def gradient_step(self, p, group):
        if p.grad is None:
            return None
        return -group['lr'] * p.grad.data

    def extrapolation(self):
        """Performs the extrapolation step and save a copy of the current parameters for the update step.
        """
        if self.old_iterate:
            raise RuntimeError('Need to call step before calling extrapolation again.')
        for group in self.param_groups:
            for p in group['params']:
                self.old_iterate.append(p.data.clone())

                if p.grad is None:
                    continue

                extrapolation_direction = -group['lr'] * p.grad.data
                # Save the current parameters for the update step.
                # Several extrapolation step can be made before each update but only the parameters before the first extrapolation step are saved.

                # Update the current parameters
                p.data.add_(extrapolation_direction)

    def step(self, closure=None):
        if len(self.old_iterate) == 0:
            raise RuntimeError('Need to call extrapolation before calling step.')

        i = -1
        for group in self.param_groups:
            for p in group['params']:
                i += 1
                normal_to_plane = -group['lr'] * p.grad.data
                if normal_to_plane is None:
                    continue
                # Update the parameters saved during the extrapolation step
                p.data = self.old_iterate[i].add_(normal_to_plane)

        # Free the old parameters
        self.old_iterate.clear()
