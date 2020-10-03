import math
import torch
from torch.optim import SGD


class SparserSGD(SGD):

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad

                grad = grad.coalesce()
                grad_inds = grad._indices()
                grad_values = grad._values()
                size = grad.size()

                def make_sparse(values):
                    constructor = grad.new
                    if grad_inds.dim() == 0 or values.dim() == 0:
                        return constructor().resize_as_(grad)
                    return constructor(grad_inds, values.reshape(grad_values.shape), size)

                if weight_decay != 0:
                    param_values = p.data[grad_inds].squeeze()
                    grad_values.add_(param_values, alpha=weight_decay)

                if momentum != 0:
                    param_state = self.state[p]

                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.clone(grad).detach().to_dense()
                    else:
                        buf = param_state['momentum_buffer']
                        # Only update momentum_buffer where sparse gradient is non-zero
                        buf[grad_inds].mul_(momentum)
                        buf.add_(grad, alpha=(1-dampening))

                    mom_values = buf[grad_inds].squeeze()

                    if nesterov:
                        mom_values = grad_values.add(mom_values, alpha=momentum)

                    p.data.add_(make_sparse(mom_values), alpha=-lr)
                else:
                    p.add_(grad, alpha=-lr)

        return loss
