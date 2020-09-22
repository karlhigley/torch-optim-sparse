import math
import torch
from torch.optim import Optimizer


class SparserAdamW(Optimizer):

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0):
        params = list(params)

        if not 0.0 < lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 < eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        sparse_params = []
        for index, param in enumerate(params):
            if isinstance(param, dict):
                for d_index, d_param in enumerate(param.get("params", [])):
                    if d_param.is_sparse:
                        sparse_params.append([index, d_index])
            elif param.is_sparse:
                sparse_params.append(index)
        if sparse_params:
            raise ValueError(
                f"Sparse params at indices {sparse_params}: SparserAdamW requires dense parameter tensors"
            )

        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super(SparserAdamW, self).__init__(params, defaults)

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
            for p in group['params']:
                if p.grad is None:
                    continue

                # Perform optimization step
                grad = p.grad
                if not grad.is_sparse:
                    raise RuntimeError('SparseAdam does not support dense gradients, please consider Adam instead')

                # State initialization
                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                state['step'] += 1

                grad = grad.coalesce()  # the update is non-linear so indices must be unique
                grad_indices = grad._indices()
                grad_values = grad._values()
                size = grad.size()

                # Perform weight decay step
                p[grad_indices].mul_(1 - group['lr'] * group['weight_decay'])
                # p.sub_(p.sparse_mask(grad), alpha=(group['lr'] * group['weight_decay']))

                def make_sparse(values):
                    constructor = grad.new
                    if grad_indices.dim() == 0 or values.dim() == 0:
                        return constructor().resize_as_(grad)
                    return constructor(grad_indices, values, size)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                # Decay the first and second moment running average coefficient
                #      old <- b * old + (1 - b) * new
                # <==> old += (1 - b) * (new - old)
                old_exp_avg_values = exp_avg.sparse_mask(grad)._values()
                exp_avg_update_values = grad_values.sub(old_exp_avg_values).mul_(1 - beta1)
                exp_avg.add_(make_sparse(exp_avg_update_values))
                old_exp_avg_sq_values = exp_avg_sq.sparse_mask(grad)._values()
                exp_avg_sq_update_values = grad_values.pow(2).sub_(old_exp_avg_sq_values).mul_(1 - beta2)
                exp_avg_sq.add_(make_sparse(exp_avg_sq_update_values))

                # Dense addition again is intended, avoiding another sparse_mask
                numer = exp_avg_update_values.add_(old_exp_avg_values)
                exp_avg_sq_update_values.add_(old_exp_avg_sq_values)
                denom = exp_avg_sq_update_values.sqrt_().add_(group['eps'])
                del exp_avg_update_values, exp_avg_sq_update_values

                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1

                p.add_(make_sparse(-step_size * numer.div_(denom)))

        return loss
