__version__ = '0.1.3'

from .sparser_adam import SparserAdam
from .sparser_adamw import SparserAdamW
from .sparser_sgd import SparserSGD
from .sparser_sgdw import SparserSGDW


def convert_lr(eff_lr, momentum=0.0, beta1=0.0, beta2=0.0, batch_size=1):
    """Calculates what learning rate to use for rough equivalence with plain SGD

    Useful for supplying one set of hyper-parameters to sweep across with multiple optimizers
    and getting them all to converge with hyper-parameters that are somewhere near the same order
    of magnitude. Accounts for the effects of optimizer batch size, momentum, and adaptive
    learning rates in Adam and SGD variants.

    All params except the effective learning rate are optional; only supply the params that are
    relevant to the optimizer you want to use.

    Args:
        eff_lr (float): The effective learning rate you want.
        momentum (float, optional): The SGD momentum coefficient. Defaults to 0.0, but 0.9 is typical.
        beta1 (float, optional): The Adam first moment coefficient. Defaults to 0.0, but 0.9 is typical.
        beta2 (float, optional): The Adam second moment coefficient. Defaults to 0.0, but 0.999 is typical.
        batch_size (int, optional): The number of examples in a mini-batch. Defaults to 1.

    Returns:
        lr (float): The adjusted learning rate to supply to the optimizer
    """
    lr = eff_lr

    if beta1 != 1.0 or beta2 != 1.0:
        lr = lr * (1 - beta2) / (1 - beta1)

    if momentum != 0.0:
        lr = lr * (1 - momentum)
    
    if batch_size > 1:
        lr = lr * batch_size

    return lr
