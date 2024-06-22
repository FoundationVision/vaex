import torch
from torch.nn import functional as F

# usage: like tot_loss = hinge_loss(logits_real) + hinge_loss(-logits_fake)
def hinge_loss(logits: torch.Tensor): return (1 - logits).relu().mean()
def softplus_loss(logits: torch.Tensor): return F.softplus(-logits).mean()
def linear_loss(logits: torch.Tensor): return (-logits).mean()


def focal_l1_loss(
    pred, target, reduction='none',
    alpha=0.2, gamma=1.0, activate='sigmoid', residual=False, weight=None
):
    r"""Calculate Focal L1 loss.

    Delving into Deep Imbalanced Regression. In ICML, 2021.
    <https://arxiv.org/abs/2102.09554>

    Args:
        pred (torch.Tensor): The prediction with shape (N, \*).
        target (torch.Tensor): The regression target with shape (N, \*).
        alpha (float): A balanced form for Focal Loss. Defaults to 0.2.
        gamma (float): The gamma for calculating the modulating factor.
            Defaults to 1.0.
        activate (str): activate methods in Focal loss in {'sigmoid', 'tanh'}.
            Defaults to 'sigmoid'.
        residual (bool): Whether to use the original l1_loss, i.e., l1 + focal_l1.
            Defaults to False.
        weight (tensor): Sample-wise reweight of (N, \*) or element-wise
            reweight of (1, \*). Defaults to None.
        reduction (str): The method used to reduce the loss.

    Returns:
        torch.Tensor: The calculated loss
    """
    _loss = F.l1_loss(pred, target, reduction='none')
    if activate == 'tanh':
        loss = _loss * (torch.tanh(alpha * _loss)) ** gamma
    else:
        loss = _loss * (2. * torch.sigmoid(alpha * _loss) - 1.) ** gamma
    if residual:
        loss += _loss
    
    if weight is not None:
        loss *= weight.expand_as(loss)
    if reduction == 'mean':
        loss = loss.mean()
    elif reduction == 'sum':
        loss = loss.sum()
    return loss


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import torch as tc
    x = tc.linspace(-1.3, 1.3, 200)
    gt = tc.zeros_like(x)
    l1 = F.l1_loss(x, gt, reduction='none')
    l2 = F.mse_loss(x, gt, reduction='none')
    fl1 = focal_l1_loss(x, gt, reduction='none')
    plt.plot(x, l1, 'r', x, l2, 'g', x, fl1, 'b')
    plt.show()
    