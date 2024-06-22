import math
from typing import Tuple

import torch
# from deepspeed.ops import lamb
from torch.optim.optimizer import Optimizer


class LAMBtimm(Optimizer):
    """Implements a pure pytorch variant of FuseLAMB (NvLamb variant) optimizer from apex.optimizers.FusedLAMB
    reference: https://github.com/NVIDIA/DeepLearningExamples/blob/master/PyTorch/LanguageModeling/Transformer-XL/pytorch/lamb.py

    LAMB was proposed in `Large Batch Optimization for Deep Learning: Training BERT in 76 minutes`_.

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining parameter groups.
        lr (float, optional): learning rate. (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its norm. (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability. (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        grad_averaging (bool, optional): whether apply (1-beta2) to grad when
            calculating running averages of gradient. (default: True)
        max_grad_norm (float, optional): value used to clip global grad norm (default: 1.0)
        trust_clip (bool): enable LAMBC trust ratio clipping (default: False)
        always_adapt (boolean, optional): Apply adaptive learning rate to 0.0
            weight decay parameter (default: False)

    .. _Large Batch Optimization for Deep Learning - Training BERT in 76 minutes:
        https://arxiv.org/abs/1904.00962
    .. _On the Convergence of Adam and Beyond:
        https://openreview.net/forum?id=ryQu7f-RZ
    """
    
    def __init__(
        self, params, lr=1e-3, bias_correction=True, betas=(0.9, 0.999), eps=1e-7,
        weight_decay=0.01, grad_averaging=True, max_grad_norm=2.0, trust_clip=False, always_adapt=False):
        defaults = dict(
            lr=lr, bias_correction=bias_correction, betas=betas, eps=eps, weight_decay=weight_decay,
            grad_averaging=grad_averaging, max_grad_norm=max_grad_norm,
            trust_clip=trust_clip, always_adapt=always_adapt)
        super().__init__(params, defaults)
        self.global_grad_norm = torch.tensor(0.1)
    
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
        
        device = self.param_groups[0]['params'][0].device
        one_tensor = torch.tensor(1.0, dtype=torch.float32, device=device)  # because torch.where doesn't handle scalars correctly
        global_grad_norm = torch.full(size=(1,), fill_value=1e-12, dtype=torch.float32, device=device)
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError('Lamb does not support sparse gradients, consider SparseAdam instad.')
                global_grad_norm.add_(grad.pow(2).sum())
        
        global_grad_norm = torch.sqrt(global_grad_norm)
        self.global_grad_norm = global_grad_norm
        max_grad_norm = torch.tensor(self.defaults['max_grad_norm'], dtype=torch.float32, device=device)
        clip_global_grad_norm = 1 / torch.where(
            global_grad_norm > max_grad_norm,
            global_grad_norm / max_grad_norm,
            one_tensor)
        
        for group in self.param_groups:
            bias_correction = 1 if group['bias_correction'] else 0
            beta1, beta2 = group['betas']
            grad_averaging = 1 if group['grad_averaging'] else 0
            beta3 = 1 - beta1 if grad_averaging else 1.0
            
            # assume same step across group now to simplify things
            # per parameter step can be easily support by making it tensor, or pass list into kernel
            if 'step' in group:
                group['step'] += 1
            else:
                group['step'] = 1
            
            if bias_correction:
                bias_correction1 = 1 - beta1 ** group['step']
                bias_correction2 = 1 - beta2 ** group['step']
            else:
                bias_correction1, bias_correction2 = 1.0, 1.0
            
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.mul_(clip_global_grad_norm)
                state = self.state[p]
                
                # State initialization
                if len(state) == 0:
                    # Exponential moving average of gradient valuesa
                    state['exp_avg'] = torch.zeros_like(p)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p)
                
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                
                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(grad, alpha=beta3)  # m_t
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)  # v_t
                
                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])
                update = (exp_avg / bias_correction1).div_(denom)
                
                weight_decay = group['weight_decay']
                if weight_decay != 0:
                    update.add_(p, alpha=weight_decay)
                
                if weight_decay != 0 or group['always_adapt']:
                    # Layer-wise LR adaptation. By default, skip adaptation on parameters that are
                    # excluded from weight decay, unless always_adapt == True, then always enabled.
                    w_norm = p.norm(2.0)
                    g_norm = update.norm(2.0)
                    trust_ratio = torch.where(
                        w_norm > 0,
                        torch.where(g_norm > 0, w_norm / g_norm, one_tensor),
                        one_tensor,
                        )
                    if group['trust_clip']:
                        # LAMBC trust clipping, upper bound fixed at one
                        trust_ratio = torch.minimum(trust_ratio, one_tensor)
                    update.mul_(trust_ratio)
                
                p.add_(update, alpha=-group['lr'])
        
        return loss


class Lion(Optimizer):
    def __init__(
        self,
        params,
        lr: float = 1e-4,
        betas: Tuple[float, float] = (0.9, 0.99),
        weight_decay: float = 0.0,
        use_triton: bool = False
    ):
        assert lr > 0.
        assert all([0. <= beta <= 1. for beta in betas])
        
        defaults = dict(
            lr=lr,
            betas=betas,
            weight_decay=weight_decay
        )
        
        super().__init__(params, defaults)
    
    def update_fn(self, p, grad, exp_avg, lr, wd, beta1, beta2):
        # stepweight decay
        p.data.mul_(1 - lr * wd)
        
        # weight update
        update = exp_avg.clone().mul_(beta1).add(grad, alpha=1 - beta1).sign_()
        p.add_(update, alpha=-lr)
        
        # decay the momentum running average coefficient
        exp_avg.mul_(beta2).add_(grad, alpha=1 - beta2)
    
    @torch.no_grad()
    def step(
        self,
        closure=None
    ):
        
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        for group in self.param_groups:
            for p in filter(lambda p: p.grad is not None, group['params']):
                
                grad, lr, wd, beta1, beta2, state = p.grad, group['lr'], group['weight_decay'], *group['betas'], self.state[p]
                
                # init state - exponential moving average of gradient values
                
                if len(state) == 0:
                    state['exp_avg'] = torch.zeros_like(p)
                
                exp_avg = state['exp_avg']
                
                self.update_fn(
                    p,
                    grad,
                    exp_avg,
                    lr,
                    wd,
                    beta1,
                    beta2
                )
        
        return loss
