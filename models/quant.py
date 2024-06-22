from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch import distributed as tdist, nn as nn
from torch.nn import functional as F

import dist


# this file only provides the VectorQuantizer2 used in VQVAE
__all__ = ['VectorQuantizer', ]


class NormalizedEmbedding(nn.Embedding):
    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__(num_embeddings=num_embeddings, embedding_dim=embedding_dim)
        self.norm_scale = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
    
    def forward(self, idx):
        return F.embedding(
            idx, F.normalize(self.weight, dim=1).mul_(self.norm_scale.sigmoid()), self.padding_idx, self.max_norm,
            self.norm_type, self.scale_grad_by_freq, self.sparse
        )


class ResConv(nn.Conv2d):
    def __init__(self, embed_dim, quant_resi):
        ks = 3 if quant_resi < 0 else 1
        super().__init__(in_channels=embed_dim, out_channels=embed_dim, kernel_size=ks, stride=1, padding=ks//2)
        self.resi_ratio = abs(quant_resi)
    
    def forward(self, h_BChw):
        return h_BChw.mul(1-self.resi_ratio) + super().forward(h_BChw).mul_(self.resi_ratio)


class VectorQuantizer(nn.Module):
    def __init__(
        self, vocab_size: int, vocab_width: int, vocab_norm: bool, beta: float = 0.25, quant_resi=-0.5,
    ):
        super().__init__()
        self.vocab_size: int = vocab_size
        self.vocab_width: int = vocab_width
        self.register_buffer('vocab_usage', torch.zeros(self.vocab_size))
        self.vocab_usage_record_times: int = 0
        
        self.vocab_norm: bool = vocab_norm
        self.quant_resi = ResConv(self.vocab_width, quant_resi=quant_resi)
        # self.embedding = (NormalizedEmbedding if vocab_norm else nn.Embedding)(self.vocab_size, self.vocab_width)
        self.embedding = nn.Embedding(self.vocab_size, self.vocab_width)
        self.beta: float = beta
    
    def init_vocab(self, eini: float):
        if eini > 0:
            nn.init.trunc_normal_(self.embedding.weight.data, std=eini)
        elif eini < 0:
            base = self.vocab_width ** -0.5
            base /= 36
            self.embedding.weight.data.uniform_(-abs(eini) * base, abs(eini) * base)
    
    def extra_repr(self) -> str:
        return f'beta={self.beta:g}'
    
    # ===================== `forward` is only used in VAE training =====================
    def forward(self, f_BChw: torch.Tensor, ret_usages=False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[float]]:
        f_BChw = f_BChw.float()
        B, C, h, w = f_BChw.shape
        # find the nearest embedding
        query_NxC = f_BChw.detach().permute(0, 2, 3, 1).reshape(-1, C)
        if self.vocab_norm:
            query_NxC = F.normalize(query_NxC, dim=-1)
            idx_N = torch.argmax(query_NxC @ F.normalize(self.embedding.weight.data.T, dim=0), dim=1)
        else:
            E_dist = torch.sum(query_NxC.square(), dim=1, keepdim=True) + torch.sum(self.embedding.weight.data.square(), dim=1, keepdim=False)
            E_dist.addmm_(query_NxC, self.embedding.weight.data.T, alpha=-2, beta=1)  # (B*h*w, vocab_size)
            idx_N = torch.argmin(E_dist, dim=1)
        
        prob_per_class_is_chosen = idx_N.bincount(minlength=self.vocab_size).float()
        handler = tdist.all_reduce(prob_per_class_is_chosen, async_op=True) if (self.training and dist.initialized()) else None
        
        # look up
        idx_Bhw = idx_N.view(B, h, w)
        fhat_BChw = self.quant_resi(self.embedding(idx_Bhw).permute(0, 3, 1, 2).contiguous())
        
        # calc loss
        vq_loss = F.mse_loss(fhat_BChw.detach(), f_BChw).mul_(self.beta) + F.mse_loss(fhat_BChw, f_BChw.detach())
        
        # VQVAE: straight through gradient estimation, copy the gradient on fhat_BChw to f_BChw
        fhat_BChw = (fhat_BChw.detach() - f_BChw.detach()).add_(f_BChw)
        
        # update vocab_usage
        if handler is not None: handler.wait()
        prob_per_class_is_chosen /= prob_per_class_is_chosen.sum()
        vocab_usage = (prob_per_class_is_chosen > 0.01 / self.vocab_size).float().mean().mul_(100)
        
        if self.vocab_usage_record_times == 0: self.vocab_usage.copy_(prob_per_class_is_chosen)
        elif self.vocab_usage_record_times < 100: self.vocab_usage.mul_(0.9).add_(prob_per_class_is_chosen, alpha=0.1)
        else: self.vocab_usage.mul_(0.99).add_(prob_per_class_is_chosen, alpha=0.01)
        self.vocab_usage_record_times += 1
        
        entropy_loss = 0.0 # todo: not implemented yet
        return fhat_BChw, vq_loss, entropy_loss, (vocab_usage if ret_usages else None)
    # ===================== `forward` is only used in VAE training =====================
    
    def f_to_idx(self, f_BChw: torch.Tensor) -> torch.LongTensor:
        f_BChw = f_BChw.float()
        B, C, h, w = f_BChw.shape
        with torch.cuda.amp.autocast(enabled=False):
            # find the nearest embedding
            query_NxC = f_BChw.detach().permute(0, 2, 3, 1).reshape(-1, C)
            if self.vocab_norm:
                query_NxC = F.normalize(query_NxC, dim=-1)
                idx_N = torch.argmax(query_NxC @ F.normalize(self.embedding.weight.data.T, dim=0), dim=1)
            else:
                E_dist = torch.sum(query_NxC.square(), dim=1, keepdim=True) + torch.sum(self.embedding.weight.data.square(), dim=1, keepdim=False)
                E_dist.addmm_(query_NxC, self.embedding.weight.data.T, alpha=-2, beta=1)  # (B*h*w, vocab_size)
                idx_N = torch.argmin(E_dist, dim=1)
        return idx_N.view(B, h, w)
