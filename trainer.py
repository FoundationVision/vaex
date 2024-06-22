import sys
from copy import deepcopy
from pprint import pformat
from typing import Callable, Optional, Tuple

import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib.colors import ListedColormap
from torch.nn.parallel import DistributedDataParallel as DDP

from models import VectorQuantizer, VQVAE, DinoDisc
from utils import arg_util, misc, nan
from utils.amp_opt import AmpOptimizer
from utils.diffaug import DiffAug
from utils.loss import hinge_loss, linear_loss, softplus_loss
from utils.lpips import LPIPS

# from memory_profiler import profile

FTen = torch.Tensor
ITen = torch.LongTensor
BTen = torch.BoolTensor

class VAETrainer(object):
    def __init__(
        self, is_visualizer: bool,
        vae: DDP, vae_wo_ddp: VQVAE, disc: DDP, disc_wo_ddp: DinoDisc, ema_ratio: float,  # decoder, en_de_lin=True, seg_embed=False,
        dcrit: str, vae_opt: AmpOptimizer, disc_opt: AmpOptimizer,
        daug=1.0, lpips_loss: LPIPS = None, lp_reso=64, wei_l1=1.0, wei_l2=0.0, wei_entropy=0.0, wei_lpips=0.5, wei_disc=0.6, adapt_type=1, bcr=5.0, bcr_cut=0.5, reg=0.0, reg_every=16,
        disc_grad_ckpt=False,
        dbg_unused=False, dbg_nan=False,
    ):
        super(VAETrainer, self).__init__()
        self.dbg_unused, self.dbg_nan = dbg_unused, dbg_nan
        if self.dbg_nan:
            print('[dbg_nan mode on]')
            nan.debug_nan_hook(vae)
            nan.debug_nan_hook(disc)
        
        self.vae, self.disc = vae, disc
        self.vae_opt, self.disc_opt = vae_opt, disc_opt
        self.vae_wo_ddp: VQVAE = vae_wo_ddp  # after torch.compile
        self.disc_wo_ddp: DinoDisc = disc_wo_ddp  # after torch.compile
        self.vae_params: Tuple[nn.Parameter] = tuple(self.vae_wo_ddp.parameters())
        self.disc_params: Tuple[nn.Parameter] = tuple(self.disc_wo_ddp.parameters())
        
        self.ema_ratio = ema_ratio
        self.is_visualizer = is_visualizer
        self.using_ema = is_visualizer
        if self.using_ema:
            self.vae_ema: VQVAE = deepcopy(vae_wo_ddp).eval()
        else:
            self.vae_ema: VQVAE = None
        
        self.cmap_sim: ListedColormap = sns.color_palette('viridis', as_cmap=True)
        
        self.dcrit = dcrit
        self.d_criterion: Callable = {  # 'hg' by default
            'hg': hinge_loss, 'hinge': hinge_loss,
            'sp': softplus_loss, 'softplus': softplus_loss,
            'ln': linear_loss, 'lin': linear_loss, 'linear': linear_loss
        }[dcrit]
        
        self.daug = DiffAug(prob=daug, cutout=0.2)
        self.wei_l1, self.wei_l2, self.wei_entropy = wei_l1, wei_l2, wei_entropy
        self.lpips_loss: LPIPS = lpips_loss
        self.lp_reso = lp_reso
        self.adapt_wei_disc = wei_disc > 0
        self.adapt_type = adapt_type
        self.ema_gada: torch.Tensor = None
        self.wei_lpips, self.wei_disc = wei_lpips*2, abs(wei_disc)
        self.reg = 0.5 * reg * reg_every
        # balanced_consistency_regularization, 10.0 is used by StyleSwin
        self.bcr = bcr * 2  # LEGACY *2: in the old version, bcr MSE losses on real/fake images are calculated separately and added up; so *2 in the new version
        if self.bcr > 0:
            self.bcr_strong_aug = DiffAug(prob=1, cutout=bcr_cut)
        self.disc_grad_ckpt = disc_grad_ckpt
    
    # @profile(precision=4, stream=open('trainstep.log', 'w+'))
    def train_step(
        self, ep: int, it: int, g_it: int, stepping: bool, regularizing: bool, metric_lg: misc.MetricLogger, logging_params: bool, tb_lg: misc.TensorboardLogger,
        inp: FTen, warmup_disc_schedule: float, fade_blur_schedule: float,
        maybe_record_function: Callable,
        args: arg_util.Args,
    ) -> Tuple[torch.Tensor, Optional[float], Optional[torch.Tensor], Optional[float]]:
        if warmup_disc_schedule < 1e-6: warmup_disc_schedule = 0
        if fade_blur_schedule < 1e-6: fade_blur_schedule = 0
        loggable = (g_it == 0 or (g_it + 1) % 600 == 0) and self.is_visualizer
        
        # [vae loss]
        with maybe_record_function('VAE_rec'):
            with self.vae_opt.amp_ctx:
                self.vae_wo_ddp.forward
                rec_B3HW, Lq, Le, usage = self.vae(inp, ret_usages=loggable)
                B = rec_B3HW.shape[0]
                inp_rec_no_grad = torch.cat((inp, rec_B3HW.data), dim=0)
            
            Lrec = F.l1_loss(rec_B3HW, inp)
            Lrec_for_log = Lrec.data.clone()
            Lrec *= self.wei_l1
            if self.wei_l2 > 0:
                Lrec += F.mse_loss(rec_B3HW, inp).mul_(self.wei_l2)
            # if self.wei_llaplace > 0:
            #     inp_01_09 = inp.mul(0.4).add_(0.5)
            #     dist = (rec_B3HW.sigmoid() - inp_01_09.sigmoid()).abs()
            #     # dist /= lnb.exp().square().mul_(inp_01_09.add(inp_01_09).mul_(1-inp_01_09)).add_(1).mul_(0.5)
            #     dist /= inp_01_09.add(inp_01_09).mul_(1-inp_01_09).add_(1).mul_(0.5)
            #     Lrec += dist.mean().mul_(self.wei_llaplace)
            
            using_lpips = inp.shape[-2] >= self.lp_reso and self.wei_lpips > 0
            if using_lpips:
                self.lpips_loss.forward
                Lpip = self.lpips_loss(inp, rec_B3HW)
                Lnll = Lrec + self.wei_lpips * Lpip
            else:
                Lpip = torch.tensor(0.)
                Lnll = Lrec
        
        if warmup_disc_schedule > 0:
            with maybe_record_function('VAE_disc'):
                for d in self.disc_params: d.requires_grad = False
                self.disc_wo_ddp.eval()
                with self.disc_opt.amp_ctx:
                    self.disc_wo_ddp.forward
                    Lg = -self.disc_wo_ddp(self.daug.aug(rec_B3HW, fade_blur_schedule), grad_ckpt=False).mean()  # todo: aug or not?
                self.disc_wo_ddp.train()
                
                wei_g = warmup_disc_schedule * self.wei_disc
                if self.adapt_wei_disc:
                    last_layer = self.vae_wo_ddp.decoder.conv_out.weight
                    w = (
                        torch.autograd.grad(Lnll, last_layer, retain_graph=True)[0].data.norm()
                        / (torch.autograd.grad(Lg, last_layer, retain_graph=True)[0].data.norm().add_(1e-6))
                    )
                    if self.adapt_type % 10 == 0:
                        w.clamp_(0.0, 1e4)
                    elif self.adapt_type % 10 == 1:
                        w.clamp_(0.015, 1e4)
                    elif self.adapt_type % 10 == 2:
                        w.clamp_(0.1, 10)
                        w = min(max(w, 0.1), 10)
                    elif self.adapt_type % 10 == 3:
                        w.clamp_(0.0, 1e4).sqrt_()
                    
                    if self.adapt_type >= 10:
                        if self.ema_gada is None:
                            self.ema_gada = w
                        else:
                            self.ema_gada.mul_(0.9).add_(w, alpha=0.1)
                            w = self.ema_gada
                    wei_g = wei_g * w
                
                Lv = Lnll + Lq + self.wei_entropy * Le + wei_g * Lg
        else:
            Lv = Lnll + Lq + self.wei_entropy * Le
            Lg = torch.tensor(0.)
            wei_g = None
        
        # todo: G D backward together;   less calling .item()
        # todo: G D backward together;   less calling .item()
        with maybe_record_function('VAE_backward'):
            grad_norm_g, scale_log2_g = self.vae_opt.backward_clip_step(stepping=stepping, loss=Lv)
        
        # [discriminator loss]
        if warmup_disc_schedule > 0:
            with maybe_record_function('Disc_forward'):
                for d in self.disc_params: d.requires_grad = True
                with self.disc_opt.amp_ctx:
                    self.disc_wo_ddp.forward
                    logits = self.disc(self.daug.aug(inp_rec_no_grad, fade_blur_schedule), grad_ckpt=self.disc_grad_ckpt).float()
                
                logits_real, logits_fake = logits[:B], logits[B:]
                acc_real, acc_fake = (logits_real.data > 0).float().mean().mul_(100), (logits_fake.data < 0).float().mean().mul_(100)
                
                Ld = self.d_criterion(logits_real) + self.d_criterion(-logits_fake)
            
            if self.bcr:
                with maybe_record_function('Disc_bCR'):
                    with self.disc_opt.amp_ctx:
                        self.disc_wo_ddp.forward
                        logits2 = self.disc(self.bcr_strong_aug.aug(inp_rec_no_grad, 0.0), grad_ckpt=self.disc_grad_ckpt).float()
                    Lbcr = F.mse_loss(logits2, logits).mul_(self.bcr)
                    Ld += Lbcr
            else:
                Lbcr = torch.tensor(0.)
            
            if regularizing:
                with maybe_record_function('Disc_reg'):
                    self.disc_wo_ddp.eval()
                    with torch.cuda.amp.autocast(enabled=False):    # todo: why AMP is disabled in this disc forward?
                        inp.requires_grad_(True)
                        self.disc_wo_ddp.forward
                        grad_real = torch.autograd.grad(outputs=self.disc(self.daug.aug(inp, fade_blur_schedule), grad_ckpt=False).sum(), inputs=inp, create_graph=True)[0]
                        Lreg = grad_real.square().flatten(1).sum(dim=1).mean()
                        Ld += self.reg * Lreg
                        Lreg = Lreg.item()
                        inp.requires_grad_(False)
                    self.disc_wo_ddp.train()
            else:
                Lreg = 0.
            
            with maybe_record_function('Disc_backward'):
                grad_norm_d, scale_log2_d = self.disc_opt.backward_clip_step(stepping=stepping, loss=Ld)
                Ld = Ld.data.clone()
        else:
            Ld = acc_real = acc_fake = grad_norm_d = scale_log2_d = None
            Lbcr = torch.tensor(0.)
        
        # [zero_grad]
        if stepping:
            if self.using_ema:
                with maybe_record_function('EMA_upd'):
                    self.ema_update(g_it)
            
            if self.dbg_nan:
                nan.debug_nan_grad(self.vae_wo_ddp), nan.debug_nan_grad(self.disc_wo_ddp)
                nan.debug_nan_param(self.vae_wo_ddp), nan.debug_nan_param(self.disc_wo_ddp)
            if self.dbg_unused:
                ls = []
                for n, p in self.vae_wo_ddp.named_parameters():
                    if p.grad is None and n not in {'quantize.embedding.weight'}: # or tuple(p.grad.shape) == (512, 512, 1, 1):
                        ls.append(n)
                for n, p in self.disc_wo_ddp.named_parameters():
                    if p.grad is None: # or tuple(p.grad.shape) == (512, 512, 1, 1):
                        ls.append(n)
                if len(ls):
                    print(f'unused param: {ls}', flush=True, file=sys.stderr)
            
            with maybe_record_function('opt_step'):
                self.vae_opt.optimizer.zero_grad(set_to_none=True)
                self.disc_opt.optimizer.zero_grad(set_to_none=True)
        
        with maybe_record_function('trainer_log'):
            # [metric logging]
            if it == 0 or it in metric_lg.log_iters:
                Lpip = Lpip.item()
                Lnll = Lrec_for_log + Lpip
                metric_lg.update(L1=Lrec_for_log, NLL=Lnll, Ld=Ld, Wg=wei_g, acc_real=acc_real, acc_fake=acc_fake, gnm=grad_norm_g, dnm=grad_norm_d)
            
            # [tensorboard logging]
            if loggable:
                Lbcr, Lq, Le, Lg = Lbcr.item(), Lq.item(), Le if isinstance(Le, (int, float)) else Le.item(), Lg.item()
                
                # vae_vocab_size = self.vae_wo_ddp.vocab_size
                # prob_per_class_is_chosen = idx_N.bincount()
                # prob_per_class_is_chosen = F.pad(prob_per_class_is_chosen, pad=(0, vae_vocab_size-prob_per_class_is_chosen.shape[0]), mode='constant', value=0).float() / prob_per_class_is_chosen.sum()
                # log_perplexity = (-(prob_per_class_is_chosen * torch.log(prob_per_class_is_chosen + 1e-10)).sum())
                # cluster_usage = (prob_per_class_is_chosen > 0.05 / vae_vocab_size).float().mean() * 100
                kw = dict(
                    # total=Lnll + Lq + self.wei_disc * Lg,
                    Nll=Lnll, RecL1=Lrec_for_log, quant=Lq,
                    # z_log_perplex=log_perplexity, z_voc_usage=cluster_usage
                )
                kw[f'z_voc_usage'] = usage
                if Le > 1e-6: kw['entropy'] = Le
                if Lpip > 1e-6: kw['Lpip'] = Lpip
                tb_lg.update(head='PT_iter_V_loss', step=g_it, **kw)
                
                if warmup_disc_schedule > 0:
                    kw = dict(Disc=Ld-Lbcr-Lreg, bcr=Lbcr, give_vae=Lg)
                    if Lreg > 1e-6: kw['regR1'] = Lreg
                    tb_lg.update(head='PT_iter_D_loss', step=g_it, **kw)
                    tb_lg.update(
                        head='PT_iter_pred',
                        logits_real=logits_real.data.mean(), logits_fake=logits_fake.data.mean(),
                        logits_L1dis_normed=F.l1_loss(logits_real.data, logits_fake.data).mul_(3.0178) / (logits_real.data.abs().mean() + logits_fake.data.abs().mean()),
                        acc_real=acc_real, acc_fake=acc_fake, step=g_it
                    )
                
                tb_lg.update(head='PT_iter_schedule', warm_disc=warmup_disc_schedule, fade_blur=fade_blur_schedule, step=g_it)
        
        return grad_norm_g, scale_log2_g, grad_norm_d, scale_log2_d
    
    def __repr__(self):
        return (
            f'\n'
            f'[{type(self).__name__}.config]: {pformat(self.get_config(), indent=2, width=250)}\n'
            f'[{type(self).__name__}.structure]: {super(VAETrainer, self).__repr__().replace(VAETrainer.__name__, "")}'
        )
    
    # p_ema = p_ema*0.9 + p*0.1 <==> p_ema.lerp_(p, 0.1)
    # p_ema.mul_(self.ema_ratio).add_(p.mul(self.ema_ratio_1))
    # @profile(precision=4, stream=open('ema_update.log', 'w+'))
    def ema_update(self, g_it):
        ema_ratio = min(self.ema_ratio, (g_it//2 + 1) / (g_it//2 + 10))
        for p_ema, p in zip(self.vae_ema.parameters(), self.vae_wo_ddp.parameters()):
            if p.requires_grad:
                p_ema.data.mul_(ema_ratio).add_(p.data, alpha=1-ema_ratio)
        for p_ema, p in zip(self.vae_ema.buffers(), self.vae_wo_ddp.buffers()):
            p_ema.data.copy_(p.data)
        quant, quant_ema = self.vae_wo_ddp.quantize, self.vae_ema.quantize
        quant: VectorQuantizer
        if hasattr(quant, 'using_ema') and quant.using_ema: # then embedding.weight requires no grad, thus is not in self.vae_ema_params; so need to update it manually
            if hasattr(quant, 'using_restart') and quant.using_restart:
                # cannot use ema, cuz quantize.embedding uses replacement (rand restart)
                quant_ema.embedding.weight.data.copy_(quant.embedding.weight.data)
            else:
                quant_ema.embedding.weight.data.mul_(ema_ratio).add_(quant.embedding.weight.data, alpha=1-ema_ratio)
    
    def get_config(self):
        return {
            'ema_ratio': self.ema_ratio,
            'dcrit': self.dcrit,
            'wei_l1': self.wei_l1, 'wei_l2': self.wei_l2, 'wei_lpips': self.wei_lpips, 'wei_disc': self.wei_disc,
            'bcr': self.bcr, 'reg': self.reg,
        }
    
    def state_dict(self):
        state = {'config': self.get_config()}
        for k in ('vae_wo_ddp', 'vae_ema', 'disc_wo_ddp', 'vae_opt', 'disc_opt'):
            m = getattr(self, k)
            if m is not None:
                if hasattr(m, '_orig_mod'):
                    m = m._orig_mod
                state[k] = m.state_dict()
        return state
    
    def load_state_dict(self, state, strict=True):
        for k in ('vae_wo_ddp', 'vae_ema', 'disc_wo_ddp', 'vae_opt', 'disc_opt'):
            m = getattr(self, k)
            if m is not None:
                if hasattr(m, '_orig_mod'):
                    m = m._orig_mod
                ret = m.load_state_dict(state[k], strict=strict)
                if ret is not None:
                    missing, unexpected = ret
                    print(f'[VAETr.load_state_dict] {k} missing:  {missing}')
                    print(f'[VAETr.load_state_dict] {k} unexpected:  {unexpected}')
        config: dict = state.pop('config', None)
        if config is not None:
            for k, v in self.get_config().items():
                if config.get(k, None) != v:
                    err = f'[VAETr.load_state_dict] config mismatch:  this.{k}={v} (ckpt.{k}={config.get(k, None)})'
                    if strict:
                        raise AttributeError(err)
                    else:
                        print(err)
