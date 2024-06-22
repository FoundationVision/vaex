import time
import warnings
from typing import List, Tuple

import PIL.Image as PImage, PIL.ImageDraw as PImageDraw
import numpy as np
import seaborn as sns
import torch
import torch.nn.functional as F
import torchvision
from matplotlib.colors import ListedColormap

from dist import for_visualize
from trainer import VAETrainer
from utils import misc


class Visualizer(object):
    def __init__(self, enable: bool, device, trainer: VAETrainer):
        self.enable = enable
        if enable:
            self.trainer: VAETrainer
            self.device, self.trainer = device, trainer
            # self.data_m = torch.tensor(dataset_mean, dtype=torch.float32, device=self.device).view(1, 3, 1, 1)
            # self.data_s = torch.tensor(dataset_std, dtype=torch.float32, device=self.device).view(1, 3, 1, 1)
            
            self.inp_B3HW: torch.Tensor = ...
            self.bound_mask: torch.Tensor = ...
            self.cmap_div: ListedColormap = sns.color_palette('mako', as_cmap=True)
            self.cmap_div: ListedColormap = sns.color_palette('icefire', as_cmap=True)
            self.cmap_seq = ListedColormap(sns.color_palette('ch:start=.2, rot=-.3', as_cmap=True).colors[::-1])
            self.cmap_seq: ListedColormap = sns.color_palette('RdBu_r', as_cmap=True)
            self.cmap_sim: ListedColormap = sns.color_palette('viridis', as_cmap=True)
    
    @for_visualize
    def vis_prologue(self, inp_B3HW: torch.Tensor) -> None:
        if not self.enable: return
        self.inp_B3HW = inp_B3HW
        
        # self.bound_mask = get_boundary(self.patch_size, self.vis_needs_loss_BL)
        # todo: multi scale log
        # imgs = {}
        # denormed_inp = self.vgpt_wo_ddp.denormalize(self.ls_inp_B3HW)
        # bchw = denormed_inp
        # # mean = (self.bound_mask * denormed_inp).sum(dim=(2, 3), keepdim=True) / self.bound_mask.sum(dim=(2, 3), keepdim=True)  # BC11
        # # self.bound_mask = self.bound_mask * (1 - mean * 0.99)  # BCHW
        # # bchw = torch.where(self.bound_mask > 0, self.bound_mask, denormed_inp)
        # chw = torchvision.utils.make_grid(bchw, padding=2, pad_value=1, nrow=10)
        # imgs[f'1_gt'] = chw
        # if log_inp:
        #     tb_lg.log_image(f'1_gt', chw, step=start_ep)
        # tb_lg.flush()
        # return imgs
    
    def denormalize(self, BCHW):
        # BCHW = BCHW * self.data_s
        # BCHW += self.data_m
        return BCHW.add(1).mul_(0.5).clamp_(0, 1)
    
    @for_visualize
    def vis(self, tb_lg: misc.TensorboardLogger, ep: int, png_path: str) -> Tuple[float, float]:
        if not self.enable: return -1., -1.
        vis_stt = time.time()
        warnings.filterwarnings('ignore', category=DeprecationWarning)
        
        # get recon
        B = self.inp_B3HW.shape[0]
        with torch.inference_mode():
            rec_B3HW_ema = self.trainer.vae_ema.img_to_reconstructed_img(self.inp_B3HW)
            training = self.trainer.vae_wo_ddp.training
            self.trainer.vae_wo_ddp.eval()
            rec_B3HW = self.trainer.vae_wo_ddp.img_to_reconstructed_img(self.inp_B3HW)
            self.trainer.vae_wo_ddp.train(training)
            
            L1_ema = F.l1_loss(rec_B3HW_ema, self.inp_B3HW).item()
            L1 = F.l1_loss(rec_B3HW, self.inp_B3HW).item()
            Lpip_ema = self.trainer.lpips_loss(rec_B3HW_ema, self.inp_B3HW).item()
            Lpip = self.trainer.lpips_loss(rec_B3HW, self.inp_B3HW).item()
            diff_ema = (L1_ema + Lpip_ema) / 2
            diff = (L1 + Lpip) / 2
            ema_better = diff_ema < diff
        
        # calc loss for logging
        tb_lg.update(
            head='PT_viz', step=ep+1,
            Diff=diff, Diff_ema=diff_ema,
            L1rec=L1, L1rec_ema=L1_ema,
            Lpips=Lpip, Lpips_ema=Lpip_ema,
            z_ema_adv=diff - diff_ema
        )
        
        # viz
        H, W = rec_B3HW.shape[-2], rec_B3HW.shape[-1]
        cmp_grid = torchvision.utils.make_grid(self.denormalize(torch.cat((self.inp_B3HW, rec_B3HW_ema, rec_B3HW), dim=0)), padding=0, pad_value=1, nrow=B)
        tb_lg.log_image('Raw_RecEMA_Rec', cmp_grid, step=ep+1)
        if png_path:
            chw = cmp_grid.permute(1, 2, 0).mul_(255).cpu().numpy()
            chw = PImage.fromarray(chw.astype(np.uint8))
            if not chw.mode == 'RGB':
                chw = chw.convert('RGB')
            PImageDraw.Draw(chw).text((H//10, W//10), (f'EMA {ep+1}' if ema_better else f'SELF {ep+1}'), (10, 10, 10))
            chw.save(png_path)
        
        # dt = self.trainer.disc_wo_ddp.training
        # self.trainer.disc_wo_ddp.eval()
        # todo: 这个地方disc网络绝对是不要求梯度的状态，因为每个iter开始的时候，都是先disc要求，再disc不要求，再return该iter，换句话说，disc仅在forward内部会要求梯度
        # todo: vis
        # for (inp, rec, rec2) in zip(self.ls_inp_B3HW, ls_rec_B3HW, ls_rec_BCHW2): inp.requires_grad = rec.requires_grad = rec2.requires_grad = True
        # self.trainer.d_criterion(self.trainer.disc_wo_ddp( torch.cat(ls_inp + ls_rec1 + ls_rec2, dim=0) )).backward()
        # self.trainer.disc_wo_ddp.train(dt)
        
        # for rec in ls_rec_B3HW:
        #     # if inp.grad is not None:
        #     #     grad_i, grad_r = inp.grad.mean(dim=1), rec.grad.mean(dim=1)
        #     #     inp.requires_grad = rec.requires_grad = False
        #     #     del inp.grad, rec.grad
        #     #     inp.grad = rec.grad = None
        #     #     grad_i = grad_i.sub(grad_i.mean()).div_(grad_i.std()+1e-5).mul_(0.3).add_(0.5)
        #     #     grad_r = grad_r.sub(grad_r.mean()).div_(grad_r.std()+1e-5).mul_(0.3).add_(0.5)
        #     #     grad_i = torch.from_numpy(self.cmap_div(grad_i.cpu().numpy())[:, :, :, :3]).to(device=inp.device, dtype=inp.dtype).permute(0, 3, 1, 2)
        #     #     grad_r = torch.from_numpy(self.cmap_div(grad_r.cpu().numpy())[:, :, :, :3]).to(device=inp.device, dtype=inp.dtype).permute(0, 3, 1, 2)
        #     #     ls = [self.denormalize(inp), self.denormalize(rec), grad_i, grad_r]
        #     # else:
        #     ls = [self.denormalize(inp), self.denormalize(rec)]
        #
        #     tb_lg.log_image(f'A_{rec.shape[-2]}', torchvision.utils.make_grid(torch.cat(ls, dim=0), padding=1, pad_value=1, nrow=B), step=ep+1)
        #     if png_path: pngs.append(torchvision.utils.make_grid(torch.cat((
        #         F.interpolate(self.denormalize(inp), final_reso, mode='nearest'),
        #         F.interpolate(self.denormalize(rec), final_reso, mode='nearest'),
        #     ), dim=0), padding=1, pad_value=1, nrow=B))
        
        # self.trainer.vae_wo_ddp.vis_key_params(tb_lg, ep)
        # self.trainer.disc_wo_ddp.vis_key_params(tb_lg, ep)
        
        print(f'  [*] [vis]    {L1=:.3f}, {Lpip=:.3f}  |  {L1_ema=:.3f}, {Lpip_ema=:.3f}  cost={time.time()-vis_stt:.2f}s', force=True)
        
        warnings.resetwarnings()
        return min(diff, diff_ema)


# import numba as nb
# @nb.jit(nopython=True, nogil=True, fastmath=True)
def get_boundary(patch_size, needs_loss, boundary_wid=3):  # vis_img: BCHW, needs_loss: BL
    """
    get the boundary of `False`-value connected components on given boolmap `needs_loss`
    """
    B, L = needs_loss.shape
    hw = round(L ** 0.5)
    boolmap = (~needs_loss).view(B, 1, hw, hw)  # BL => B1hw
    boolmap = boolmap.repeat_interleave(patch_size, dim=2).repeat_interleave(patch_size, dim=3)  # B1hw => B1HW
    
    k_size = boundary_wid * 2 + 1
    conv_kernel = torch.ones(1, 1, k_size, k_size).to(boolmap.device)
    bound_mask = F.conv2d(boolmap.float(), conv_kernel, padding=boundary_wid)
    bound_mask = ((bound_mask - k_size ** 2).abs() < 0.1) ^ boolmap  # B1HW
    
    return bound_mask.float()
