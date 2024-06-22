import gc
import glob
import math
import os
import shutil
import subprocess
import sys
import time
import warnings
from collections import deque
from contextlib import nullcontext
from functools import partial
from typing import List, Optional, Tuple

import GPUtil
import colorama
import numpy as np
import torch
from torch.autograd.profiler import record_function
from torch.utils.data import DataLoader

import dist
from utils import arg_util, misc
from utils.data import build_dataset, pil_load
from utils.data_sampler import DistInfiniteBatchSampler


def create_tb_lg(args: arg_util.Args):
    tb_lg: misc.TensorboardLogger
    with_tb_lg = dist.is_master()
    if with_tb_lg:
        os.makedirs(args.tb_log_dir_path, exist_ok=True)
        # noinspection PyTypeChecker
        tb_lg = misc.DistLogger(misc.TensorboardLogger(log_dir=args.tb_log_dir_online, filename_suffix=f'_{misc.time_str("%m%d_%H%M")}'))
        tb_lg.flush()
    else:
        # noinspection PyTypeChecker
        tb_lg = misc.DistLogger(None)
    dist.barrier()
    return tb_lg


def maybe_auto_resume(args: arg_util.Args, pattern='ckpt*.pth') -> Tuple[List[str], int, int, str, List[Tuple[float, float]], dict, dict]:
    info = []
    resume = None
    if len(args.resume):
        resume = args.resume
        info.append(f'[auto_resume] load from args.resume @ {resume} ...')
    elif not args.local_debug:
        all_ckpt = lyoko.glob_with_latest_modified_first(os.path.join(args.bed, pattern))
        if len(all_ckpt) == 0:
            resume = resume
            info.append(f'[auto_resume] no ckpt found @ {pattern}')
            info.append(f'[auto_resume quit]')
        else:
            resume = all_ckpt[0]
            info.append(f'[auto_resume] auto load from @ {resume} ...')
        info.append(f'[auto_resume quit]')
    else:
        info.append(f'[auto_resume] disabled')
        info.append(f'[auto_resume quit]')
    
    if resume is None:
        return info, 0, 0, '[no acc str]', [], {}, {}
    
    try:
        ckpt = torch.load(resume, map_location='cpu')
    except Exception as e:
        info.append(f'[auto_resume] failed, {e} @ {resume}')
        return info, 0, 0, '[no acc str]', [], {}, {}
    
    dist.barrier()
    ep, it = (ckpt['epoch'], ckpt['iter']) if 'iter' in ckpt else (ckpt['epoch'] + 1, 0)
    eval_milestone = ckpt.get('milestones', [])
    info.append(f'[auto_resume success] resume from ep{ep}, it{it},    eval_milestone: {eval_milestone}')
    return info, ep, it, ckpt.get('acc_str', '[no acc str]'), eval_milestone, ckpt['trainer'], ckpt['args']


def build_things_from_args(args: arg_util.Args):
    # set seed
    auto_resume_info, start_ep, start_it, acc_str, eval_milestone, trainer_state, args_state = maybe_auto_resume(args, 'ckpt*.pth')
    args.load_state_dict_vae_only(args_state)
    args.diffs = ' '.join([f'{d:.3f}'[2:] for d in eval_milestone])   # args updated
    tb_lg = create_tb_lg(args)
    print(f'global bs={args.bs}, local bs={args.lbs}')
    print(f'initial args:\n{str(args)}')
    
    if start_ep == args.ep:
        print(f'[vlip] Training finished ({acc_str}), skipping ...\n\n')
        return args, tb_lg
    
    # build data
    # swin: -1~1, resize to (reso, reso) by LANCZOS
    # xl: -1~1,t
    if not args.local_debug:
        print(f'[build PT data] ...\n')
        dataset_train, val_transform = build_dataset(datasets_str=args.data, subset_ratio=args.subset, final_reso=args.img_size, mid_reso=args.mid_reso, hflip=args.hflip)
        ld_train = DataLoader(
            dataset=dataset_train, num_workers=args.workers, pin_memory=True,
            generator=args.get_different_generator_for_each_rank(), # worker_init_fn=worker_init_fn,
            batch_sampler=DistInfiniteBatchSampler(
                dataset_len=len(dataset_train), glb_batch_size=args.bs, same_seed_for_all_ranks=args.same_seed_for_all_ranks,
                shuffle=True, fill_last=True, rank=dist.get_rank(), world_size=dist.get_world_size(), start_ep=start_ep, start_it=start_it,
            ),
        )
        del dataset_train
        [print(l) for l in auto_resume_info]
        print(f'[dataloader multi processing] ...', end='', flush=True)
        stt = time.time()
        iters_train = len(ld_train) # 479   # len(ld_train)
        ld_train = iter(ld_train) # iter(range(20000000))
        # noinspection PyArgumentList
        print(f'     [dataloader multi processing](*) finished! ({time.time()-stt:.2f}s)', flush=True, clean=True)
        print(f'[dataloader] gbs={args.bs}, lbs={args.lbs}, iters_train={iters_train}')
    else:
        # dataset_mean, dataset_std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
        iters_train = ld_train = None
        from torchvision.transforms import transforms, InterpolationMode
        from utils.data import normalize_01_into_pm1
        val_transform = transforms.Compose([
            transforms.Resize(round(args.img_size*1.3), interpolation=InterpolationMode.LANCZOS),   # shorter edge would be the size
            transforms.CenterCrop((args.img_size, args.img_size)),
            transforms.ToTensor(),
            # transforms.Normalize(mean, std, inplace=True),
            normalize_01_into_pm1
        ])
        [print(l) for l in auto_resume_info]
    
    # import heavy packages after Dataloader object creation
    from torch.nn.parallel import DistributedDataParallel as DDP
    from models import build_vae_disc, VQVAE, DinoDisc
    from trainer import VAETrainer
    from utils.amp_opt import AmpOptimizer
    from utils.lpips import LPIPS
    from utils.lr_control import filter_params
    from utils import optimizer
    
    # build models
    vae_wo_ddp, disc_wo_ddp = build_vae_disc(args)
    vae_wo_ddp: VQVAE
    disc_wo_ddp: DinoDisc
    
    print(f'[PT] VAE model ({args.vae}) = {vae_wo_ddp}\n')
    if isinstance(disc_wo_ddp, DinoDisc):
        print(f'[PT] Disc model (frozen part) = {disc_wo_ddp.dino_proxy[0]}\n')
    print(f'[PT] Disc model (trainable part) = {disc_wo_ddp}\n\n')
    
    assert all(p.requires_grad for p in vae_wo_ddp.parameters())
    assert all(p.requires_grad for p in disc_wo_ddp.parameters())
    count_p = lambda m: f'{sum(p.numel() for p in m.parameters())/1e6:.2f}'
    print(f'[PT][#para] ' + ', '.join([f'{k}={count_p(m)}' for k, m in (
        ('VAE', vae_wo_ddp), ('VAE.enc', vae_wo_ddp.encoder), ('VAE.dec', vae_wo_ddp.decoder), ('VAE.quant', vae_wo_ddp.quantize)
    )]))
    print(f'[PT][#para] ' + ', '.join([f'{k}={count_p(m)}' for k, m in (
        ('Disc', disc_wo_ddp),
        # ('from_wave', disc_wo_ddp.ls_from_wavelet12c), ('resi', disc_wo_ddp.ls_resi),
        # ('fpn_conv', disc_wo_ddp.ls_fpn_conv), ('head', disc_wo_ddp.ls_head), ('down', disc_wo_ddp.ls_down),
        # ('glb_cls', disc_wo_ddp.glb_cls),
    )]) + '\n\n')
    
    # build optimizers
    optimizers: List[AmpOptimizer] = []
    for model_name, model_wo_ddp, opt_beta, lr, wd, clip in (('vae', vae_wo_ddp, args.vae_opt_beta, args.vae_lr, args.vae_wd, args.grad_clip), ('dis', disc_wo_ddp, args.disc_opt_beta, args.disc_lr, args.disc_wd, args.grad_clip)):
        if args.local_debug:
            lr, wd, clip = 5e-5, 5e-4, 20
        
        # sync model parameters
        for p in model_wo_ddp.parameters():
            if p.requires_grad:
                dist.broadcast(p.data, src_rank=0)
        ndim_dict = {name: para.ndim for name, para in model_wo_ddp.named_parameters() if para.requires_grad}
        
        # build optimizer
        nowd_keys = {
            'cls_token', 'start_token', 'task_token', 'cfg_uncond',
            'pos_embed', 'pos_1LC', 'pos_start', 'start_pos', 'lvl_embed',
            'gamma', 'beta',
            'ada_gss', 'moe_bias',
            'class_emb', 'embedding',
            'norm_scale',
        }
        names, paras, para_groups = filter_params(model_wo_ddp, ndim_dict, nowd_keys=nowd_keys)
        
        beta1, beta2 = map(float, opt_beta.split('_'))
        opt_clz = {
            'adam':  partial(torch.optim.AdamW, betas=(beta1, beta2), fused=args.fuse_opt),
            'adamw': partial(torch.optim.AdamW, betas=(beta1, beta2), fused=args.fuse_opt),
            'lamb':  partial(optimizer.LAMBtimm, betas=(beta1, beta2), max_grad_norm=clip), # eps=1e-7
            'lion':  partial(optimizer.Lion, betas=(beta1, beta2), max_grad_norm=clip),     # eps=1e-7
        }[args.opt]
        opt_kw = dict(lr=lr, weight_decay=0)
        if args.oeps: opt_kw['eps'] = args.oeps
        
        print(f'[vlip] optim={opt_clz}, opt_kw={opt_kw}\n')
        optimizers.append(AmpOptimizer(model_name, model_maybe_fsdp=None, fp16=args.fp16, bf16=args.bf16, zero=args.zero, optimizer=opt_clz(params=para_groups, **opt_kw), grad_clip=clip, n_gradient_accumulation=args.grad_accu))
        del names, paras, para_groups
    vae_optim, disc_optim = optimizers[0], optimizers[1]
    
    vae_wo_ddp, disc_wo_ddp = args.compile_model(vae_wo_ddp, args.compile_vae), args.compile_model(disc_wo_ddp, args.compile_disc)
    lpips_loss: LPIPS = args.compile_model(LPIPS(args.lpips_path).to(args.device), fast=args.compile_lpips)
    
    # distributed wrapper
    ddp_class = DDP if dist.initialized() else NullDDP
    vae: DDP = ddp_class(vae_wo_ddp, device_ids=[dist.get_local_rank()], find_unused_parameters=False, static_graph=args.ddp_static, broadcast_buffers=False)
    disc: DDP = ddp_class(disc_wo_ddp, device_ids=[dist.get_local_rank()], find_unused_parameters=False, static_graph=args.ddp_static, broadcast_buffers=False)
    
    vae_optim.model_maybe_fsdp = vae if args.zero else vae_wo_ddp
    disc_optim.model_maybe_fsdp = disc if args.zero else disc_wo_ddp
    
    trainer = VAETrainer(
        is_visualizer=dist.is_master(),
        vae=vae, vae_wo_ddp=vae_wo_ddp, disc=disc, disc_wo_ddp=disc_wo_ddp, ema_ratio=args.ema,
        dcrit=args.dcrit, vae_opt=vae_optim, disc_opt=disc_optim,
        daug=args.disc_aug_prob, lpips_loss=lpips_loss, lp_reso=args.lpr, wei_l1=args.l1, wei_l2=args.l2, wei_entropy=args.le, wei_lpips=args.lp, wei_disc=args.ld, adapt_type=args.gada, bcr=args.bcr, bcr_cut=args.bcr_cut, reg=args.reg, reg_every=args.reg_every,
        disc_grad_ckpt=args.disc_grad_ckpt,
        dbg_unused=args.dbg_unused, dbg_nan=args.dbg_nan
    )
    if trainer_state is not None and len(trainer_state):
        trainer.load_state_dict(trainer_state, strict=False)
    del vae, vae_wo_ddp, disc, disc_wo_ddp, vae_optim, disc_optim
    
    func = lambda x: os.path.basename(x) not in {'v3_008d0681123bcdf1.jpg', 'v4_00938fc5a0223cf4.jpg', 'v6_013afe5493a1a41c.jpg'}
    val_imgs = list(filter(func,  sorted(glob.glob(args.val_img_pattern))))
    
    if args.local_debug:
        inp = []
        for im in val_imgs:
            im = pil_load(im, args.img_size * 2)
            inp.append(val_transform(im))
        inp = torch.stack(inp, dim=0).to(args.device, non_blocking=True)
        print(f'[{inp.shape=}]')
        
        me = misc.MetricLogger(delimiter='  ')
        dbg_it = 599
        me.log_iters = {0, dbg_it}
        print(f'{trainer.vae_wo_ddp.encoder.conv_in.weight.data.view(-1)[:4]=}')
        args.seed_everything()
        trainer.train_step(
            ep=0, it=0, g_it=0, stepping=True, regularizing=False, metric_lg=me, logging_params=True, tb_lg=tb_lg,
            inp=inp,
            warmup_disc_schedule=0.0, fade_blur_schedule=0.8,
            maybe_record_function=nullcontext,
            args=args
        )
        trainer.train_step(
            ep=1, it=dbg_it, g_it=dbg_it, stepping=True, regularizing=True, metric_lg=me, logging_params=True, tb_lg=tb_lg,
            inp=inp,
            warmup_disc_schedule=0.8, fade_blur_schedule=0.0,
            maybe_record_function=nullcontext,
            args=args
        )
        print({k: meter.global_avg for k, meter in me.meters.items()})
        
        if isinstance(sys.stdout, dist.BackupStreamToFile) and isinstance(sys.stderr, dist.BackupStreamToFile):
            sys.stdout.close(), sys.stderr.close()
        exit(0)
    
    vis_dir, vis_file = '_vis_cached', f'{"vae_oi1in" if is_old_exp else "vae_mine"}_8x{args.img_size}.pth'
    vis_path = os.path.join(vis_dir, vis_file)
    
    print(f'[dld {vis_file}] before dld')
    if not os.path.exists(vis_path):
        if dist.is_local_master():
            misc.os_system(f'mkdir -p {vis_dir}; cp {lyoko.BNAS_DATA}/ckpt_vgpt/{vis_file} {vis_dir}/ >/dev/null 2>&1')
    dist.barrier()
    
    print(f'[dld {vis_file}] before load')
    if os.path.exists(vis_path):
        inp, label = torch.load(vis_path, map_location='cpu')
        inp, label = inp.to(args.device, non_blocking=True), label.to(args.device, non_blocking=True)
        print(f'[dld {vis_file}] {vis_path} successfully loaded.', flush=True)
    else:
        print(f'[dld {vis_file}] {vis_path} not found, now create and upload.', flush=True)
        inp, label = [], []
        for im in val_imgs:
            im = pil_load(im, args.img_size * 2)
            inp.append(val_transform(im))
            label.append(0)
        inp, label = torch.stack(inp, dim=0).to(args.device, non_blocking=True), torch.tensor(label, dtype=torch.long).to(args.device, non_blocking=True)
        if dist.is_master():
            torch.save([inp, label], vis_path)
            misc.os_system(f'mkdir -p {lyoko.BNAS_DATA}/ckpt_vgpt; cp {vis_path} {lyoko.BNAS_DATA}/ckpt_vgpt/ >/dev/null 2>&1')
    dist.barrier()
    
    del inp, label, val_transform
    return (
        tb_lg, trainer,
        start_ep, start_it, acc_str, eval_milestone, iters_train, ld_train,
    )


g_speed_ls = deque(maxlen=128)
def train_one_ep(ep: int, is_first_ep: bool, start_it: int, saver: CKPTSaver, args: arg_util.Args, tb_lg: misc.TensorboardLogger, ld_or_itrt, iters_train: int, trainer, logging_params_milestone):
    # import heavy packages after Dataloader object creation
    from trainer import VAETrainer
    from utils.lr_control import lr_wd_annealing
    trainer: VAETrainer
    
    step_cnt = 0
    me = misc.MetricLogger(delimiter='  ')
    [me.add_meter(x, misc.SmoothedValue(window_size=1, fmt='{value:.2g}')) for x in ['glr', 'dlr']]
    [me.add_meter(x, misc.SmoothedValue(window_size=1, fmt='{median:.2f} ({global_avg:.2f})')) for x in ['gnm', 'dnm']]
    for l in ['L1', 'NLL', 'Ld', 'Wg']:
        me.add_meter(l, misc.SmoothedValue(fmt='{median:.3f} ({global_avg:.3f})'))
    header = f'[Ep]: [{ep:4d}/{args.ep}]'
    
    touching_secs = 120
    if is_first_ep:
        warnings.filterwarnings('ignore', category=DeprecationWarning)
        warnings.filterwarnings('ignore', category=UserWarning)
    g_it, wp_it, max_it = ep * iters_train, args.warmup_ep * iters_train, args.ep * iters_train
    disc_start = args.disc_start_ep * iters_train
    disc_wp_it, disc_max_it = args.disc_warmup_ep * iters_train, max_it - disc_start
    
    doing_profiling = args.prof and is_first_ep and (args.profall or dist.is_master())
    maybe_record_function = record_function if doing_profiling else nullcontext
    trainer.vae_wo_ddp.maybe_record_function = maybe_record_function
    if args.zero:
        pref = 'hybrid' if args.hsdp else 'fsdp'
        if args.buck in {'0', '0.0', '0e0', '0.0e0'}:
            parallel = f'ep{ep}_{pref}{args.zero}_module_orig{args.fsdp_orig:d}'
        else:
            parallel = f'ep{ep}_{pref}{args.zero}_buk{args.buck}_orig{args.fsdp_orig:d}'
    else:
        parallel = 'ddp'
    if os.getenv('NCCL_CROSS_NIC', '0') == '1':
        parallel += f'_NIC1'
    profiling_name = f'{args.vae}_bs{args.bs}_{parallel}_gradckpt{args.vae_grad_ckpt:d}__GPU{dist.get_rank_str_zfill()}of{dist.get_world_size()}'
    
    profiler = None
    if doing_profiling:
        profiler = torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            schedule=torch.profiler.schedule(
                wait=40,
                warmup=3,
                active=2,
                repeat=1,
            ),
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
            on_trace_ready=TraceHandler('./', f'{profiling_name}.pt.trace.json', args.tos_profiler_file_prefix, args.bed)
        )
        profiler.start()
    
    last_t_perf = time.perf_counter()
    speed_ls: deque = g_speed_ls
    FREQ = min(50, iters_train//2-1)
    NVIDIA_IT_PLUS_1 = set(FREQ*i for i in (1, 2, 3, 4, 6, 8))
    PRINTABLE_IT_PLUS_1 = set(FREQ*i for i in (1, 2, 3, 4, 6, 8, 12, 16, 24, 32))
    for it, inp in me.log_every(start_it, iters_train, ld_or_itrt, max(10, iters_train // 1000), header):
        if (it+1) % FREQ == 0:
            speed_ls.append((time.perf_counter()-last_t_perf)/FREQ)
            iter_speed = float(np.median(speed_ls))
            img_per_sec = args.bs / iter_speed
            img_per_day = img_per_sec * 3600 * 24 / 1e6
            args.iter_speed, args.img_per_day = iter_speed, img_per_day
            
            if (it+1) in NVIDIA_IT_PLUS_1: args.max_nvidia_smi = max(args.max_nvidia_smi, max(gpu.memoryUsed for gpu in GPUtil.getGPUs()) / 1024)
            mem_infos_dict = torch.cuda.memory_stats()
            memory_allocated = round(mem_infos_dict['allocated_bytes.all.current']/1024**3, 2)
            memory_reserved = round(mem_infos_dict['reserved_bytes.all.current']/1024**3, 2)
            args.max_memory_allocated = round(mem_infos_dict['allocated_bytes.all.peak']/1024**3, 2)
            args.max_memory_reserved = round(mem_infos_dict['reserved_bytes.all.peak']/1024**3, 2)
            args.num_alloc_retries = mem_infos_dict['num_alloc_retries']
            if (ep <= 1 or ep == math.floor(args.disc_start_ep + 1e-4)) and (it+1) in PRINTABLE_IT_PLUS_1:
                tails = list(speed_ls)[-10:]
                print(
                    colorama.Fore.LIGHTCYAN_EX +
                    f"[profiling]  "
                    f"speed: {iter_speed:.3f} ({min(tails):.3f}~{max(tails):.2f}) sec/iter  |  "
                    f"{img_per_sec:.1f} imgs/sec  |  "
                    f"{img_per_day:.2f}M imgs/day  |  "
                    f"{img_per_day*(args.img_size//trainer.vae_wo_ddp.downsample_ratio)**2/1e3:.2f}B token/day  ||  "
                    f"Peak nvidia-smi: {args.max_nvidia_smi:.2f} GB  ||  "
                    f"PyTorch mem - "
                    f"alloc: {memory_allocated:.2f}  |  "
                    f"max_alloc: {args.max_memory_allocated:.2f}  |  "
                    f"reserved: {memory_reserved:.2f}  |  "
                    f"max_reserved: {args.max_memory_reserved:.2f}  |  "
                    f"num_alloc_retries: {args.num_alloc_retries}" + colorama.Fore.RESET + colorama.Back.RESET + colorama.Style.RESET_ALL,
                    flush=True
                )
            last_t_perf = time.perf_counter()
        
        if it < start_it: continue
        if is_first_ep and it == start_it: warnings.resetwarnings()
        
        if doing_profiling: profiler.step()
        
        with maybe_record_function('before_train'):
            inp = inp.to(args.device, non_blocking=True)
            
            g_it = ep * iters_train + it
            disc_g_it = g_it - disc_start
            args.cur_it = f'{it+1}/{iters_train}'
            min_glr, max_glr, min_gwd, max_gwd = lr_wd_annealing(args.sche, trainer.vae_opt.optimizer, args.vae_lr, args.vae_wd, g_it, wp_it, max_it, wp0=args.wp0, wpe=args.sche_end)
            if disc_g_it >= 0:
                min_dlr, max_dlr, min_dwd, max_dwd = lr_wd_annealing(args.sche, trainer.disc_opt.optimizer, args.disc_lr, args.disc_wd, disc_g_it, disc_wp_it, disc_max_it, wp0=args.wp0, wpe=args.sche_end)
            else:
                min_dlr = max_dlr = min_dwd = max_dwd = 0
            
            stepping = (g_it + 1) % args.grad_accu == 0
            step_cnt += int(stepping)
            warmup_disc_schedule = 0 if disc_g_it < 0 else min(1.0, disc_g_it / disc_wp_it)
            fade_blur_schedule = 0 if disc_g_it < 0 else min(1.0, disc_g_it / (disc_wp_it * 2))
            fade_blur_schedule = 1 - fade_blur_schedule
        
        grad_norm_g, scale_log2_g, grad_norm_d, scale_log2_d = trainer.train_step(
            ep=ep, it=it, g_it=g_it, stepping=stepping, regularizing=args.reg > 0 and (g_it % args.reg_every == 0),
            metric_lg=me, logging_params=stepping and step_cnt == 1 and (ep < 4 or ep in logging_params_milestone), tb_lg=tb_lg,
            inp=inp,
            warmup_disc_schedule=warmup_disc_schedule,
            fade_blur_schedule=fade_blur_schedule,
            maybe_record_function=maybe_record_function,
            args=args,
        )
        
        with maybe_record_function('after_train'):
            me.update(glr=max_glr, dlr=max_dlr)
            tb_lg.set_step(step=g_it)
            if tb_lg.loggable():
                if args.max_nvidia_smi > 0:
                    tb_lg.update(head='Profiling/speed', iter_cost=args.iter_speed, img_per_day=args.img_per_day)
                    tb_lg.update(head='Profiling/cuda_mem', max_nvi_smi=args.max_nvidia_smi, max_alloc=args.max_memory_allocated, max_reserve=args.max_memory_reserved, alloc_retries=args.num_alloc_retries)
                
                tb_lg.update(head='PT_opt_lr/lr_max', sche_glr=max_glr, sche_dlr=max_dlr)
                tb_lg.update(head='PT_opt_lr/lr_min', sche_glr=min_glr, sche_dlr=min_dlr)
                tb_lg.update(head='PT_opt_wd/wd_max', sche_gwd=max_gwd, sche_dwd=max_dwd)
                tb_lg.update(head='PT_opt_wd/wd_min', sche_gwd=min_gwd, sche_dwd=min_dwd)
                if scale_log2_g is not None:
                    tb_lg.update(head='PT_opt_grad/fp16', scale_log2_g=scale_log2_g, scale_log2_d=scale_log2_d)
                
                tb_lg.update(head='PT_opt_grad/grad', grad_norm_g=grad_norm_g, grad_norm_d=grad_norm_d)
                g_ratio = 1 if grad_norm_g is None else min(1.0, args.grad_clip / (grad_norm_g + 1e-7))
                d_ratio = 1 if grad_norm_d is None else min(1.0, args.grad_clip / (grad_norm_d + 1e-7))
                tb_lg.update(head='PT_opt_lr/lr_max', actu_glr=g_ratio*max_glr, actu_dlr=d_ratio*max_dlr)
                tb_lg.update(head='PT_opt_lr/lr_min', actu_glr=g_ratio*min_glr, actu_dlr=d_ratio*min_dlr)
    
    me.synchronize_between_processes()
    return {k: meter.global_avg for k, meter in me.meters.items()}, me.iter_time.time_preds(max_it - (g_it + 1) + (args.ep - ep) * 15)  # +15: other cost


def main_training():
    args: arg_util.Args = arg_util.init_dist_and_get_args()
    if args.dbg_unused:
        torch.autograd.set_detect_anomaly(True)
    
    ret = build_things_from_args(args)
    if len(ret) < 8:
        return ret
    (
        tb_lg, trainer,
        start_ep, start_it, acc_str, eval_milestone, iters_train, ld_train,
    ) = ret
    
    # import heavy packages after Dataloader object creation
    from trainer import VAETrainer
    ret: Tuple[
        misc.TensorboardLogger, VAETrainer,
        int, int, str, List[float], Optional[int], Optional[DataLoader],
    ]
    saver = CKPTSaver(dist.is_master(), eval_milestone)
    
    # train
    start_time, min_Lnll, min_Ld, disc_start = time.time(), 999., 999., False
    # seg8 = np.linspace(1, args.ep, 8+1, dtype=int).tolist()
    seg5 = np.linspace(1, args.ep, 5+1, dtype=int).tolist()
    # noinspection PyTypeChecker
    logging_params_milestone: List[int] = np.linspace(1, args.ep, 10+1, dtype=int).tolist()
    eval_milestone_ep = set(seg5[:])    # seg4
    vis_milestone_ep = set(seg5[:]) | set(x for x in (2, 4, 8, 16) if x <= args.ep)
    for x in [6, 12, 3, 24, 18, 48, 72, 96]:
        if len(vis_milestone_ep) < 10 and x <= args.ep:
            vis_milestone_ep.add(x)
    
    # save_milestone = list(range(5, args.ep, 2)) + [args.ep - 1]
    # for i, m in enumerate(save_milestone):
    #     if m != args.ep - 1 and m % 100 in {99, 0}:
    #         save_milestone[i] -= 1
    # save_milestone = set(save_milestone)
    # if 0 in save_milestone: save_milestone.remove(0)
    print(f'[PT milestones] eval={sorted(eval_milestone_ep)} vis={sorted(vis_milestone_ep)}')
    
    diff_t = torch.tensor([0.0, 0.0], dtype=torch.float32, device=args.device)
    trainer.vae_opt.log_param(ep=-1, tb_lg=tb_lg)
    trainer.disc_opt.log_param(ep=-1, tb_lg=tb_lg)
    time.sleep(3), gc.collect(), torch.cuda.empty_cache(), time.sleep(3)
    ep_lg = max(1, args.ep // 10) if args.ep <= 100 else max(1, args.ep // 20)
    for ep in range(start_ep, args.ep):
        if ep % ep_lg == 0 or ep == start_ep:
            print(f'[PT info] this exp is from ep{start_ep} it{start_it}, acc_str: {acc_str}, diffs: {args.diffs},        ==========>   bed: {args.bed}   h2: {args.tb_log_dir_online}  < ==========\n')
        if hasattr(ld_train, 'sampler') and hasattr(ld_train.sampler, 'set_epoch'):
            ld_train.sampler.set_epoch(ep)
            if 0 <= ep <= 3:
                print(f'[ld_train.sampler.set_epoch({ep})]')
        tb_lg.set_step(ep * iters_train)
        
        if args.flash_attn:
            sdp_kernel_select_ctx = torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=False)
        else:
            sdp_kernel_select_ctx = torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=True, enable_mem_efficient=False)
        if args.local_debug:
            sdp_kernel_select_ctx = nullcontext()
        with sdp_kernel_select_ctx:
            stats, (sec, remain_time, finish_time) = train_one_ep(
                ep, ep == start_ep, start_it if ep == start_ep else 0, saver, args, tb_lg, ld_train, iters_train, trainer, logging_params_milestone
            )
        
        Lnll, L1, Ld, wei_g = stats['NLL'], stats['L1'], stats['Ld'], stats['Wg']
        min_Lnll, min_Ld = min(min_Lnll, Lnll), min(min_Ld, min_Ld if Ld < 1e-7 else Ld)
        acc_real, acc_fake = stats.get('acc_real', -1), stats.get('acc_fake', -1)
        acc_all = (acc_real + acc_fake) * 0.5
        args.last_Lnll, args.last_L1, args.last_Ld, args.last_wei_g, args.acc_all, args.acc_real, args.acc_fake = Lnll, L1, Ld, wei_g, acc_all, acc_real, acc_fake
        if not math.isfinite(Lnll + Ld + L1 + wei_g):
            for n, v in zip(
                    ('Lnll', 'Ld', 'L1', 'wei_g'),
                    (Lnll, Ld, L1, wei_g),
            ):
                if not math.isfinite(v):
                    # noinspection PyArgumentList
                    print(f'[rk{dist.get_rank():02d}] {n} is {v}, stopping training!', force=True, flush=True)
            sys.exit(666)
        
        args.cur_phase = 'PT'
        args.cur_ep = f'{ep+1}/{args.ep}'
        args.remain_time, args.finish_time = remain_time, finish_time
        
        from torch.nn.parallel import DistributedDataParallel as DDP
        if isinstance(trainer.vae, DDP):
            vae_ddp_static = trainer.vae._get_ddp_logging_data().get('can_set_static_graph')
            disc_ddp_static = trainer.disc._get_ddp_logging_data().get('can_set_static_graph')
            tail = colorama.Fore.LIGHTGREEN_EX + f'  |  static_graph: vae={vae_ddp_static}, disc={disc_ddp_static}' + colorama.Fore.RESET + colorama.Back.RESET + colorama.Style.RESET_ALL
        else:
            tail = ''
        if ep > args.ep // 20:
            print(f'  [*] [ep{ep}]  Min Lnll: {min_Lnll:.3f},  Ld: {min_Ld:.3f},  Remain: {remain_time},  Finish: {finish_time}' + tail)
            tb_lg.update(head='PT_y_result', step=ep+1, min_Lnll=min_Lnll, min_Ld=None if min_Ld > 200 else min_Ld)
        else:
            print(f'  [*] [ep{ep}]  Remain: {remain_time},  Finish: {finish_time}' + tail)
        
        disc_start = acc_all >= 0
        if disc_start:
            kw = dict(L1rec=L1, Lnll=Lnll, Ld=Ld, wei_g=wei_g, acc_all=acc_all, acc_fake=acc_fake, acc_real=acc_real)
        else:
            kw = dict(L1rec=L1, Lnll=Lnll)
        tb_lg.update(head='PT_ep_loss', step=ep+1, **kw)
        tb_lg.update(head='PT_z_burnout', step=ep+1, rest_hours=round(sec / 60 / 60, 2))
        
        is_val_and_also_saving = (ep + 1) % 10 == 0 or (ep + 1) == args.ep
        if is_val_and_also_saving:
            print(f' [*] [ep{ep}]  (val {tot})  Lm: {L_mean:.4f}, Lt: {L_tail:.4f}, Acc m&t: {acc_mean:.2f} {acc_tail:.2f},  Val cost: {cost:.2f}s')
        
        if dist.is_local_master():
            local_out_ckpt = os.path.join(args.local_out_dir_path, 'ckpt-last.pth')
            local_out_ckpt_best = os.path.join(args.local_out_dir_path, 'ckpt-best.pth')
            print(f'[saving ckpt] ...', end='', flush=True)
            torch.save({
                'epoch':    ep+1,
                'iter':     0,
                'trainer':  trainer.state_dict(),
                'args':     args.state_dict(),
            }, local_out_ckpt)
            if best_updated:
                shutil.copy(local_out_ckpt, local_out_ckpt_best)
            print(f'     [saving ckpt](*) finished!  @ {local_out_ckpt}', flush=True, clean=True)
        dist.barrier()
    
    total_time = f'{(time.time() - start_time) / 60 / 60:.1f}h'
    print('\n\n')
    print(f'  [*] [finished]  Total Time: {total_time},   Lg: {min_Lnll:.3f},   Ld: {min_Ld:.3f}')
    print('\n\n')
    
    del iters_train, ld_train
    tb_lg.flush(); tb_lg.close()
    dist.barrier()


class NullDDP(torch.nn.Module):
    def __init__(self, module, *args, **kwargs):
        super(NullDDP, self).__init__()
        self.module = module
        self.require_backward_grad_sync = False
    
    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)


if __name__ == '__main__':
    try: main_training()
    finally:
        dist.finalize()
        if isinstance(sys.stdout, dist.BackupStreamToFile) and isinstance(sys.stderr, dist.BackupStreamToFile):
            sys.stdout.close(), sys.stderr.close()

