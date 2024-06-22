import json
import math
import os
import os.path as osp
import random
import re
import subprocess
import sys
import time
from collections import OrderedDict
from typing import Optional, Union

import numpy as np
import torch

try:
    from tap import Tap
except ImportError as e:
    print(f'`>>>>>>>> from tap import Tap` failed, please run:      pip3 install typed-argument-parser     <<<<<<<<', file=sys.stderr, flush=True)
    print(f'`>>>>>>>> from tap import Tap` failed, please run:      pip3 install typed-argument-parser     <<<<<<<<', file=sys.stderr, flush=True)
    time.sleep(5)
    raise e

import dist

class Args(Tap):
    exp_name: str       # MUST BE specified as `<tag>-<exp_name>`, e.g., vlip-exp1_cnn_lr1e-4
    bed: str            # MUST BE specified, Bytenas Experiment Directory
    resume: str = ''            # if specified, load this checkpoint; if not, load the latest checkpoint in bed (if existing)
    lpips_path: str = ''        # lpips VGG model weights
    dino_path: str = ''         # vit_small_patch16_224.pth model weights
    val_img_pattern: str = ''
    data: str = 'o_cc' # datasets, split by - or _, o: openimages, cc: cc12m, co: coco, fa: face data(ffhq+HumanArt+afhq+Internal), mj: midjourney, p: pinterest, px: (pexels+pixabay+unsplash)
    
    # speed-up: torch.compile
    zero: int = 0               # todo: FSDP zero 2/3
    compile_vae: int = 0        # torch.compile VAE; =0: not compile; 1: compile with 'reduce-overhead'; 2: compile with 'max-autotune'
    compile_disc: int = 0       # torch.compile discriminator; =0: not compile; 1: compile with 'reduce-overhead'; 2: compile with 'max-autotune'
    compile_lpips: int = 0      # torch.compile LPIPS; =0: not compile; 1: compile with 'reduce-overhead'; 2: compile with 'max-autotune'
    # speed-up: ddp
    ddp_static: bool = False    # whether to use static graph in DDP
    # speed-up: large batch
    vae_grad_ckpt: bool = False  # gradient checkpointing
    disc_grad_ckpt: bool = False # gradient checkpointing
    grad_accu: int = 1      # gradient accumulation
    prof: bool = False      # whether to do profile
    profall: bool = False   # whether to do profile on all ranks
    tos_profiler_file_prefix: str = ''
    
    # VAE: vitamin or cnn
    vae: str = 'cnn'        # 's', 'b', 'l' for using vitamin; 'cnn', 'conv', or '' for using CNN
    drop_path: float = 0.1  # following https://github.com/Beckschen/ViTamin/blob/76f1b1524ce03fcaa3449c7db678711f0961ebc2/ViTamin/open_clip/model_configs/ViTamin-L.json#L9
    # VAE: CNN encoder and CNN decoder
    ch: int = 160
    drop_out: float = 0.05
    # VAE: quantization layer
    vocab_size: int = 4096
    vocab_width: int = 32
    vocab_norm: bool = False
    vq_beta: float = 0.25           # commitment loss weight
    
    # DINO discriminator
    dino_depth: int = 12        # 12: use all layers
    dino_kernel_size: int = 9   # 9 is stylegan-T's setting
    disc_norm: str = 'sbn'      # gn: group norm, bn: batch norm, sbn: sync batch norm, hbn: hybrid sync batch norm
    disc_spec_norm: bool = True # whether to use SpectralNorm on Conv1ds in discriminator
    disc_aug_prob: float = 1.0  # discriminator augmentation probability (see models/vae/diffaug.py)
    disc_start_ep: float = 0    # start using disc loss for VAE after dep epochs; =0: will be automatically set to 0.22 * args.ep
    disc_warmup_ep: float = 0   # disc loss warm up epochs; =0: will be automatically set to 0.02 * args.ep
    reg: float = 0.0    # [NOT IMPLEMENTED YET] float('KEVIN_LOCAL' in os.environ)    # discriminator r1 regularization (grad penalty), =10
    reg_every: int = 4  # [NOT IMPLEMENTED YET]
    
    # initialization
    vae_init: float = -0.5  # <0: xavier_normal_(gain=abs(init)); >0: trunc_normal_(std=init)
    vocab_init: float = -1  # <0: uniform(-abs(init)*base, abs(init)*base), where base = 20/vocab_size; >0: trunc_normal_(std=init)
    disc_init: float = 0.02 # <0: xavier_normal_(gain=abs(init)); >0: trunc_normal_(std=init)
    
    # optimization
    fp16: bool = False
    bf16: bool = False
    vae_lr: float = 3e-4    # learning rate
    disc_lr: float = 3e-4   # learning rate
    vae_wd: float = 0.005   # weight decay
    disc_wd: float = 0.0005 # weight decay
    grad_clip: float = 10   # <=0 for not using grad clip
    ema: float = 0.9999     # ema ratio
    
    warmup_ep: float = 0    # lr warmup: epochs
    wp0: float = 0.005      # lr warmup: initial lr ratio
    sche: str = 'cos'       # lr schedule type
    sche_end: float = 0.3   # lr schedule: final lr ratio
    
    ep: int = 250               # epochs
    lbs: int = 0                # local batch size (exclusive to --bs) if this is specified, --bs will be ignored
    bs: int = 768               # global batch size (exclusive to --lbs)
    
    opt: str = 'adamw'      # adamw, lamb, or lion: https://cloud.tencent.com/developer/article/2336657?areaId=106001 lr=5e-5 (0.25x) wd=0.8 (8x); Lion needs a large bs to work
    oeps: float = 0
    fuse_opt: bool = torch.cuda.is_available()      # whether to use fused optimizer
    vae_opt_beta: str = '0.5_0.9'   # beta1, beta2 of optimizer
    disc_opt_beta: str = '0.5_0.9'  # beta1, beta2 of optimizer
    
    # gan optimization
    l1: float = 0.2     # L1 rec loss weight
    l2: float = 1.0     # L2 rec loss weight
    lp: float = 0.5     # lpips loss weight (WOULD BE *2 TO ADAPT LEGACY)
    lpr: int = 48       # only calculate lpips >= this image resolution
    ld: float = 0.4     # discriminator loss weight; if <0: NO ADAPTIVE WEIGHT
    le: float = 0.0     # VQ entropy loss weight
    gada: int = 1       # 0: local, 1: local+clamp(0.015, 1e4), 2: local+clamp(0.1, 10), 3: local+sqrt;   10, 11, 12, 13: with ema
    bcr: float = 4.     # balanced Consistency Regularization, used on small dataset with low reso, StyleSwin: 10.0
    bcr_cut: float = 0.2# cutout ratio (0.5: 50% width)
    dcrit: str = 'hg'   # hg hinge, sp softplus, ln linear
    # T: g=ln, d=hg;  XL: g=ln, d=hg;  Swin: g=sp, d=sp
    
    # other hps
    flash_attn: bool = True       # whether to use torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=False)
    
    # data
    subset: float = 1.0         # < 1.0 for use subset
    img_size: int = 256
    mid_reso: float = 1.125     # aug: first resize to mid_reso = 1.125 * data_load_reso, then crop to data_load_reso
    hflip: bool = False         # augmentation: horizontal flip
    workers: int = 8            # num workers; 0: auto, -1: don't use multiprocessing in DataLoader
    
    # debug
    local_debug: bool = 'KEVIN_LOCAL' in os.environ
    dbg_unused: bool = False
    dbg_nan: bool = False   # 'KEVIN_LOCAL' in os.environ
    
    # would be automatically set in runtime
    cmd: str = ' '.join(sys.argv[1:])  # [automatically set; don't specify this]
    branch: str = subprocess.check_output(f'git symbolic-ref --short HEAD 2>/dev/null || git rev-parse HEAD', shell=True).decode('utf-8').strip() or '[unknown]' # [automatically set; don't specify this]
    commit_id: str = subprocess.check_output(f'git rev-parse HEAD', shell=True).decode('utf-8').strip() or '[unknown]'  # [automatically set; don't specify this]
    commit_msg: str = (subprocess.check_output(f'git log -1', shell=True).decode('utf-8').strip().splitlines() or ['[unknown]'])[-1].strip()    # [automatically set; don't specify this]
    
    acc_all: float = None   # [automatically set; don't specify this]
    acc_real: float = None  # [automatically set; don't specify this]
    acc_fake: float = None  # [automatically set; don't specify this]
    last_Lnll: float = None # [automatically set; don't specify this]
    last_L1: float = None   # [automatically set; don't specify this]
    last_Ld: float = None   # [automatically set; don't specify this]
    last_wei_g: float = None# [automatically set; don't specify this]
    grad_boom: str = None   # [automatically set; don't specify this]
    diff: float = None      # [automatically set; don't specify this]
    diffs: str = ''         # [automatically set; don't specify this]
    diffs_ema: str = None   # [automatically set; don't specify this]
    cur_phase: str = ''         # [automatically set; don't specify this]
    cur_ep: str = ''            # [automatically set; don't specify this]
    cur_it: str = ''            # [automatically set; don't specify this]
    remain_time: str = ''       # [automatically set; don't specify this]
    finish_time: str = ''       # [automatically set; don't specify this]
    
    iter_speed: float = None    # [automatically set; don't specify this]
    img_per_day: float = None   # [automatically set; don't specify this]
    max_nvidia_smi: float = 0            # [automatically set; don't specify this]
    max_memory_allocated: float = None   # [automatically set; don't specify this]
    max_memory_reserved: float = None    # [automatically set; don't specify this]
    num_alloc_retries: int = None        # [automatically set; don't specify this]
    
    # environment
    local_out_dir_path: str = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'local_output')  # [automatically set; don't specify this]
    tb_log_dir_path: str = '...tb-...'  # [automatically set; don't specify this]
    tb_log_dir_online: str = '...tb-...'# [automatically set; don't specify this]
    log_txt_path: str = '...'           # [automatically set; don't specify this]
    last_ckpt_pth_bnas: str = '...'     # [automatically set; don't specify this]
    
    tf32: bool = True       # whether to use TensorFloat32
    device: str = 'cpu'     # [automatically set; don't specify this]
    seed: int = None        # seed
    deterministic: bool = False
    same_seed_for_all_ranks: int = 0     # this is only for distributed sampler
    def seed_everything(self):
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        if self.seed is not None:
            print(f'[in seed_everything] {self.deterministic=}', flush=True)
            if self.deterministic:
                torch.backends.cudnn.benchmark = False
                torch.backends.cudnn.deterministic = True
                torch.use_deterministic_algorithms(True)
                os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'
            seed = self.seed + dist.get_rank()*16384
            os.environ['PYTHONHASHSEED'] = str(seed)
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
    
    def get_different_generator_for_each_rank(self) -> Optional[torch.Generator]:   # for random augmentation
        if self.seed is None: return None
        g = torch.Generator()
        g.manual_seed(self.seed * dist.get_world_size() + dist.get_rank())
        return g
    
    def compile_model(self, m, fast):
        if fast == 0 or self.local_debug or not hasattr(torch, 'compile'):
            return m
        mode = {
            1: 'reduce-overhead',
            2: 'max-autotune',
            3: 'default',
        }[fast]
        print(f'[TORCH.COMPILE: {mode=}] compile {type(m)} ...', end='', flush=True)
        stt = time.perf_counter()
        m = torch.compile(m, mode=mode)
        print(f'     finished! ({time.perf_counter()-stt:.2f}s)', flush=True, clean=True)
        return m
    
    def state_dict(self, key_ordered=True) -> Union[OrderedDict, dict]:
        d = (OrderedDict if key_ordered else dict)()
        # self.as_dict() would contain methods, but we only need variables
        for k in self.class_variables.keys():
            if k not in {'device'}:     # these are not serializable
                d[k] = getattr(self, k)
        return d
    
    def load_state_dict(self, d: Union[OrderedDict, dict, str]):
        if isinstance(d, str):  # for compatibility with old version
            d: dict = eval('\n'.join([l for l in d.splitlines() if '<bound' not in l and 'device(' not in l]))
        for k in d.keys():
            try:
                setattr(self, k, d[k])
            except Exception as e:
                print(f'k={k}, v={d[k]}')
                raise e
    
    def load_state_dict_vae_only(self, d: Union[OrderedDict, dict, str]):
        for k in d.keys():
            if k not in {
                'vae',  # todo: fill more
            }:
                continue
            try:
                setattr(self, k, d[k])
            except Exception as e:
                print(f'k={k}, v={d[k]}')
                raise e
    
    @staticmethod
    def set_tf32(tf32: bool):
        if torch.cuda.is_available():
            torch.backends.cudnn.allow_tf32 = bool(tf32)
            torch.backends.cuda.matmul.allow_tf32 = bool(tf32)
            if hasattr(torch, 'set_float32_matmul_precision'):
                torch.set_float32_matmul_precision('high' if tf32 else 'highest')
                print(f'[tf32] [precis] torch.get_float32_matmul_precision(): {torch.get_float32_matmul_precision()}')
            print(f'[tf32] [ conv ] torch.backends.cudnn.allow_tf32: {torch.backends.cudnn.allow_tf32}')
            print(f'[tf32] [matmul] torch.backends.cuda.matmul.allow_tf32: {torch.backends.cuda.matmul.allow_tf32}')
    
    def __str__(self):
        s = []
        for k in self.class_variables.keys():
            if k not in {'device', 'dbg_ks_fp'}:     # these are not serializable
                s.append(f'  {k:20s}: {getattr(self, k)}')
        s = '\n'.join(s)
        return f'{{\n{s}\n}}\n'


def init_dist_and_get_args():
    for i in range(len(sys.argv)):
        if sys.argv[i].startswith('--local-rank=') or sys.argv[i].startswith('--local_rank='):
            del sys.argv[i]
            break
    args = Args(explicit_bool=True).parse_args(known_only=True)
    if args.local_debug:
        args.bed = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'bed')
        args.lpips_path = r'C:\Users\16333\Desktop\PyCharm\vgpt\_vae\lpips_with_vgg.pth'
        args.dino_path = r'C:\Users\16333\Desktop\PyCharm\vgpt\_vae\vit_small_patch16_224.pth'
        args.val_img_pattern = r'C:\Users\16333\Desktop\PyCharm\vgpt\_vae\val_imgs\v*'
        args.seed, args.deterministic = 1, True
        args.vae_init = args.disc_init = -0.5
        
        args.img_size = 64
        args.vae = 'cnn'
        args.ch = 32
        args.vocab_width = 16
        args.vocab_size = 4096
        args.disc_norm = 'gn'
        args.dino_depth = 3
        args.dino_kernel_size = 1
        args.vae_opt_beta = args.disc_opt_beta = '0.5_0.9'
        args.l2, args.l1, args.ll, args.le = 1.0, 0.2, 0.0, 0.1
    
    # warn args.extra_args
    if len(args.extra_args) > 0:
        print(f'======================================================================================')
        print(f'=========================== WARNING: UNEXPECTED EXTRA ARGS ===========================\n{args.extra_args}')
        print(f'=========================== WARNING: UNEXPECTED EXTRA ARGS ===========================')
        print(f'======================================================================================\n\n')
    
    # init torch distributed
    from utils import misc
    os.makedirs(args.local_out_dir_path, exist_ok=True)
    dist.init_distributed_mode(local_out_path=args.local_out_dir_path, timeout_minutes=30)
    
    # set env
    args.set_tf32(args.tf32)
    args.seed_everything()
    args.device = dist.get_device()
    
    if not torch.cuda.is_available() or (not args.bf16 and not args.fp16):
        args.flash_attn = False
    
    # update args: paths
    assert args.bed
    if args.exp_name not in args.bed:
        args.bed = osp.join(args.bed, f'{args.exp_name}')
    args.bed = args.bed.rstrip(osp.sep)
    os.makedirs(args.bed, exist_ok=True)
    if not args.lpips_path:
        args.lpips_path = f'{lyoko.BNAS_DATA}/ckpt_vae/lpips_with_vgg.pth'
    if not args.dino_path:
        args.dino_path = f'{lyoko.BNAS_DATA}/ckpt_vae/vit_small_patch16_224.pth'
    if not args.val_img_pattern:
        args.val_img_pattern = f'{lyoko.BNAS_DATA}/ckpt_vae/val_imgs/v*'
    if not args.tos_profiler_file_prefix.endswith('/'):
        args.tos_profiler_file_prefix += '/'
    
    # update args: bs, lr, wd
    if args.lbs == 0:
        args.lbs = max(1, round(args.bs / args.grad_accu / dist.get_world_size()))
    args.bs = args.lbs * dist.get_world_size()
    args.workers = min(args.workers, args.lbs)
    
    # args.lr = args.grad_accu * args.base_lr * args.glb_batch_size / 256
    
    # update args: warmup
    if args.warmup_ep == 0:
        args.warmup_ep = args.ep * 0.01
    if args.disc_start_ep == 0:
        args.disc_start_ep = args.ep * 0.2
    if args.disc_warmup_ep == 0:
        args.disc_warmup_ep = args.ep * 0.02
    
    # update args: paths
    args.log_txt_path = os.path.join(args.local_out_dir_path, 'log.txt')
    args.last_ckpt_pth_bnas = os.path.join(args.bed, f'ckpt-last.pth')
    
    _reg_valid_name = re.compile(r'[^\w\-+,.]')
    tb_name = _reg_valid_name.sub(
        '_',
        f'tb-{args.exp_name}'
        f'__{args.vae}'
        f'__b{args.bs}ep{args.ep}{args.opt[:4]}vlr{args.vae_lr:g}wd{args.vae_wd:g}dlr{args.disc_lr:g}wd{args.disc_wd:g}'
    )
    
    if dist.is_master():
        os.system(f'rm -rf {os.path.join(args.bed, "ready-node*")} {os.path.join(args.local_out_dir_path, "ready-node*")}')
    
    args.tb_log_dir_path = os.path.join(args.local_out_dir_path, tb_name)
    
    return args
