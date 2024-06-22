import datetime
import functools
import glob
import os
import subprocess
import sys
import threading
import time
from collections import defaultdict, deque
from typing import Iterator, List, Tuple

import numpy as np
import pytz
import torch
import torch.distributed as tdist

import dist
from utils import arg_util

os_system = functools.partial(subprocess.call, shell=True)
def echo(info):
    os_system(f'echo "[$(date "+%m-%d-%H:%M:%S")] ({os.path.basename(sys._getframe().f_back.f_code.co_filename)}, line{sys._getframe().f_back.f_lineno})=> {info}"')
def os_system_get_stdout(cmd):
    return subprocess.run(cmd, shell=True, stdout=subprocess.PIPE).stdout.decode('utf-8')
def os_system_get_stdout_stderr(cmd):
    cnt = 0
    while True:
        try:
            sp = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30)
        except subprocess.TimeoutExpired:
            cnt += 1
            print(f'[fetch free_port file] timeout cnt={cnt}')
        else:
            return sp.stdout.decode('utf-8'), sp.stderr.decode('utf-8')


def time_str(fmt='[%m-%d %H:%M:%S]'):
    return datetime.datetime.now(tz=pytz.timezone('Asia/Shanghai')).strftime(fmt)


class DistLogger(object):
    def __init__(self, lg):
        self._lg = lg
    
    @staticmethod
    def do_nothing(*args, **kwargs):
        pass
    
    def __getattr__(self, attr: str):
        return getattr(self._lg, attr) if self._lg is not None else DistLogger.do_nothing


class TensorboardLogger(object):
    def __init__(self, log_dir, filename_suffix):
        try: import tensorflow_io as tfio
        except: pass
        from torch.utils.tensorboard import SummaryWriter
        self.log_dir = log_dir
        self.writer = SummaryWriter(log_dir=log_dir, filename_suffix=filename_suffix)
        self.step = 0
    
    def set_step(self, step=None):
        if step is not None:
            self.step = step
        else:
            self.step += 1
    
    def loggable(self):
        return self.step == 0 or (self.step + 1) % 500 == 0
    
    def update(self, head='scalar', step=None, **kwargs):
        if step is None:
            step = self.step
            if not self.loggable(): return
        for k, v in kwargs.items():
            if v is None: continue
            if hasattr(v, 'item'): v = v.item()
            self.writer.add_scalar(f'{head}/{k}', v, step)
    
    def log_tensor_as_distri(self, tag, tensor1d, step=None):
        if step is None:
            step = self.step
            if not self.loggable(): return
        try:
            self.writer.add_histogram(tag=tag, values=tensor1d, global_step=step)
        except Exception as e:
            print(f'[log_tensor_as_distri writer.add_histogram failed]: {e}')
    
    def log_image(self, tag, img_chw, step=None):
        if step is None:
            step = self.step
            if not self.loggable(): return
        self.writer.add_image(tag, img_chw, step, dataformats='CHW')
    
    def flush(self):
        self.writer.flush()
    
    def close(self):
        print(f'[{type(self).__name__}] file @ {self.log_dir} closed')
        self.writer.close()


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """
    
    def __init__(self, window_size=30, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt
    
    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n
    
    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        tdist.barrier()
        tdist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]
    
    @property
    def median(self):
        return np.median(self.deque) if len(self.deque) else 0
    
    @property
    def avg(self):
        return sum(self.deque) / (len(self.deque) or 1)
    
    @property
    def global_avg(self):
        return self.total / (self.count or 1)
    
    @property
    def max(self):
        return max(self.deque) if len(self.deque) else 0
    
    @property
    def value(self):
        return self.deque[-1] if len(self.deque) else 0
    
    def time_preds(self, counts) -> Tuple[float, str, str]:
        remain_secs = counts * self.median
        return remain_secs, str(datetime.timedelta(seconds=round(remain_secs))), time.strftime("%Y-%m-%d %H:%M", time.localtime(time.time() + remain_secs))
    
    def __str__(self):
        return self.fmt.format(median=self.median, avg=self.avg, global_avg=self.global_avg, max=self.max, value=self.value)


class MetricLogger(object):
    def __init__(self, delimiter="  "):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter
        self.iter_end_t = time.time()
        self.log_iters = set()
    
    def update(self, **kwargs):
        # if it != 0 and it not in self.log_iters: return
        for k, v in kwargs.items():
            if v is None: continue
            if hasattr(v, 'item'): v = v.item()
            # assert isinstance(v, (float, int)), type(v)
            self.meters[k].update(v)
    
    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))
    
    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            if len(meter.deque):
                loss_str.append(
                    "{}: {}".format(name, str(meter))
                )
        return self.delimiter.join(loss_str)
    
    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()
    
    def add_meter(self, name, meter):
        self.meters[name] = meter
    
    def log_every(self, start_it, max_iters, itrt, print_freq, header=None):    # also solve logging & skipping iterations before start_it
        self.log_iters = set(np.linspace(0, max_iters-1, print_freq, dtype=int).tolist())
        self.log_iters.add(start_it)
        if not header:
            header = ''
        start_time = time.time()
        self.iter_end_t = time.time()
        self.iter_time = SmoothedValue(fmt='{avg:.4f}')
        self.data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(max_iters))) + 'd'
        log_msg = [
            header,
            '[{0' + space_fmt + '}/{1}]',
            'eta: {eta}',
            '{meters}',
            'time: {time}',
            'data: {data}'
        ]
        log_msg = self.delimiter.join(log_msg)
        
        if isinstance(itrt, Iterator) and not hasattr(itrt, 'preload') and not hasattr(itrt, 'set_epoch'):
            for it in range(start_it, max_iters):
                obj = next(itrt)
                self.data_time.update(time.time() - self.iter_end_t)
                yield it, obj
                self.iter_time.update(time.time() - self.iter_end_t)
                if it in self.log_iters:
                    eta_seconds = self.iter_time.global_avg * (max_iters - it)
                    eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                    print(log_msg.format(it, max_iters, eta=eta_string, meters=str(self), time=str(self.iter_time), data=str(self.data_time)), flush=True)
                self.iter_end_t = time.time()
        else:
            if isinstance(itrt, int): itrt = range(itrt)
            for it, obj in enumerate(itrt):
                if it < start_it:
                    self.iter_end_t = time.time()
                    continue
                self.data_time.update(time.time() - self.iter_end_t)
                yield it, obj
                self.iter_time.update(time.time() - self.iter_end_t)
                if it in self.log_iters:
                    eta_seconds = self.iter_time.global_avg * (max_iters - it)
                    eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                    print(log_msg.format(it, max_iters, eta=eta_string, meters=str(self), time=str(self.iter_time), data=str(self.data_time)), flush=True)
                self.iter_end_t = time.time()
        
        cost = time.time() - start_time
        cost_str = str(datetime.timedelta(seconds=int(cost)))
        print(f'{header}   Cost of this ep:      {cost_str}   ({cost / (max_iters-start_it):.3f} s / it)', flush=True)


class TouchingDaemonDontForgetToStartMe(threading.Thread):
    def __init__(self, files: List[str], sleep_secs: int, verbose=False):
        super().__init__(daemon=True)
        self.files = tuple(files)
        self.sleep_secs = sleep_secs
        self.is_finished = False
        self.verbose = verbose
        
        f_back = sys._getframe().f_back
        file_desc = f'{f_back.f_code.co_filename:24s}'[-24:]
        self.print_prefix = f' ({file_desc}, line{f_back.f_lineno:-4d}) @daemon@ '
    
    def finishing(self):
        self.is_finished = True
    
    def run(self) -> None:
        # stt, logged = time.time(), False
        kw = {}
        if dist.initialized(): kw['clean'] = True
        
        stt = time.time()
        if self.verbose: print(f'{time_str()}{self.print_prefix}[TouchingDaemon tid={threading.get_native_id()}] start touching {self.files} per {self.sleep_secs}s ...', **kw)
        while not self.is_finished:
            for f in self.files:
                if os.path.exists(f):
                    try:
                        os.utime(f)    # todo: ByteNAS oncall: change to open(...) for force-updating mtime (use strace to ensure an `open` system call)
                        fp = open(f, 'a')
                        fp.close()
                    except: pass
                    # else:
                    #     if not logged and self.verbose and time.time() - stt > 180:
                    #         logged = True
                    #         print(f'[TouchingDaemon tid={threading.get_native_id()}] [still alive ...]')
            time.sleep(self.sleep_secs)
        
        if self.verbose: print(f'{time_str()}{self.print_prefix}[TouchingDaemon tid={threading.get_native_id()}] finish touching after {time.time()-stt:.1f} secs {self.files} per {self.sleep_secs}s. ', **kw)


def glob_with_latest_modified_first(pattern, recursive=False):
    return sorted(glob.glob(pattern, recursive=recursive), key=os.path.getmtime, reverse=True)


def auto_resume(args: arg_util.Args, pattern='ckpt*.pth') -> Tuple[List[str], int, int, dict, dict]:
    info = []
    file = os.path.join(args.local_out_dir_path, pattern)
    all_ckpt = glob_with_latest_modified_first(file)
    if len(all_ckpt) == 0:
        info.append(f'[auto_resume] no ckpt found @ {file}')
        info.append(f'[auto_resume quit]')
        return info, 0, 0, {}, {}
    else:
        info.append(f'[auto_resume] load ckpt from @ {all_ckpt[0]} ...')
        ckpt = torch.load(all_ckpt[0], map_location='cpu')
        ep, it = ckpt['epoch'], ckpt['iter']
        info.append(f'[auto_resume success] resume from ep{ep}, it{it}')
        return info, ep, it, ckpt['trainer'], ckpt['args']


def create_npz_from_sample_folder(sample_folder: str):
    """
    Builds a single .npz file from a folder of .png samples. Refer to DiT.
    """
    import os, glob
    import numpy as np
    from tqdm import tqdm
    from PIL import Image
    
    samples = []
    pngs = glob.glob(os.path.join(sample_folder, '*.png')) + glob.glob(os.path.join(sample_folder, '*.PNG'))
    assert len(pngs) == 50_000, f'{len(pngs)} png files found in {sample_folder}, but expected 50,000'
    for png in tqdm(pngs, desc='Building .npz file from samples (png only)'):
        with Image.open(png) as sample_pil:
            sample_np = np.asarray(sample_pil).astype(np.uint8)
        samples.append(sample_np)
    samples = np.stack(samples)
    assert samples.shape == (50_000, samples.shape[1], samples.shape[2], 3)
    npz_path = f'{sample_folder}.npz'
    np.savez(npz_path, arr_0=samples)
    print(f'Saved .npz file to {npz_path} [shape={samples.shape}].')
    return npz_path
