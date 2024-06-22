import math
from typing import List, Optional, Tuple, Union

import torch
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from utils import misc
from utils.log_param import get_param_for_log


class NullCtx:
    def __enter__(self):
        pass
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


class AmpOptimizer:
    def __init__(
        self,
        model_name_3letters: str, model_maybe_fsdp: Union[torch.nn.Module, FSDP], fp16: bool, bf16: bool, zero: int,
        optimizer: torch.optim.Optimizer, grad_clip: float, n_gradient_accumulation: int = 1,
    ):
        self.model_name_3letters = model_name_3letters
        self.model_maybe_fsdp = model_maybe_fsdp
        self.zero = zero
        self.enable_amp = fp16 or bf16
        self.using_fp16_rather_bf16 = fp16
        
        if self.enable_amp:
            self.amp_ctx = torch.autocast('cuda', enabled=True, dtype=torch.float16 if self.using_fp16_rather_bf16 else torch.bfloat16, cache_enabled=self.zero == 0)
            self.scaler = torch.cuda.amp.GradScaler(init_scale=2. ** 11, growth_interval=1000) if self.using_fp16_rather_bf16 else None # only fp16 needs a scaler
        else:
            self.amp_ctx = NullCtx()
            self.scaler = None
        
        self.optimizer = optimizer
        self.grad_clip = grad_clip
        self.early_clipping = self.grad_clip > 0 and not hasattr(optimizer, 'global_grad_norm')
        self.late_clipping = self.grad_clip > 0 and hasattr(optimizer, 'global_grad_norm')
        
        self.r_accu = 1 / n_gradient_accumulation   # r_accu == 1.0 / n_gradient_accumulation
    
    def backward_clip_step(
        self, stepping: bool, loss: torch.Tensor,
    ) -> Tuple[Optional[torch.Tensor], Optional[float]]:
        # backward
        loss = loss.mul(self.r_accu)   # r_accu == 1.0 / n_gradient_accumulation
        orig_norm = scaler_sc = None
        if self.scaler is not None:
            self.scaler.scale(loss).backward(retain_graph=False, create_graph=False)
        else:
            loss.backward(retain_graph=False, create_graph=False)
        # print('===' * 100)
        # for n, p in self.model_maybe_fsdp.named_parameters():
        #     if p.stride() != p.grad.stride():
        #         print(n)
        #         print(p.stride(), p.grad.stride())
        #         print(p.shape, p.grad.shape)
        #         print(p.is_contiguous(), p.grad.is_contiguous())
        #         print('*' * 50)
        # print('===' * 100)
        if stepping:
            if self.scaler is not None: self.scaler.unscale_(self.optimizer)
            if self.early_clipping:
                if self.zero:
                    orig_norm: Optional[torch.Tensor] = self.model_maybe_fsdp.clip_grad_norm_(self.grad_clip)
                else:
                    orig_norm: Optional[torch.Tensor] = torch.nn.utils.clip_grad_norm_(self.model_maybe_fsdp.parameters(), self.grad_clip)
            
            if self.scaler is not None:
                self.scaler.step(self.optimizer)
                scaler_sc: Optional[float] = self.scaler.get_scale()
                if scaler_sc > 65536.: # fp16 will overflow when >65536, so multiply 65536 could be dangerous
                    self.scaler.update(new_scale=65536.)
                else:
                    self.scaler.update()
                try:
                    scaler_sc = float(math.log2(scaler_sc))
                except Exception as e:
                    print(f'[scaler_sc = {scaler_sc}]\n' * 15, flush=True)
                    raise e
            else:
                self.optimizer.step()
            
            if self.late_clipping:
                orig_norm: Optional[torch.Tensor] = self.optimizer.global_grad_norm
            
            self.optimizer.zero_grad(set_to_none=True)
        
        return orig_norm, scaler_sc
    
    @torch.no_grad()
    def log_param(self, ep: int, tb_lg: misc.TensorboardLogger):
        if self.zero == 0:
            for name, values in get_param_for_log(self.model_name_3letters, self.model_maybe_fsdp.named_parameters()).items():
                values: List[float]
                if len(values) == 1:    # e.g., cls token will only have one value
                    values.append(values[0])
                tb_lg.log_tensor_as_distri(name, torch.tensor(values, dtype=torch.float32), step=ep+1)
    
    def state_dict(self):
        return {
            'optimizer': self.optimizer.state_dict()
        } if self.scaler is None else {
            'scaler': self.scaler.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }
    
    def load_state_dict(self, state, strict=True):
        if self.scaler is not None:
            try: self.scaler.load_state_dict(state['scaler'])
            except Exception as e: print(f'[fp16 load_state_dict err] {e}')
        self.optimizer.load_state_dict(state['optimizer'])
