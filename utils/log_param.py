from collections import defaultdict
from math import log10
from typing import Dict, List


def get_param_for_log(model_name_3letters: str, named_parameters) -> Dict[str, List[float]]:
    dists = defaultdict(list)
    
    for n, p in named_parameters:
        n: str
        if p.grad is None: continue
        post = 'B' if ('.bias' in n or '_bias' in n) else 'W'
        
        if 'gpt' in model_name_3letters:
            if 'word' in n: tag = '0-word'
            elif 'norm0_ve' in n: tag = '0-norm0_ve'
            elif 'norm0_cond' in n: tag = '0-norm0_cond'
            elif 'start' in n: tag, post = '1-start', 'T'
            elif 'class_emb' in n: tag, post = '1-cls_emb', 'W'
            elif 'cls_token' in n: tag, post = '1-cls', 'T'
            elif 'cfg_uncond' in n: tag, post = '1-cond_cfg', 'T'
            elif 'cond_sos' in n: tag, post = '1-cond_sos', 'W'
            elif 'text_proj_for_sos' in n: tag = '1-text_sos'
            elif 'text_proj_for_ca' in n: tag = '1-text_ca'
            
            elif 'ca_rpb' in n: tag, post = '2-ca_rpb', 'T'
            elif 'sa_rpb' in n: tag, post = '2-sa_rpb', 'T'
            elif 'start_p' in n or 'pos_start' in n: tag, post = '2-pos_st', 'T'
            elif 'abs_pos_embed' in n: tag, post = '2-pos_abs', 'T'
            elif 'pos_mlp' in n: tag = '2-pos_mlp'
            elif 'lvl_embed' in n: tag, post = '2-pos_lvl', 'T'
            elif 'pos_1LC' in n: tag, post = '2-pos_1LC', 'T'
            elif 'pos_task' in n: tag, post = '2-pos_task', 'T'
            
            elif 'get_affine_4num' in n: tag = '1-freq_aff'
            elif 'freq_proj' in n: tag, post = '1-freq_prj', 'W'
            elif 'task_token' in n: tag, post = '1-task', 'T'
            elif 'adaIN_elin' in n: tag = '4-aIN_elin'
            elif 'shared_ada_lin' in n: tag = '2-shared_ada_lin'
            elif 'ada_lin' in n: tag = '4-ada_lin'
            elif 'ada_gss' in n: tag, post = '4-ada_gss', 'T'
            elif 'ada_gamma' in n: tag, post = '4-aIN_elin', 'GA'
            elif 'ada_beta' in n: tag, post = '4-aIN_elin', 'BE'
            elif 'moe_bias' in n: tag, post = '4-moe_bias', 'B'
            
            elif 'scale_mul' in n: tag, post = '3-2-scale', 'LogMul'
            elif 'norm1' in n: tag = '3-1-norm1'
            elif 'sa.' in n or 'attn.' in n: tag = '3-2-sa'
            elif 'ca.' in n: tag = '3-2-ca'
            elif 'gamma1' in n: tag, post = '3-3-gam1', 'GA'
            elif 'ca_norm' in n: tag = '3-2-ca_norm'
            elif 'ca_gamma' in n: tag, post = '3-3-ca_gam', 'GA'
            
            elif 'norm2' in n: tag = '4-1-norm1'
            elif 'ffn.' in n: tag = '4-2-ffn'
            elif 'gamma2_last' in n: tag, post = '4-3-gam2-last', 'GA'
            elif 'gamma2' in n: tag, post = '4-3-gam2', 'GA'
            
            elif 'head_nm' in n: tag = '5-headnm'
            elif 'head0' in n: tag = '5-head0'
            elif 'head_bias' in n: tag = '5-head_b', 'B'
            elif 'head' in n: tag = '5-head'
            elif 'up' in n: tag = '5-up'
            
            else: tag = f'___{n}___'
        
        elif 'vae' in model_name_3letters:
            if 'encoder.' in n or 'decoder.' in n:
                i, j = (0, 'enc') if 'encoder.' in n else (7, 'dec')
                if 'conv_in' in n: tag = f'{0+i}-{j}_cin'
                elif 'down.' in n and '.block' in n: tag = f'{1+i}-{j}_res'
                elif 'down.' in n and '.downsample' in n: tag = f'{1+i}-{j}_cdown'
                elif 'down.' in n and '.attn' in n: tag = f'{1+i}-{j}_attn'
                elif 'up.' in n and '.block' in n: tag = f'{1+i}-{j}_res'
                elif 'up.' in n and '.upsample' in n: tag = f'{1+i}-{j}_cup'
                elif 'up.' in n and '.attn' in n: tag = f'{1+i}-{j}_attn'
                elif 'mid.' in n and '.block' in n: tag = f'{2+i}-{j}_mid_res'
                elif 'mid.' in n and '.attn' in n: tag = f'{2+i}-{j}_mid_at'
                elif 'norm_out' in n: tag = f'{3+i}-{j}_nout'
                elif 'conv_out' in n: tag = f'{3+i}-{j}_cout'
                else: tag = f'3-enc___{n}___'
            elif 'quant_conv' in n: tag = f'4-quan_pre'
            elif 'post_quant_conv' in n: tag = f'6-quan_post'
            elif 'quant_proj' in n: tag = f'5-0-quan_pre_proj'
            elif 'quant_resi' in n: tag = f'5-2-quan_post_resi'
            elif 'post_quant_proj' in n: tag = f'5-2-quan_post_proj'
            elif 'quant' in n and 'norm_scale' in n: tag = f'5-1-quan_norm_scale'
            elif 'quant' in n and 'embed' in n: tag = f'5-1-quan_emb'
            else:
                tag = f'uk___{n}___'
        
        elif 'disc' in model_name_3letters or 'dsc' in model_name_3letters:   # discriminator
            if 'dwt' in n: tag = '0-dwt'
            elif 'from' in n: tag = '0-from'
            elif 'resi' in n: tag = '0-resi'
            elif 'fpn' in n: tag = '1-fpn'
            elif 'down' in n: tag = '2-down'
            elif 'head_conv' in n: tag = '3-head_conv'
            elif 'head_cls' in n: tag = '4-head_cls'
            elif 'norm.' in n: tag = 'x_norm'
            elif 'head.' in n:  # DinoDisc
                tag = n.split('heads.')[-1][0]
                if p.ndim == 3: tag += '.conv1d'
                else: tag += '.other'
            else:   # StyleGanDisc
                tag = n.rsplit('.', maxsplit=1)[0]
                if p.ndim == 4: tag += '.conv'
                else: tag += '.other'
        
        else: tag = f'uk___{n}___'
        
        m = p.grad.norm().item()
        m = log10(m) if m > 1e-9 else -10
        dists[f'Gnorm_{model_name_3letters}.{tag}.{post}'].append(m)
        m = p.data.abs().mean().item()
        m = log10(m) if m > 1e-9 else -10
        dists[f'Para_{model_name_3letters}.{tag}.{post}'].append(m)
    
    return dists
