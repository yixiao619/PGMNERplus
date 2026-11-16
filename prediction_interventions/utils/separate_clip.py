import torch

def clip_main_and_aux(model,max_norm):
    main=[];aux=[]
    for name,p in model.named_parameters():
        (aux if name.startswith('local_aux_head.') else main).append(p)
    return (torch.nn.utils.clip_grad_norm_(main,max_norm),
            torch.nn.utils.clip_grad_norm_(aux,max_norm))
