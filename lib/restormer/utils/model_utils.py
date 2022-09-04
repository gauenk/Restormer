import math
import torch as th
import torch.nn as nn
import os
from collections import OrderedDict
import copy
ccopy = copy.copy
from einops import repeat

def freeze(model):
    for p in model.parameters():
        p.requires_grad=False

def unfreeze(model):
    for p in model.parameters():
        p.requires_grad=True

def is_frozen(model):
    x = [p.requires_grad for p in model.parameters()]
    return not all(x)

def save_checkpoint(model_dir, state, session):
    epoch = state['epoch']
    model_out_path = os.path.join(model_dir,"model_epoch_{}_{}.pth".format(epoch,session))
    th.save(state, model_out_path)

def load_checkpoint_qkv(model, weights):
    checkpoint = th.load(weights)
    state_dict = checkpoint["state_dict"]
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        # -- standard mod --
        name = k[7:] if 'module.' in k else k
        if not("attn.qkv" in name):
            new_state_dict[name] = v

        # -- mod qkv --
        if "attn.qkv" in name:
            if "to_q" in name:
                if "weight" in name:
                    new_state_dict[name] = v.data[:,:,None,None]
                elif "bias" in name:
                    new_state_dict[name] = v.data
            elif "to_kv" in name:
                if "weight" in name:
                    # -- shapes --
                    half = v.shape[0]//2

                    # -- create v --
                    name_v = ccopy(name)
                    name_v = name_v.replace("to_kv","to_k")
                    new_state_dict[name_v] = v[:half,:,None,None]

                    # -- create k --
                    name_k = ccopy(name)
                    name_k = name_k.replace("to_kv","to_v")
                    new_state_dict[name_k] = v[half:,:,None,None]

                if "bias" in name:
                    # -- shapes --
                    half = v.shape[0]//2

                    # -- create v --
                    name_v = ccopy(name)
                    name_v = name_v.replace("to_kv","to_k")
                    new_state_dict[name_v] = v[:half,...]

                    # -- create k --
                    name_k = ccopy(name)
                    name_k = name_k.replace("to_kv","to_v")
                    new_state_dict[name_k] = v[half:,...]
            else:
                print("What??")
                print(name)

    model.load_state_dict(new_state_dict)

def load_checkpoint(model, weights):
    checkpoint = th.load(weights)
    try:
        # model.load_state_dict(checkpoint["state_dict"])
        raise ValueError("")
    except Exception as e:
        # print(list(checkpoint.keys())
        # state_dict = checkpoint["state_dict"]
        # new_state_dict = OrderedDict()
        # for k, v in state_dict.items():
        #     name = k[7:] if 'module.' in k else k
        #     new_state_dict[name] = v
        model.load_state_dict(checkpoint['params'])

def load_checkpoint_multigpu(model, weights):
    checkpoint = th.load(weights)
    state_dict = checkpoint["state_dict"]
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)

def load_start_epoch(weights):
    checkpoint = th.load(weights)
    epoch = checkpoint["epoch"]
    return epoch

def load_optim(optimizer, weights):
    checkpoint = th.load(weights)
    optimizer.load_state_dict(checkpoint['optimizer'])
    for p in optimizer.param_groups: lr = p['lr']
    return lr

def get_arch(opt):
    from model import UNet,Uformer,Uformer_Cross,Uformer_CatCross
    arch = opt.arch

    print('You choose '+arch+'...')
    if arch == 'UNet':
        model_restoration = UNet(dim=opt.embed_dim)
    elif arch == 'Uformer':
        model_restoration = Uformer(img_size=opt.train_ps,embed_dim=opt.embed_dim,win_size=opt.win_size,token_projection=opt.token_projection,token_mlp=opt.token_mlp)
    elif arch == 'Uformer16':
        model_restoration = Uformer(img_size=opt.train_ps,embed_dim=16,win_size=8,token_projection='linear',token_mlp='leff')
    elif arch == 'Uformer32':
        model_restoration = Uformer(img_size=opt.train_ps,embed_dim=32,win_size=8,token_projection='linear',token_mlp='leff')
    elif arch == 'Uformer_CatCross':
        model_restoration = Uformer_CatCross(img_size=opt.train_ps,embed_dim=opt.embed_dim,win_size=8,token_projection=opt.token_projection,token_mlp=opt.token_mlp)
    elif arch == 'Uformer_Cross':
        model_restoration = Uformer_Cross(img_size=opt.train_ps,embed_dim=opt.embed_dim,win_size=opt.win_size,token_projection=opt.token_projection,token_mlp=opt.token_mlp)
    else:
        raise Exception("Arch error!")

    return model_restoration

def remove_lightning_load_state(state):
    names = list(state.keys())
    for name in names:
        name_new = name.split(".")[1:]
        name_new = ".".join(name_new)
        state[name_new] = state[name]
        del state[name]

def temporal_chop(x,tsize,fwd_fxn,flows=None):
    nframes = x.shape[0]
    nslice = (nframes-1)//tsize+1
    x_agg = []
    for ti in range(nslice):
        ts = ti*tsize
        te = min((ti+1)*tsize,nframes)
        tslice = slice(ts,te)
        if flows:
            x_t = fwd_fxn(x[tslice],flows)
        else:
            x_t = fwd_fxn(x[tslice])
        x_agg.append(x_t)
    x_agg = th.cat(x_agg)
    return x_agg


def expand2square(timg,factor=16.0):
    t, _, h, w = timg.size()

    X = int(math.ceil(max(h,w)/float(factor))*factor)

    img = th.zeros(t,3,X,X).type_as(timg) # 3, h,w
    mask = th.zeros(t,1,X,X).type_as(timg)

    print(img.size(),mask.size())
    # print((X - h)//2, (X - h)//2+h, (X - w)//2, (X - w)//2+w)
    img[:,:, ((X - h)//2):((X - h)//2 + h),((X - w)//2):((X - w)//2 + w)] = timg
    mask[:,:, ((X - h)//2):((X - h)//2 + h),((X - w)//2):((X - w)//2 + w)].fill_(1)

    return img, mask


# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
#
#     Extracting Relevant Fields from Larger Dict
#
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def extract_search(cfg):
    fields = ["attn_mode","ws","wt","k","ps","pt","stride0",
              "stride1","dil","nbwd","rbwd","exact","bs",
              "noise_version"]
    model_cfg = {}
    for field in fields:
        if field in cfg:
            model_cfg[field] = cfg[field]
    return model_cfg

# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
#
#     Modifying Layers In-Place after Loading
#
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def reset_product_attn_mods(model):
    for name,param in model.named_parameters():
        if "relative_position_bias_table" in name:
            submod = model
            submods = name.split(".")
            for submod_i in submods:
                submod = getattr(submod,submod_i)
            submod.data = th.randn_like(submod.data).clamp(-1,1)/100.

def filter_product_attn_mods(model):
    for name,param in model.named_parameters():
        if "relative_position_bias_table" in name:
            submod = model
            submods = name.split(".")
            for submod_i in submods[:-1]:
                submod = getattr(submod,submod_i)
            setattr(submod,submods[-1],None)

def get_product_attn_params(model):
    params = []
    for name,param in model.named_parameters():
        if "relative_position_bias_table" in name:
            # delattr(model,name)
            param.requires_grad_(False)
            continue
        params.append(param)
    return params
