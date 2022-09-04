"""

Test versions of Restormer to differences in output due to code modifications.

"""

# -- misc --
import sys,tqdm,pytest,math,random
from pathlib import Path

# -- dict data --
import copy
from easydict import EasyDict as edict

# -- vision --
from PIL import Image

# -- testing --
import unittest
import tempfile

# -- linalg --
import torch as th
import numpy as np
from einops import rearrange,repeat

# -- data --
import data_hub

# -- package imports [to test] --
import dnls # supporting
from torchvision.transforms.functional import center_crop

# -- package imports [to test] --
import restormer
from restormer.utils.gpu_mem import print_gpu_stats,print_peak_gpu_stats
from restormer.utils.misc import rslice_pair

# -- check if reordered --
from scipy import optimize
MAX_NFRAMES = 85
DATA_DIR = Path("./data/")
SAVE_DIR = Path("./output/tests/test_denose_rgb/")
if not SAVE_DIR.exists():
    SAVE_DIR.mkdir(parents=True)

def set_seed(seed):
    random.seed(seed)
    th.manual_seed(seed)
    np.random.seed(seed)
    # th.use_deterministic_algorithms(True)

def pytest_generate_tests(metafunc):
    seed = 123
    set_seed(seed)
    # test_lists = {"ps":[3],"stride":[2],"dilation":[2],
    #               "top":[3],"btm":[57],"left":[7],"right":[57]}
    # test_lists = {"sigma":[50.],"ref_version":["ref","original"]}
    test_lists = {"sigma":[50.],"ref_version":["ref"]}
    for key,val in test_lists.items():
        if key in metafunc.fixturenames:
            metafunc.parametrize(key,val)

def test_augmented_fwd(sigma,ref_version):

    # -- params --
    device = "cuda:0"
    # vid_set = "sidd_rgb"
    # vid_name = "00"
    # dset = "val"
    vid_set = "set8"
    vid_name = "motorbike"
    verbose = False
    isize = "128_128"
    dset = "te"
    flow = False
    noise_version = "blur"

    # -- timer --
    timer = restormer.utils.timer.ExpTimer()

    # -- setup cfg --
    cfg = edict()
    cfg.dname = vid_set
    cfg.vid_name = vid_name
    cfg.isize = isize
    cfg.sigma = 30.
    cfg.nframes = 2
    cfg.device = device

    # -- video --
    data,loaders = data_hub.sets.load(cfg)
    groups = data[dset].groups
    indices = [i for i,g in enumerate(groups) if cfg.vid_name in g]
    index = indices[0]

    # -- unpack --
    sample = data[dset][index]
    region = sample['region']
    noisy,clean = sample['noisy'],sample['clean']
    noisy,clean = rslice_pair(noisy,clean,region)
    noisy,clean = noisy.to(device),clean.to(device)
    vid_frames = sample['fnums']
    noisy /= 255.
    # print("noisy.shape: ",noisy.shape)

    # -- flows --
    t,c,h,w = noisy.shape
    flows = edict()
    flows.fflow = th.zeros((t,2,h,w),device=noisy.device)
    flows.bflow = th.zeros((t,2,h,w),device=noisy.device)

    # -- original exec --
    model_gt = restormer.original.load_model(sigma,noise_version=noise_version)
    model_gt = model_gt.to(cfg.device)
    model_gt.eval()
    with th.no_grad():
        _ = model_gt(noisy).detach()
    timer.start("original")
    with th.no_grad():
        deno_gt = model_gt(noisy).detach()
    th.cuda.synchronize()
    timer.stop("original")

    # -- refactored exec --
    t,c,h,w = noisy.shape
    region = None#[0,t,0,0,h,w] if ref_version == "ref" else None
    # attn_mode = "window_original"
    # attn_mode = "window_dnls"
    attn_mode = "product_dnls"
    # attn_mode = "window_refactored"
    # model_te = restormer.original.load_model(sigma,noise_version=noise_version)
    model_te = restormer.augmented.load_model(sigma,attn_mode=attn_mode,
                                            stride=8,noise_version=noise_version)
    model_te = model_te.to(cfg.device)
    print("-"*10)
    model_te.eval()
    timer.start("aug")
    with th.no_grad():
        deno_te = model_te(noisy).detach()
    th.cuda.synchronize()
    timer.stop("aug")

    # -- viz --
    print(timer)
    if verbose:
        print(deno_gt[0,0,:3,:3])
        print(deno_te[0,0,:3,:3])

    # -- viz --
    diff_s = th.abs(deno_gt - deno_te)# / (deno_gt.abs()+1e-5)
    print(diff_s.max())
    diff_s /= diff_s.max()
    print("diff_s.shape: ",diff_s.shape)
    dnls.testing.data.save_burst(diff_s[:3],SAVE_DIR,"diff")
    dnls.testing.data.save_burst(deno_gt[:3],SAVE_DIR,"deno_gt")
    dnls.testing.data.save_burst(deno_te[:3],SAVE_DIR,"deno_te")

    # -- test --
    error = th.abs(deno_gt - deno_te).mean().item()
    if verbose: print("error: ",error)
    assert error < 1e-5


def test_augmented_bwd(sigma,ref_version):

    # -- params --
    device = "cuda:0"
    # vid_set = "sidd_rgb"
    # vid_name = "00"
    # dset = "val"
    vid_set = "set8"
    vid_name = "motorbike"
    verbose = True
    isize = "96_96"
    dset = "te"
    flow = False
    noise_version = "blur"

    # -- timer --
    timer = restormer.utils.timer.ExpTimer()

    # -- setup cfg --
    cfg = edict()
    cfg.dname = vid_set
    cfg.vid_name = vid_name
    cfg.isize = isize
    cfg.sigma = 30.
    cfg.nframes = 2
    cfg.device = device

    # -- video --
    data,loaders = data_hub.sets.load(cfg)
    groups = data[dset].groups
    indices = [i for i,g in enumerate(groups) if cfg.vid_name in g]
    index = indices[0]

    # -- unpack --
    sample = data[dset][index]
    region = sample['region']
    noisy,clean = sample['noisy'],sample['clean']
    noisy,clean = rslice_pair(noisy,clean,region)
    noisy,clean = noisy.to(device),clean.to(device)
    vid_frames = sample['fnums']
    noisy /= 255.

    # -- flows --
    t,c,h,w = noisy.shape
    flows = edict()
    flows.fflow = th.zeros((t,2,h,w),device=noisy.device)
    flows.bflow = th.zeros((t,2,h,w),device=noisy.device)

    # -- original exec --
    model_gt = restormer.original.load_model(sigma,noise_version=noise_version)
    model_gt = model_gt.to(cfg.device)
    params_require_grad(model_gt)
    timer.start("original")
    deno_gt = model_gt(noisy.clone())
    th.cuda.synchronize()
    timer.stop("original")

    # -- refactored exec --
    t,c,h,w = noisy.shape
    region = None#[0,t,0,0,h,w] if ref_version == "ref" else None
    # attn_mode = "window_original"
    attn_mode = "window_refactored"
    # attn_mode = "window_dnls"
    # model_te = restormer.original.load_model(sigma,noise_version=noise_version)
    # model_te.train()
    model_te = restormer.augmented.load_model(sigma,attn_mode=attn_mode,
                                              stride=8,noise_version=noise_version)
    model_te = model_te.to(cfg.device)
    params_require_grad(model_te)
    # model_te.train()
    timer.start("aug")
    deno_te = model_te(noisy.clone())
    th.cuda.synchronize()
    timer.stop("aug")

    # -- viz --
    print(timer)
    if verbose:
        print(deno_gt[0,0,:3,:3])
        print(deno_te[0,0,:3,:3])

    # -- backward --
    th.autograd.backward(deno_gt,clean)
    th.autograd.backward(deno_te,clean)
    # params_cmp(model_gt,model_te)

    # -- get grads --
    _,params_gt = pack_params(model_gt)
    names,params_te = pack_params(model_te)

    # -- viz --
    diff = th.abs(params_gt - params_te)/(params_gt.abs()+1e-5)
    args0 = th.where(params_gt.abs()>1.)
    args = th.where(diff[args0]>1.)
    print(params_gt[args0][args])
    print(params_te[args0][args])

    # -- compare --
    diff = th.abs(params_gt - params_te)/(params_gt.abs()+1e-5)
    error = diff.mean().item()
    if verbose: print("error: ",error)
    assert error < 1e-4

    args = th.where(params_gt.abs() > 1.)
    error = diff[args].max().item()
    if verbose: print("error: ",error)
    assert error < 1e-2

def params_cmp(model_gt,model_te):
    params_cmp_names(model_gt,model_te)
    params_cmp_sizes(model_gt,model_te)
    # exit(0)

def params_cmp_names(model_gt,model_te):
    names_gt = []
    for name,param in model_gt.named_parameters():
        grad = param.grad
        if grad is None: continue

        names_gt.append(name)
    names_te = []
    for name,param in model_te.named_parameters():
        grad = param.grad
        if grad is None: continue

        names_te.append(name)
    names_gt = set(names_gt)
    names_te = set(names_te)
    print(names_gt - names_te)
    print(names_te - names_gt)

def params_cmp_sizes(model_gt,model_te):
    names_gt = {}
    for name,param in model_gt.named_parameters():
        grad = param.grad
        if grad is None: continue
        names_gt[name] = np.array(list(grad.size()))
    names_te = {}
    for name,param in model_te.named_parameters():
        grad = param.grad
        if grad is None: continue
        names_te[name] = np.array(list(grad.size()))
    for name,size_gt in names_gt.items():
        neq = True
        if name in names_te:
            size_te = names_te[name]
            if np.abs(size_gt - size_te).sum() < 1e-10:
                neq = False
        if neq:
            print(name)

def pack_params(model):
    names,params = [],[]
    for name,param in model.named_parameters():
        grad = param.grad
        if grad is None: continue
        grad_f = grad.ravel()
        params.append(grad_f)
        N = len(grad_f)
        names.extend([name for _ in range(N)])
    params = th.cat(params)
    return names,params

def params_require_grad(model):
    """
    Set params to require grad AND
    turn off all dropout layers for comparison
    """
    for param in model.parameters():
        # print(dir(param))
        param.requires_grad_(True)

