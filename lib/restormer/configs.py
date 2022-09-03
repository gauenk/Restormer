"""

Default Configs for Training/Testing

"""

# -- easy dict --
import random
import numpy as np
import torch as th
from easydict import EasyDict as edict

def default_cfg():
    # -- config --
    cfg = edict()
    cfg.nframes = 0
    cfg.frame_start = 0
    cfg.frame_end = 0
    cfg.saved_dir = "./output/saved_results/"
    cfg.num_workers = 1
    cfg.device = "cuda:0"
    cfg.seed = 123
    cfg.noise_version = "blur"
    return cfg

def default_train_cfg():
    # -- config --
    cfg = edict()
    cfg.nframes = 5
    cfg.checkpoint_dir = "/home/gauenk/Documents/packages/uformer/output/checkpoints/"
    cfg.num_workers = 8
    cfg.device = "cuda:0"
    cfg.batch_size = 32
    cfg.batch_size_val = 1
    cfg.batch_size_te = 1
    cfg.saved_dir = "./output/saved_results/"
    cfg.device = "cuda:0"
    cfg.dname = "gopro_cropped"
    cfg.nsamples_at_testing = 2
    cfg.nsamples_tr = 0
    cfg.nsamples_val = 2
    cfg.rand_order_tr = True
    cfg.rand_order_val = False
    cfg.index_skip_val = 5
    cfg.log_root = "./output/log"
    cfg.cropmode = "rand"
    cfg.persistent_workers = True
    cfg.seed = 123
    return cfg

def set_seed(seed):
    random.seed(seed)
    th.manual_seed(seed)
    np.random.seed(seed)
