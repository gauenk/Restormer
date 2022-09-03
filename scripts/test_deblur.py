
# -- misc --
import os
import pprint
pp = pprint.PrettyPrinter(indent=4)

# -- vision --
import scipy.io
from PIL import Image

# -- linalg --
import numpy as np
import torch as th
from einops import rearrange,repeat

# -- data mngmnt --
from pathlib import Path
from easydict import EasyDict as edict

# -- data --
import data_hub

# -- optical flow --
import svnlb

# -- caching results --
import cache_io

# -- network --
import restormer
from restormer import lightning
from restormer.utils.misc import optional,rslice_pair
from restormer.utils.model_utils import temporal_chop,expand2square
from restormer.utils.metrics import compute_psnrs,compute_ssims

def run_exp(cfg):

    # -- set device --
    th.cuda.set_device(int(cfg.device.split(":")[1]))

    # -- init results --
    results = edict()
    results.psnrs = []
    results.ssims = []
    results.noisy_psnrs = []
    results.noisy_ssims = []
    results.deno_fns = []
    results.vid_frames = []
    results.vid_name = []
    results.timer_flow = []
    results.timer_deno = []

    # # -- network --
    # noise_version = "blur"
    # if cfg.model_type == "original":
    #     model = restormer.original.load_model(noise_version=noise_version).to(cfg.device)
    # elif cfg.model_type == "aug_refactored":
    #     attn_mode = "window_refactored"
    #     model = restormer.augmented.load_model(noise_version=noise_version,
    #                                          attn_mode=attn_mode,
    #                                          stride=cfg.stride,sb=-1)
    #                                          # ws=cfg.ws,wt=cfg.wt)#*1024)
    # elif cfg.model_type == "aug_dnls":
    #     attn_mode = "window_dnls"
    #     model = restormer.augmented.load_model(noise_version=noise_version,
    #                                          attn_mode=attn_mode,
    #                                          stride=cfg.stride)
    #                                          # ws=cfg.ws,wt=cfg.wt)
    # elif cfg.model_type == "product_dnls":
    #     attn_mode = "product_dnls"
    #     model = restormer.augmented.load_model(noise_version=noise_version,
    #                                          attn_mode=attn_mode,
    #                                          stride=cfg.stride)
    #                                          # ws=cfg.ws,wt=cfg.wt)
    # else:
    #     raise ValueError(f"Uknown model_type [{cfg.model_type}]")
    # model.eval()

    # -- load model --
    model_cfg = restormer.extract_search(cfg)
    model = restormer.load_model(model_cfg)
    load_checkpoint(model,cfg.use_train,cfg.model_type)
    imax = 255.

    # -- data --
    data,loaders = data_hub.sets.load(cfg)
    groups = data[cfg.dset].groups
    indices = [i for i,g in enumerate(groups) if cfg.vid_name in g]

    # -- optional filter --
    frame_start = optional(cfg,"frame_start",-1)
    frame_end = optional(cfg,"frame_end",-1)
    if frame_start >= 0 and frame_end >= 0:
        def fbnds(fnums,lb,ub): return (lb <= np.min(fnums)) and (ub >= np.max(fnums))
        indices = [i for i in indices if fbnds(data[cfg.dset].paths['fnums'][groups[i]],
                                               cfg.frame_start,cfg.frame_end)]

    # -- each subsequence with video name --
    for index in indices:

        # -- clean memory --
        th.cuda.empty_cache()

        # -- unpack --
        sample = data[cfg.dset][index]
        noisy,clean = sample['blur'],sample['sharp']
        noisy,clean = noisy.to(cfg.device),clean.to(cfg.device)
        vid_frames,region = sample['fnums'],optional(sample,'region',None)
        fstart = min(vid_frames)
        noisy,clean = rslice_pair(noisy,clean,region)
        print("[%d] noisy.shape: " % index,noisy.shape)
        print("[%d] noisy.shape: " % index,noisy.shape)

        # -- create timer --
        timer = uformer.utils.timer.ExpTimer()

        # -- optical flow --
        timer.start("flow")
        if cfg.flow == "true":
            noisy_np = noisy.cpu().numpy()
            flows = svnlb.compute_flow(noisy_np,cfg.sigma)
            flows = edict({k:th.from_numpy(v).to(cfg.device) for k,v in flows.items()})
        else:
            flows = None
        timer.stop("flow")

        # -- denoise --
        timer.start("deno")
        with th.no_grad():

            vshape = noisy.shape
            print("noisy.shape: ",noisy.shape)
            if cfg.model_type == "original" or True:
                noisy_sq,mask = expand2square(noisy)
            else: noisy_sq = noisy
            print("noisy_sq.shape: ",noisy_sq.shape)

            tsize = 2
            deno = temporal_chop(noisy_sq/imax,tsize,model)

            print("deno.shape: ",deno.shape)
            if cfg.model_type == "original" or True:
                deno = th.masked_select(deno,mask.bool()).reshape(*vshape)
            print("deno.shape: ",deno.shape)
            # t = noisy.shape[0]
            # deno = []
            # for ti in range(t):
            #     deno_t = model(noisy[[ti]]/imax)
            #     deno.append(deno_t)
            # deno = th.cat(deno)

            deno = th.clamp(deno,0.,1.)*imax
        timer.stop("deno")

        # -- save example --
        out_dir = Path(cfg.saved_dir) / cfg.dname / cfg.model_type / cfg.vid_name
        deno_fns = uformer.utils.io.save_burst(deno,out_dir,"deno",
                                               fstart=fstart,div=1.,fmt="np")
        deno_fns = uformer.utils.io.save_burst(deno,out_dir,"deno",
                                               fstart=fstart,div=1.,fmt="png")
        # uformer.utils.io.save_burst(clean,out_dir,"clean",
        #                             fstart=fstart,div=1.,fmt="np")
        # uformer.utils.io.save_burst(noisy,out_dir,"noisy",
        #                             fstart=fstart,div=1.,fmt="np")

        # -- psnr --
        noisy_psnrs = compute_psnrs(clean,noisy,div=imax)
        psnrs = compute_psnrs(clean,deno,div=imax)
        noisy_ssims = compute_ssims(clean,noisy,div=imax)
        ssims = compute_ssims(clean,deno,div=imax)
        print(noisy_psnrs)
        print(psnrs)

        # -- append results --
        results.noisy_psnrs.append(noisy_psnrs)
        results.psnrs.append(psnrs)
        results.noisy_ssims.append(noisy_ssims)
        results.ssims.append(ssims)
        results.deno_fns.append(deno_fns)
        results.vid_frames.append(vid_frames.numpy())
        results.vid_name.append([cfg.vid_name])
        for name,time in timer.items():
            results[name].append(time)

    return results

def load_checkpoint(model,use_train,model_type):
    load = use_train == "true"# or "product_dnls" == model_type
    croot = Path("output/checkpoints/")
    print(load)
    if load:
        print("loading!")
        # mpath = croot / "993b7b7f-0cbd-48ac-b92a-0dddc3b4ce0e-epoch=38.ckpt"
        mpath = croot / "067f3bb0-5f50-423a-a02f-6ef6bdaf0336-epoch=05.ckpt"
        state = th.load(str(mpath))['state_dict']
        lightning.remove_lightning_load_state(state)
        model.load_state_dict(state)

def default_cfg():
    # -- config --
    cfg = edict()
    cfg.nframes = 0
    cfg.frame_start = -1
    cfg.frame_end = -1
    cfg.saved_dir = "./output/saved_results/"
    cfg.num_workers = 1
    cfg.device = "cuda:0"
    cfg.sigma = 50. # use large sigma to approx real noise for optical flow
    return cfg

def main():

    # -- (0) start info --
    verbose = True
    pid = os.getpid()
    print("PID: ",pid)

    # -- get cache --
    cache_dir = ".cache_io"
    # cache_name = "test_rgb_net"
    cache_name = "gopro_bench"
    cache = cache_io.ExpCache(cache_dir,cache_name)
    # cache.clear()

    # -- get mesh --
    dnames = ["gopro"]
    dset = ["te"]
    vid_names = ["%02d" % x for x in np.arange(0,40)]
    vid_names = vid_names[1:2]

    # dnames = ["set8"]
    # vid_names = ["park_joy"]
    # dset = ["te"]

    flow = ["false"]
    ws,wt = [8],[0]
    isizes = ["none"]
    stride = [1]
    use_train = ["false"]
    model_type = ["aug_refactored","aug_dnls","product_dnls"]
    exp_lists = {"dname":dnames,"vid_name":vid_names,"dset":dset,
                 "flow":flow,"ws":ws,"wt":wt,"model_type":model_type,
                 "isize":isizes,"stride":stride,"use_train":use_train}
    exps_a = cache_io.mesh_pydicts(exp_lists) # create mesh

    # -- version 3 --
    exp_lists['use_train'] = ['true']
    exp_lists['model_type'] = ['product_dnls']
    exps_c = cache_io.mesh_pydicts(exp_lists) # create mesh

    # -- exps version 2 --
    exp_lists['ws'] = [-1]
    exp_lists['wt'] = [-1]
    exp_lists['flow'] = ["false"]
    exp_lists['use_train'] = ["false"]
    exp_lists['stride'] = [1]
    exp_lists['model_type'] = ['original']
    exps_b = cache_io.mesh_pydicts(exp_lists) # create mesh
    exps = exps_b + exps_a + exps_c

    # -- group with default --
    cfg = default_cfg()
    # cfg.isize = "256_256"
    cfg.nframes = 1
    cfg.frame_start = 0
    cfg.frame_end = cfg.frame_start + cfg.nframes - 1
    # cfg.isize = "256_256"
    cache_io.append_configs(exps,cfg) # merge the two

    # -- run exps --
    nexps = len(exps)
    for exp_num,exp in enumerate(exps):

        # -- info --
        if verbose:
            print("-="*25+"-")
            print(f"Running experiment number {exp_num+1}/{nexps}")
            print("-="*25+"-")
            pp.pprint(exp)

        # -- logic --
        uuid = cache.get_uuid(exp) # assing ID to each Dict in Meshgrid
        # cache.clear_exp(uuid)
        # if exp.model_type == "original":
        #     cache.clear_exp(uuid)
        # if exp.model_type == "aug_refactored":
        #     cache.clear_exp(uuid)
        # if exp.model_type == "aug_dnls":
        #     cache.clear_exp(uuid)
        # if exp.model_type == "product_dnls":
        #     cache.clear_exp(uuid)
        if exp.use_train == "true":
            cache.clear_exp(uuid)
        results = cache.load_exp(exp) # possibly load result
        if results is None: # check if no result
            exp.uuid = uuid
            results = run_exp(exp)
            cache.save_exp(uuid,exp,results) # save to cache

    # -- load results --
    records = cache.load_flat_records(exps)
    print(records[['model_type','use_train','psnrs']])

    for model_type,mdf in records.groupby("model_type"):
        for use_tr,tdf in mdf.groupby("use_train"):
            for stride,sdf in tdf.groupby("stride"):
                for vname,vdf in sdf.groupby("vid_name"):
                    ssims = np.stack(np.array(vdf['ssims'])).ravel()
                    psnrs = np.stack(np.array(vdf['psnrs'])).ravel()
                    dtimes = np.stack(np.array(vdf['timer_deno'])).ravel()
                    ssims_m = ssims.mean()
                    psnrs_m = psnrs.mean()
                    dtimes_m = dtimes.mean()
                    print(model_type,use_tr,vname,stride,psnrs_m,ssims_m,dtimes)

    exit(0)

    for model_type,mdf in records.groupby('model_type'):
        print(mdf['deno_fns'])
        prepare_sidd(mdf,model_type)
    exit(0)
    print(records)
    print(records.filter(like="timer"))
    print(records['psnrs'].mean())
    ssims = np.stack(np.array(records['ssims']))
    psnrs = np.stack(np.array(records['psnrs']))
    print(ssims)
    print(psnrs)
    print(psnrs.shape)
    print(psnrs.mean())
    print(ssims.mean())
    exit(0)

    # -- print by dname,sigma --
    for dname,ddf in records.groupby("dname"):
        # field = "internal_adapt_nsteps"
        field = "adapt_mtype"
        for adapt,adf in ddf.groupby(field):
            adapt_psnrs = np.stack(adf['adapt_psnrs'].to_numpy())
            print("adapt_psnrs.shape: ",adapt_psnrs.shape)
            print(adapt_psnrs)
            for cflow,fdf in adf.groupby("flow"):
                for ws,wsdf in fdf.groupby("ws"):
                    for wt,wtdf in wsdf.groupby("wt"):
                        print("adapt,ws,wt,cflow: ",adapt,ws,wt,cflow)
                        for sigma,sdf in wtdf.groupby("sigma"):
                            ave_psnr,ave_time,num_vids = 0,0,0
                            for vname,vdf in sdf.groupby("vid_name"):
                                print("vdf.psnrs.shape: ",vdf.psnrs.shape)
                                ave_psnr += vdf.psnrs[0].mean()
                                ave_time += vdf['timer_deno'].iloc[0]/len(vdf)
                                num_vids += 1
                            ave_psnr /= num_vids
                            ave_time /= num_vids
                            total_frames = len(sdf)
                            fields = (sigma,ave_psnr,ave_time,total_frames)
                            print("[%d]: %2.3f @ ave %2.2f sec for %d frames" % fields)


if __name__ == "__main__":
    main()
