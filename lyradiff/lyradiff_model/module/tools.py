import torch
import numpy as np
import os, sys
import time

class LyraChecker:
    def __init__(self, dir_data, tol):
        self.dir_data = dir_data
        self.tol = tol
    
    def cmp(self, fpath1, fpath2="", tol=0):
        tolbk = self.tol
        if tol != 0:
            self.tol = tol
        if fpath2 == "":
            fpath2 = fpath1
            fpath1 += "_1"
            fpath2 += "_2"
        v1 = self.get_npy(fpath1) #np.load(os.path.join(self.dir_data, fpath1))
        v2 = self.get_npy(fpath2) #np.load(os.path.join(self.dir_data, fpath2))
        name = fpath1
        if ".npy" in fpath1:
            name = ".".join(os.path.basename(fpath1).split(".")[:-1])
        self._cmp_inner(v1, v2, name)
        self.tol = tolbk

    def _cmp_inner(self, v1, v2, name):
        print(v1.shape, v2.shape)
        if v1.shape != v2.shape:
            if v1.shape[1] == v2.shape[1]:
                v2 = v2.reshape([v2.shape[0], v2.shape[1], -1])
            else:
                v2 = torch.tensor(v2).permute(0, 3, 1, 2).numpy()
            print(v1.shape, v2.shape)
        self._check_data(name, v1, v2)
        print(np.size(v1))

    def _check_data(self, stage, x_out, x_gt):
        print(f"========== {stage} =============")
        print(x_out.shape, x_gt.shape)
        if np.allclose(x_gt, x_out, atol=self.tol):
            print(f"[OK] At {stage}, tol: {self.tol}")
        else:
            diff_cnt = np.count_nonzero(np.abs(x_gt - x_out)>self.tol)
            print(f"[FAIL]At {stage}, not aligned. tol: {self.tol}")
            print("    [INFO]Max diff: ", np.max(np.abs(x_gt - x_out)))
            print("    [INFO]Diff count: ", diff_cnt, ", ratio: ", round(diff_cnt/np.size(x_out), 2))
        print(f">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")


    def cmp_query(self, fpath1, fpath2):
        v1 = np.load(os.path.join(self.dir_data, fpath1))
        vk = np.load(os.path.join(self.dir_data, fpath1).replace("query", "key"))
        vv = np.load(os.path.join(self.dir_data, fpath1).replace("query", "value"))

        v2 = np.load(os.path.join(self.dir_data, fpath2))
        # print(v1.shape, v2.shape)
        q2 = v2[:,:,0,:,:].transpose([0,2,1,3])
        # print(v1.shape, q2.shape)
        self.check_data("query", v1, q2)
        # print(vk.shape, v2.shape)
        k2 = v2[:,:,1,:,:].transpose([0,2,1,3])
        self.check_data("key", vk, k2)
        vv2 = v2[:,:,2,:,:].transpose([0,2,1,3])
        # print(vv.shape, vv2.shape)
        self.check_data("value", vv, vv2)

    def _get_data_fpath(self, fname):
        fpath = os.path.join(self.dir_data, fname)
        if not fpath.endswith(".npy"):
            fpath += ".npy"
        return fpath

    def get_npy(self, fname):
        fpath = self._get_data_fpath(fname)
        return np.load(fpath)

        


class MkDataHelper:
    def __init__(self, data_dir="/data/home/kiokaxiao/data"):
        self.data_dir = data_dir

    def mkdata(self, subdir, name, shape, dtype=torch.float16):
        outdir = os.path.join(self.data_dir, subdir)
        os.makedirs(outdir, exist_ok=True)
        fpath = os.path.join(outdir, name+".npy")
        data = torch.randn(shape, dtype=torch.float16)
        np.save(fpath, data.to(dtype).numpy())
        return data

    def gen_out_with_func(self, func, inputs):
        output = func(inputs)
        return output

    def savedata(self, subdir, name, data):
        outdir = os.path.join(self.data_dir, subdir)
        os.makedirs(outdir, exist_ok=True)
        fpath = os.path.join(outdir, name+".npy")
        np.save(fpath, data.cpu().numpy())


class TorchSaver:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        os.makedirs(self.data_dir, exist_ok=True)
        self.is_save = True

    def save_v(self, name, v):
        if not self.is_save:
            return
        fpath = os.path.join(self.data_dir, name+"_1.npy")
        np.save(fpath, v.detach().cpu().numpy())

    def save_v2(self, name, v):
        if not self.is_save:
            return
        fpath = os.path.join(self.data_dir, name+"_1.npy")
        np.save(fpath, v.detach().cpu().numpy())

def timer_annoc(funct):
    def inner(*args,**kwargs):
        start = time.perf_counter()
        res = funct(*args,**kwargs)
        torch.cuda.synchronize()
        end = time.perf_counter()
        print("torch cost: ", end-start)
        return res
    return inner

def get_mem_use():
    f = os.popen("nvidia-smi | grep MiB" )
    line = f.read().strip()
    while "  " in line:
        line = line.replace("  ", " ")
    memuse = line.split(" ")[8]
    return memuse

if __name__ == "__main__":
    dir_data = sys.argv[1]
    fname_v1 = sys.argv[2]
    fname_v2 = sys.argv[3]
    tol = 0.01
    if len(sys.argv) > 4:
        tol = float(sys.argv[4])
    checker = LyraChecker(dir_data, tol)
    checker.cmp(fname_v1, fname_v2)