import numpy as np
import torch
from tqdm import trange
import re

def from_idx_to_patches(x, idx, ps):
    res = []
    for k in range(idx.shape[0]):
        i = idx[k, 0]
        j = i + ps
        l = idx[k, 1]
        m = l + ps
        res.append(x[i:j, l:m, ...])
    return torch.tensor(res)

def affinity(x):
    _, h, w, c = x.shape
    x_1 = torch.unsqueeze(x.reshape(-1, h * w, c), 2)
    x_2 = torch.unsqueeze(x.reshape(-1, h * w, c), 1)
    A = torch.norm(x_1 - x_2, dim=-1)
    krnl_width = torch.topk(A, k=A.shape[-1]).values
    krnl_width = torch.mean(krnl_width[:, :, (h * w) // 4], dim=1)
    krnl_width = krnl_width.reshape(-1, 1, 1)
    krnl_width = (torch.where((torch.eq(krnl_width,torch.zeros_like(krnl_width))), torch.ones_like(krnl_width), krnl_width))
    A=torch.exp(-(torch.div(A, krnl_width) ** 2))
    return A

def alpha(x, y):
    Ax = affinity(x)
    Ay = affinity(y)
    ps = int(Ax.shape[1] ** (0.5))
    Ag = torch.mean(torch.abs(Ax - Ay), dim=-1)
    alpha = Ag.reshape(-1, ps, ps)
    return alpha

def patch_indecies(i_max, j_max, ps, pstr=None):
    if pstr is None:
        pstr = ps
    idx = []
    for i in range(0, i_max - ps + 1, pstr):
        for j in range(0, j_max - ps + 1, pstr):
            idx.append([i, j])
    return torch.tensor(idx)

def patched_alpha(x, y, ps, pstr):
    bs = 500
    assert x.shape[:2] == y.shape[:2]
    y_max, x_max = x.shape[:2]

    Alpha = np.zeros((y_max, x_max), dtype=np.float32)
    covers = np.zeros((y_max, x_max), dtype=np.float32)

    idx = patch_indecies(y_max, x_max, ps, pstr)

    runs = idx.shape[0] // bs
    print("Runs: {}".format(runs))
    print("Leftovers: {}".format(idx.shape[0] % bs))
    done_runs = 0
    if idx.shape[0] % bs != 0:
        runs += 1
    for m in trange(runs):
        temp_idx = idx[:bs]
        batch_t1 = from_idx_to_patches(x, temp_idx, ps)
        batch_t2 = from_idx_to_patches(y, temp_idx, ps)
        al = np.array(alpha(batch_t1, batch_t2))
        for i in range(temp_idx.shape[0]):
            p = temp_idx[i, 0]
            q = p + ps
            r = temp_idx[i, 1]
            s = r + ps
            Alpha[p:q, r:s] += al[i]
            covers[p:q, r:s] += 1
        idx = idx[bs:]
    Alpha = np.divide(Alpha, covers)
    return Alpha
