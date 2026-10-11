#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""在新服务器(GPU)上从 rows_pack/gmean_rows 计算 z0_pack 和 gene_mean_by_ct"""
import numpy as np, torch, sys, time, os
sys.path.insert(0, "/root/autodl-tmp/nidus_bundle")
from nidus_light_lib import load_light

BASE = "/root/autodl-tmp/nidus_bundle"
t0 = time.time()
env = load_light(BASE, dev="cuda", with_packs=False)
G, G_PAD, lut = env["G"], env["G_PAD"], env["lut"]
cond_vocab = env["cond_vocab"]
NULL_COND = env["NULL_COND"]
print(f"load {time.time()-t0:.0f}s", flush=True)

def csr_to_dense(ptr, index, data, i0, i1):
    n = i1 - i0
    x = np.zeros((n, G), dtype=np.int8)
    for i in range(n):
        s, e = ptr[i0 + i], ptr[i0 + i + 1]
        x[i, index[s:e]] = data[s:e]
    return x

def z0_of(x):
    xp = torch.from_numpy(np.pad(x, ((0, 0), (0, G_PAD - G)))).to(env["dev"])
    with torch.no_grad():
        return env["model"].z0_token(xp).mean(0).cpu().numpy().astype(np.float32)

rp = np.load(f"{BASE}/rows_pack.npz", allow_pickle=True)
ptr, index, data = rp["ptr"], rp["index"], rp["data"]
cts, seeds, conds = rp["ct"], rp["seed"], rp["cond"]

pack = {}
for ct in np.unique(cts):
    for s in range(10):
        m = (cts == ct) & (seeds == s)
        idx = np.where(m)[0]
        if len(idx) == 0:
            continue
        x = csr_to_dense(ptr, index, data, idx[0], idx[-1] + 1)
        key = f"{ct}|{s}"
        pack[key + "|z0"] = z0_of(x)
        pack[key + "|tgt"] = (ptr[idx + 1] - ptr[idx]).astype(np.int64)
        pack[key + "|cd"] = np.array([cond_vocab.get(str(c), NULL_COND) for c in conds[idx]], dtype=np.int64)
        pack[key + "|real_mean"] = lut[x].mean(0).astype(np.float32)
        print(f"{key} ok n={len(idx)} {time.time()-t0:.0f}s", flush=True)
np.savez_compressed(f"{BASE}/z0_pack.npz", **pack)

gm = np.load(f"{BASE}/gmean_rows.npz", allow_pickle=True)
gptr, gindex, gdata, gct = gm["ptr"], gm["index"], gm["data"], gm["ct"]
gmean = {}
i0 = 0
for ct in np.unique(gct):
    idx = np.where(gct == ct)[0]
    x = csr_to_dense(gptr, gindex, gdata, idx[0], idx[-1] + 1)
    gmean[ct] = lut[x].mean(0).astype(np.float32)
    print(f"gmean {ct} ok {time.time()-t0:.0f}s", flush=True)
np.savez_compressed(f"{BASE}/gene_mean_by_ct.npz", **gmean)
print("PREP_NEW_DONE", flush=True)
