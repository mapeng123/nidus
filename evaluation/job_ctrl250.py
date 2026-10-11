#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""49个对照基因 x 3细胞型 x 10种子 KO (N_CELL=100, 与Morris口径一致)
输出: /root/autodl-tmp/nidus_out/ctrl3ct_{gene}_{ct}.npz (delta_mu)
"""
import numpy as np, torch, json, time, sys, os
sys.path.insert(0, "/root/autodl-tmp/nidus_bundle")
from nidus_light_lib import load_light
from nidus_kofix_lib import gen_ko

BASE = "/root/autodl-tmp/nidus_bundle"
OUT = "/root/autodl-tmp/nidus_out"
os.makedirs(OUT, exist_ok=True)
N_SEED, N_CELL = 10, 100
CTS = ["osteoblast", "mesenchymal stem cell", "osteoclast"]
CT_TAG = {"osteoblast": "ob", "mesenchymal stem cell": "msc", "osteoclast": "oc"}

GENES = open(f"{BASE}/ctrl_genes_new201.txt").read().split()

t0 = time.time()
env = load_light(BASE)
gidx, lut, z0p, ct_vocab = env["gidx"], env["lut"], env["z0_pack"], env["ct_vocab"]
print(f"load {time.time()-t0:.0f}s", flush=True)

wt_cache = {}
def get_wt(ct_name, s):
    key = (ct_name, s)
    if key not in wt_cache:
        cid = ct_vocab[ct_name]
        z0 = torch.from_numpy(z0p[f"{ct_name}|{s}|z0"]).to(env["dev"])
        ct_a = np.full(N_CELL, cid, dtype=np.int64)
        cd_a = z0p[f"{ct_name}|{s}|cd"][:N_CELL]
        tgt = z0p[f"{ct_name}|{s}|tgt"][:N_CELL]
        wt_cache[key] = lut[gen_ko(env, ct_a, cd_a, tgt, z0, ko_gene=None,
                                   seed=7000 + s).astype(int)].mean(0)
        print(f"WT {ct_name} s{s} {time.time()-t0:.0f}s", flush=True)
    return wt_cache[key]

import os as _os
for gene in GENES:
    if all(_os.path.exists(f"{OUT}/ctrl250_{gene}_{t}.npz") for t in ["ob","msc","oc"]):
        print(f"EXISTS {gene}", flush=True); continue
    if gene not in gidx:
        print(f"SKIP {gene}", flush=True); continue
    pg = gidx[gene]
    for ct_name in CTS:
        cid = ct_vocab[ct_name]
        deltas = []
        for s in range(N_SEED):
            z0 = torch.from_numpy(z0p[f"{ct_name}|{s}|z0"]).to(env["dev"])
            ct_a = np.full(N_CELL, cid, dtype=np.int64)
            cd_a = z0p[f"{ct_name}|{s}|cd"][:N_CELL]
            tgt = z0p[f"{ct_name}|{s}|tgt"][:N_CELL]
            g_k = gen_ko(env, ct_a, cd_a, tgt, z0, ko_gene=pg, seed=7000 + s)
            deltas.append(lut[g_k.astype(int)].mean(0) - get_wt(ct_name, s))
        D = np.stack(deltas)
        tag = CT_TAG[ct_name]
        np.savez_compressed(f"{OUT}/ctrl250_{gene}_{tag}.npz",
                            delta_mu=D.mean(0), delta_se=D.std(0) / np.sqrt(N_SEED))
    print(f"{gene} 3ct done {time.time()-t0:.0f}s", flush=True)
print("CTRL250_DONE", flush=True)
