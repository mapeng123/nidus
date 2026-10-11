#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""阴性对照KO + 基线对比 (回应审稿意见#3/#5/#10)
设计: 7个阳参靶基因(带文献锚点) vs 25个随机基因 vs 25个表达匹配基因
  - 每基因: 10种子 x 500细胞 硬KO, WT按(ct,seed)缓存复用
  - 指标: 锚点基因方向一致率(真靶 vs 对照), 全局脱靶幅度, mean-delta基线
输出: /root/autodl-tmp/nidus_out/negctrl_ko_10seed.json + negctrl_*.npz
"""
import numpy as np, torch, json, time, sys, os
sys.path.insert(0, "/root/autodl-tmp/nidus_bundle")
from nidus_light_lib import load_light
from nidus_kofix_lib import gen_ko

BASE = "/root/autodl-tmp/nidus_bundle"
OUT = "/root/autodl-tmp/nidus_out"
os.makedirs(OUT, exist_ok=True)
N_SEED = 10

TARGETS = {
    "LRP5":      {"ct": "osteoblast", "down": ["SP7", "BGLAP", "COL1A1", "ALPL"], "up": []},
    "SOST":      {"ct": "osteoblast", "down": ["SOST"], "up": ["WNT16", "CTNNB1"]},
    "TNFSF11":   {"ct": "osteoclast", "down": ["NFATC1", "CTSK", "ACP5"], "up": []},
    "TNFRSF11A": {"ct": "osteoclast", "down": ["NFATC1", "CTSK", "ACP5"], "up": []},
    "TNFRSF11B": {"ct": "osteoclast", "down": [], "up": ["NFATC1", "CTSK"]},
    "DKK1":      {"ct": "osteoblast", "down": [], "up": ["SP7", "BGLAP", "ALPL"]},
    "WNT16":     {"ct": "osteoblast", "down": ["SP7", "BGLAP"], "up": []},
}
ANCHOR_SET = {g for v in TARGETS.values() for gl in (v["down"], v["up"]) for g in gl}
CTS = ["osteoblast", "osteoclast", "mesenchymal stem cell"]

t0 = time.time()
env = load_light(BASE)
genes, gidx, lut = env["genes"], env["gidx"], env["lut"]
ct_vocab, z0p, gmean = env["ct_vocab"], env["z0_pack"], env["gmean"]
G = env["G"]
print(f"load {time.time()-t0:.0f}s", flush=True)

# ---- 对照基因选择 ----
rng = np.random.default_rng(42)
tgt_names = set(TARGETS)
exclude = tgt_names | ANCHOR_SET
# 表达匹配: 每个靶基因在其锚定细胞型中找4个表达量最接近的非锚点基因
matched = []
for t, v in TARGETS.items():
    m = gmean[v["ct"]]
    mt = m[gidx[t]]
    cand = [g for g in genes if g not in exclude and m[gidx[g]] > 0]
    sc = sorted(cand, key=lambda g: abs(np.log(m[gidx[g]] + 1e-3) - np.log(mt + 1e-3)))
    picked = [g for g in sc[:4] if g not in matched]
    matched.extend(picked)
matched = matched[:25]
# 随机:  osteoblast中表达>0.5的非锚点基因
mob = gmean["osteoblast"]
pool = [g for g in genes if g not in exclude and g not in matched and mob[gidx[g]] > 0.5]
randoms = list(rng.choice(pool, size=25, replace=False))

JOBS = []
for t, v in TARGETS.items():
    JOBS.append(("target", t, v["ct"]))
for g, (t, v) in zip(matched, list(TARGETS.items()) * 4):
    JOBS.append(("matched", g, v["ct"]))
for g in randoms:
    JOBS.append(("random", g, "osteoblast"))
print(f"targets=7 matched={len(matched)} random={len(randoms)} total={len(JOBS)}", flush=True)

# ---- WT 缓存 ----
wt_cache = {}
def get_wt(ct_name, s):
    key = (ct_name, s)
    if key not in wt_cache:
        cid = ct_vocab[ct_name]
        z0 = torch.from_numpy(z0p[f"{ct_name}|{s}|z0"]).to(env["dev"])
        n = len(z0p[f"{ct_name}|{s}|tgt"])
        ct_a = np.full(n, cid, dtype=np.int64)
        cd_a = z0p[f"{ct_name}|{s}|cd"]
        tgt = z0p[f"{ct_name}|{s}|tgt"]
        wt_cache[key] = lut[gen_ko(env, ct_a, cd_a, tgt, z0, ko_gene=None,
                                   seed=7000 + s).astype(int)].mean(0)
        print(f"WT {ct_name} s{s} done {time.time()-t0:.0f}s", flush=True)
    return wt_cache[key]

results = {"matched_genes": matched, "random_genes": randoms, "genes": {}}
for kind, gene, ct_name in JOBS:
    if gene not in gidx:
        continue
    pg, cid = gidx[gene], ct_vocab[ct_name]
    deltas = []
    for s in range(N_SEED):
        z0 = torch.from_numpy(z0p[f"{ct_name}|{s}|z0"]).to(env["dev"])
        n = len(z0p[f"{ct_name}|{s}|tgt"])
        ct_a = np.full(n, cid, dtype=np.int64)
        cd_a = z0p[f"{ct_name}|{s}|cd"]
        tgt = z0p[f"{ct_name}|{s}|tgt"]
        gen_k = gen_ko(env, ct_a, cd_a, tgt, z0, ko_gene=pg, seed=7000 + s)
        deltas.append(lut[gen_k.astype(int)].mean(0) - get_wt(ct_name, s))
    D = np.stack(deltas)
    mu = D.mean(0)
    rec = {"kind": kind, "ct": ct_name,
           "self_delta": float(mu[pg]),
           "self_delta_se": float(D.std(0)[pg] / np.sqrt(N_SEED)),
           "offtarget_abs": float(np.mean(np.abs(np.delete(mu, pg)))),
           "top_down": [genes[i] for i in np.argsort(mu)[:20]],
           "top_up": [genes[i] for i in np.argsort(-mu)[:20]]}
    if kind == "target":
        v = TARGETS[gene]
        anch = {}
        for dirc, gl in (("down", v["down"]), ("up", v["up"])):
            for g in gl:
                if g in gidx:
                    vals = D[:, gidx[g]]
                    anch[f"{dirc}:{g}"] = {
                        "mean": float(vals.mean()),
                        "dir_rate": float(np.mean(vals < 0)) if dirc == "down" else float(np.mean(vals > 0))}
        rec["anchor"] = anch
    else:
        # 对照基因: 记录其对所有锚点基因的扰动(应≈0)
        rec["anchor_abs"] = {g: float(abs(mu[gidx[g]])) for g in ANCHOR_SET if g in gidx}
    results["genes"][f"{kind}|{gene}|{ct_name}"] = rec
    np.savez_compressed(f"{OUT}/negctrl_{kind}_{gene}.npz", delta_mu=mu,
                        delta_se=D.std(0) / np.sqrt(N_SEED),
                        real_mean=z0p[f"{ct_name}|0|real_mean"])
    print(f"{kind}|{gene}|{ct_name} done {time.time()-t0:.0f}s self={mu[pg]:.3f}", flush=True)

json.dump(results, open(f"{OUT}/negctrl_ko_10seed.json", "w"), indent=1)
print("NEGCTRL_DONE", flush=True)
