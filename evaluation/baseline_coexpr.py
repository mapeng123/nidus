# -*- coding: utf-8 -*-
"""共表达传播基线: 零阶基线, 不用Nidus
对37个有小鼠表型的基因: 预测net方向 = -corr(G, FORM markers) (成骨细胞, 真实细胞共表达)
与 mouse_z 符号比较, 算 concordance。
同时输出: 16,105 panel基因中, 在任一三种核心细胞型中检测率>=1% 的基因数 (手稿用)。
"""
import numpy as np, json, pandas as pd

BASE = "/root/autodl-tmp/nidus_bundle"
panel = open(f"{BASE}/bone_genes_v4.txt").read().split()
g2i = {g: i for i, g in enumerate(panel)}
null = json.load(open("/root/autodl-tmp/program_null_stats.json"))
FORM = [g for g in null["FORM"] if g in g2i]
RES = [g for g in null["RES"] if g in g2i]

pk = np.load(f"{BASE}/rows_pack.npz")
ptr, index, data, ct = pk["ptr"], pk["index"], pk["data"], pk["ct"]
# ct 是字符串数组还是编码?
print("ct dtype:", ct.dtype, "sample:", ct[:5])
cts = np.unique(ct)
print("cell types:", cts)

def cells_of(name):
    rows = np.where(ct == name)[0]
    return rows

def to_dense(rows, genes_idx):
    """从CSR重建 dense (n_cells x len(genes_idx)) 的bin值"""
    out = np.zeros((len(rows), len(genes_idx)), dtype=np.float32)
    pos = {g: j for j, g in enumerate(genes_idx)}
    for r_i, r in enumerate(rows):
        s, e = ptr[r], ptr[r + 1]
        idx = index[s:e]
        m = np.isin(idx, genes_idx)
        for ii in idx[m]:
            out[r_i, pos[ii]] = data[s + np.where(idx == ii)[0][0]]
    return out

m = pd.read_csv(f"{BASE}/morris37.csv")
GENES = [g for g in m.gene if g in g2i]
print("testable genes in panel:", len(GENES), "/", len(m))

rows_ob = cells_of("osteoblast")
print("osteoblast cells:", len(rows_ob))

need = [g2i[g] for g in GENES] + [g2i[g] for g in FORM] + [g2i[g] for g in RES]
X = to_dense(rows_ob, need)
nG = len(GENES)
Xg = X[:, :nG]
Xf = X[:, nG:nG + len(FORM)]
Xr = X[:, nG + len(FORM):]

def meancorr(A, B):
    """A: n x p, B: n x q -> p 个基因各自与 B 的均值Pearson"""
    A_ = A - A.mean(0); B_ = B - B.mean(0)
    sa = A_.std(0) + 1e-9; sb = B_.std(0) + 1e-9
    return (A_.T @ B_ / A.shape[0]) / np.outer(sa, sb)  # p x q

Cf = meancorr(Xg, Xf)  # 37 x len(FORM)
Cr = meancorr(Xg, Xr)
pred_net = -(Cf.mean(1)) + Cr.mean(1)  # 与FORM共表达高 -> KO后形成下降 -> net为负
res = []
for i, g in enumerate(GENES):
    mz = float(m[m.gene == g].mouse_z.iloc[0])
    conc = (np.sign(pred_net[i]) == np.sign(mz))
    res.append((g, pred_net[i], mz, conc))
res_df = pd.DataFrame(res, columns=["gene", "coexpr_pred_net", "mouse_z", "concordant"])
res_df.to_csv("/root/autodl-tmp/nidus_out/baseline_coexpr37.csv", index=False)
acc = res_df.concordant.mean()
print(f"coexpression-propagation baseline concordance: {res_df.concordant.sum()}/{len(res_df)} = {acc:.2f}")

# 检测率统计 (手稿X值): 三种核心细胞型中任一检测率>=1%
allgenes = np.arange(len(panel))
rate_max = np.zeros(len(panel), dtype=np.float32)
for name in ["osteoblast", "mesenchymal stem cell", "osteoclast"]:
    rows = cells_of(name)
    cnt = np.zeros(len(panel), dtype=np.int64)
    for r in rows:
        cnt[index[ptr[r]:ptr[r + 1]]] += 1
    rate = cnt / max(len(rows), 1)
    rate_max = np.maximum(rate_max, rate)
    print(name, "cells:", len(rows))
X_ = int((rate_max >= 0.01).sum())
print("DETECT_1PCT_ANY3CT:", X_, "/", len(panel))
with open("/root/autodl-tmp/nidus_out/panel_detect_stats.json", "w") as f:
    json.dump({"detect_1pct_any3ct": X_, "panel": len(panel)}, f)
print("BASELINE_DONE")
