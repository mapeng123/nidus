# -*- coding: utf-8 -*-
"""扩充对照基因到250个: 49(已有) + 201(新)
新基因 = 150 个按496位点表达十分位分层匹配 + 51 个随机
排除: FORM/RES程序成员, SPP1, 已有49个对照, 496个测试位点本身
输出: /root/autodl-tmp/nidus_bundle/ctrl_genes_new201.txt
"""
import numpy as np, json, random

BASE = "/root/autodl-tmp/nidus_bundle"
panel = open(f"{BASE}/bone_genes_v4.txt").read().split()
null = json.load(open("/root/autodl-tmp/program_null_stats.json"))
EXCL = set(null["FORM"]) | set(null["RES"]) | {"SPP1"}
OLD49 = set(['SLC35E4','REL','NEK7','OGA','EPOP','NRGN','CTSW','STXBP4','TRIM72','KBTBD8',
             'SHC4','PADI2','PPM1K','ERN1','SSPN','BHLHE41','TTC13','ZBTB41','ENTPD6','EDIL3',
             'RRS1','GHR','GRIPAP1','CYB5D2','ADGRF5','RPL30','ITM2A','DNAJC3','SELENOK','CYB5R3',
             'SEC61G','RBP4','RTRAF','COL27A1','CD9','METRNL','NRN1','RPS21','CDC42','RPLP2',
             'MRPS21','S100A1','MDFI','NDUFS6','TPM2','UFM1','SOX4','CDV3','NKTR'])
LOCI = set(open(f"{BASE}/locus_genes_496.txt").read().split())
print("panel:", len(panel), "loci:", len(LOCI), "excl programs:", len(EXCL))

gm = np.load(f"{BASE}/gene_mean_by_ct.npz")
expr = np.mean(np.stack([gm[ct] for ct in gm.keys()]), axis=0)  # 3ct平均表达
g2i = {g: i for i, g in enumerate(panel)}

# 候选池: panel内、非程序、非旧对照、非位点、且在3ct中有可测表达
pool = [g for g in panel if g not in EXCL and g not in OLD49 and g not in LOCI
        and expr[g2i[g]] > 1e-4]
print("candidate pool:", len(pool))

# 位点表达十分位分布
loci_expr = np.array([expr[g2i[g]] for g in LOCI if g in g2i])
dec_edges = np.quantile(loci_expr, np.linspace(0, 1, 11))
loci_dec = np.clip(np.digitize(loci_expr, dec_edges[1:-1]), 0, 9)
loci_frac = np.bincount(loci_dec, minlength=10) / len(loci_dec)
print("loci decile dist:", loci_frac.round(3))

# 按位点十分位分布分层抽150个匹配基因
rng = random.Random(2024)
pool_expr = np.array([expr[g2i[g]] for g in pool])
pool_dec = np.clip(np.digitize(pool_expr, dec_edges[1:-1]), 0, 9)
matched = []
for ddec in range(10):
    want = int(round(150 * loci_frac[ddec]))
    if want == 0:
        continue
    cand_d = [pool[i] for i in range(len(pool)) if pool_dec[i] == ddec and pool[i] not in matched]
    take = min(want, len(cand_d))
    matched.extend(rng.sample(cand_d, take))
print("matched:", len(matched))

# 不足150则在邻近十分位补齐
if len(matched) < 150:
    rest = [g for g in pool if g not in matched]
    rest_sorted = sorted(rest, key=lambda g: abs(np.log1p(expr[g2i[g]]) - np.log1p(np.median(loci_expr))))
    matched.extend(rest_sorted[:150 - len(matched)])
    print("matched after fill:", len(matched))

# 51个随机
rest2 = [g for g in pool if g not in matched]
random51 = rng.sample(rest2, 51)

new201 = matched + random51
assert len(new201) == 201 and len(set(new201)) == 201
with open(f"{BASE}/ctrl_genes_new201.txt", "w") as f:
    f.write("\n".join(new201))
print("written ctrl_genes_new201.txt")

# 匹配质量检查: 位点 vs 新对照的表达分布 KS
from scipy.stats import ks_2samp
new_expr = np.array([expr[g2i[g]] for g in new201])
ks = ks_2samp(np.log1p(loci_expr), np.log1p(new_expr))
print(f"KS(log1p expr) loci vs new controls: stat={ks.statistic:.3f} P={ks.pvalue:.3f}")
print("loci median expr:", np.median(loci_expr), " new ctrl median:", np.median(new_expr))
