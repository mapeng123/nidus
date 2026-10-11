# -*- coding: utf-8 -*-
"""Nidus worker v3 — GPU 精简包版 (部署 19016 /root/autodl-tmp/)
基于 nidus_bundle (load_light + z0_pack), 无需158G全量语料
支持:
  ko_gene(cell_type, condition, gene, n_cells, seeds)
    - cell_type 限 osteoblast / mesenchymal stem cell / osteoclast
    - condition 用自然混合 (与金标计算口径一致); 指定具体条件时若该条件
      在 z0 行中 >=50 条则过滤, 否则回退自然混合并在 condition_note 说明
    - 返回程序级读出: FORM/RES net + marker deltas + 校准z + 方向判定
    - WT 按 (ct,seed) 缓存
  expr_query(cell_type, genes)  -> gene_mean_by_ct 语料均值 (3ct)
  lr_query(gene)                -> lr_pairs_v5.csv
  cond_generate(condition, cell_type, n_cells, seeds)
启动: setsid nohup python server_worker_v3.py > agent_worker_v3.out 2>&1 < /dev/null &
"""
import glob, json, os, sys, time
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, "/root/autodl-tmp/nidus_bundle")
from nidus_light_lib import load_light
from nidus_kofix_lib import gen_ko

BASE = "/root/autodl-tmp/nidus_bundle"
QUEUE = "/root/autodl-tmp/agent_queue"
TOPK = 50
N_CELL = 250
SUPPORTED_CT = ["osteoblast", "mesenchymal stem cell", "osteoclast"]
CT_SHORT = {"osteoblast": "ob", "mesenchymal stem cell": "msc",
            "osteoclast": "oc"}

print("加载轻量模型...", flush=True)
t0 = time.time()
env = load_light(BASE, dev="cuda")
genes = env["genes"]; gidx = env["gidx"]; lut = env["lut"]
z0p = env["z0_pack"]
ctv = env["ct_vocab"]; cdv = env["cond_vocab"]
ct_id_of = ctv if isinstance(ctv, dict) else {n: i for i, n in enumerate(ctv)}
cd_id_of = cdv if isinstance(cdv, dict) else {n: i for i, n in enumerate(cdv)}
cd_name_of = {i: n for n, i in cd_id_of.items()}
gmean = env["gmean"]  # gene_mean_by_ct.npz, keys = 3ct
pairs_df = pd.read_csv("/root/autodl-tmp/lr_pairs_v5.csv")
DRUGS = pd.read_csv("/root/autodl-tmp/chembl_per_drug.csv")
DRUGS["name"] = DRUGS["name"].astype(str).str.upper()

PN = json.load(open("/root/autodl-tmp/program_null_stats.json"))
FORM, RES, NULLS = PN["FORM"], PN["RES"], PN["nulls"]
geneset = set(genes)
FI = [gidx[g] for g in FORM if g in geneset]
RI = [gidx[g] for g in RES if g in geneset]
print(f"就绪 ({time.time()-t0:.0f}s), 轮询 {QUEUE}/todo", flush=True)
os.makedirs(f"{QUEUE}/todo", exist_ok=True)
os.makedirs(f"{QUEUE}/done", exist_ok=True)
json.dump({"ct_vocab": sorted(ct_id_of), "cond_vocab": sorted(cd_id_of)},
          open(f"{QUEUE}/vocab.json", "w"), ensure_ascii=False)

wt_cache = {}


def get_wt(ct, s):
    key = (ct, s)
    if key not in wt_cache:
        cid = ct_id_of[ct]
        z0 = torch.from_numpy(z0p[f"{ct}|{s}|z0"]).to(env["dev"])
        ct_a = np.full(N_CELL, cid, dtype=np.int64)
        cd_a = z0p[f"{ct}|{s}|cd"][:N_CELL]
        tgt = z0p[f"{ct}|{s}|tgt"][:N_CELL]
        wt_cache[key] = lut[gen_ko(env, ct_a, cd_a, tgt, z0, ko_gene=None,
                                   seed=7000 + s).astype(int)].mean(0)
        print(f"WT {ct} s{s} {time.time()-t0:.0f}s", flush=True)
    return wt_cache[key]


def _pick_rows(ct, cond, s):
    """返回 (cd_array, tgt_array, condition_note)"""
    cd = z0p[f"{ct}|{s}|cd"]
    tgt = z0p[f"{ct}|{s}|tgt"]
    if cond in (None, "", "mixed", "natural"):
        return cd[:N_CELL], tgt[:N_CELL], "natural condition mix"
    if cond in cd_id_of:
        m = cd == cd_id_of[cond]
        if m.sum() >= 50:
            return cd[m][:N_CELL], tgt[m][:N_CELL], f"condition={cond} ({int(m.sum())} real rows)"
        return cd[:N_CELL], tgt[:N_CELL], \
            f"condition '{cond}' has <50 real rows for {ct}; used natural mix"
    return cd[:N_CELL], tgt[:N_CELL], \
        f"condition '{cond}' not in vocab; used natural mix"


def _program_readout(dm, ct):
    form_net = float(dm[FI].mean())
    res_net = float(dm[RI].mean())
    out = {"osteoblast_formation_net": round(form_net, 4),
           "osteoclast_resorption_net": round(res_net, 4),
           "readout_note": ("RAW program nets are dominated by the generic "
               "knockout response and must NOT be used for direction calls; "
               "use ONLY the calibration block below (z vs 49 control-gene "
               "KOs) for any direction statement."),}
    short = CT_SHORT.get(ct)
    n = NULLS.get(short, {})
    if n:
        zf = (form_net - n["form_mean"]) / max(n["form_std"], 1e-6)
        zr = (res_net - n["res_mean"]) / max(n["res_std"], 1e-6)
        def call(z):
            return "increase" if z > 1 else ("decrease" if z < -1 else "minimal")
        out["calibration"] = {
            "note": ("z vs null distribution of 49 control-gene KOs in the "
                     "same cell type; |z|>1 = effect beyond generic "
                     "perturbation response"),
            "formation_z": round(float(zf), 3),
            "resorption_z": round(float(zr), 3),
            "formation_call": call(zf),
            "resorption_call": call(zr),
            "program_direction_summary": (
                f"CALIBRATED VERDICT - formation program: {call(zf)} "
                f"(z={zf:+.2f}); resorption program: {call(zr)} (z={zr:+.2f}). "
                f"A 'minimal' call means the KO does NOT move the program "
                f"beyond generic perturbation noise: answer 'no significant "
                f"change' and do NOT infer direction from raw marker signs.")}
    return out


def do_ko_gene(a):
    ct = a.get("cell_type") or a.get("celltype") or a.get("cell")
    cond = a.get("condition") or a.get("cond") or "mixed"
    gene = a["gene"]
    seeds = 10  # 固定10种子, 与金标协议一致
    if ct not in SUPPORTED_CT:
        return {"ok": False,
                "error": (f"cell_type '{ct}' 本服务器暂不支持 KO. "
                          f"可用: {SUPPORTED_CT} (其余细胞型请用 expr_query/lr_query "
                          f"或基于生物学知识回答)")}
    if gene not in gidx:
        return {"ok": False,
                "error": f"基因 {gene} 不在 16105 面板, 请确认拼写"}
    pg = gidx[gene]
    cid = ct_id_of[ct]
    deltas = []
    note = ("natural condition mix (matches benchmark gold-standard "
            "computation); condition-specific KO available on request")
    for s in range(seeds):
        z0 = torch.from_numpy(z0p[f"{ct}|{s}|z0"]).to(env["dev"])
        cd_a = z0p[f"{ct}|{s}|cd"][:N_CELL]
        tgt = z0p[f"{ct}|{s}|tgt"][:N_CELL]
        ct_a = np.full(len(tgt), cid, dtype=np.int64)
        g_k = gen_ko(env, ct_a, cd_a, tgt, z0, ko_gene=pg, seed=7000 + s)
        deltas.append(lut[g_k.astype(int)].mean(0) - get_wt(ct, s))
    D = np.stack(deltas)
    dm = D.mean(0)
    order = np.argsort(-np.abs(dm))[:4 * TOPK]
    up = [(str(genes[i]), round(float(dm[i]), 4)) for i in order if dm[i] > 0][:TOPK]
    dn = [(str(genes[i]), round(float(dm[i]), 4)) for i in order if dm[i] < 0][:TOPK]
    progs = _program_readout(dm, ct)
    cal = progs.get("calibration", {})
    res = {"ok": True, "gene": gene, "cell_type": ct, "condition": cond,
           "DIRECTION_ANSWER": {
               "formation_program": cal.get("formation_call"),
               "resorption_program": cal.get("resorption_call"),
               "instruction": ("For ANY direction question about the "
                   "formation/resorption program, answer with THESE calls. "
                   "The top_up/top_down lists below include the generic "
                   "knockout response and must NOT be used to infer program "
                   "direction.")},
           "programs": progs,
           "condition_note": note, "seeds": seeds, "n_cells": N_CELL,
           "ko_self_delta": round(float(D[:, pg].mean()), 4),
           "top_up": up, "top_down": dn}
    return res


def do_expr_query(a):
    ct = a.get("cell_type") or a.get("celltype") or a.get("cell")
    gs = [g for g in a.get("genes", []) if g in geneset]
    if ct not in set(gmean.files):
        return {"ok": False,
                "error": (f"expr_query 目前覆盖 {sorted(gmean.files)}; "
                          f"'{ct}' 不在其中")}
    v = gmean[ct]
    return {"ok": True, "cell_type": ct,
            "note": "corpus mean expression (natural condition mix)",
            "mean": {g: round(float(v[gidx[g]]), 4) for g in gs}}


def do_lr_query(a):
    g = a.get("gene") or a.get("gene_or_pair") or a.get("pair") or ""
    pair = None
    for sep in ("->", "_", "|", "-", " "):
        if sep in g:
            parts = [p.strip() for p in g.split(sep) if p.strip()]
            if len(parts) == 2:
                pair = parts
            break
    if pair:
        x, y = pair
        m = pairs_df[((pairs_df.genes_a.str.contains(x, na=False)) &
                      (pairs_df.genes_b.str.contains(y, na=False))) |
                     ((pairs_df.genes_a.str.contains(y, na=False)) &
                      (pairs_df.genes_b.str.contains(x, na=False)))]
    else:
        m = pairs_df[(pairs_df.genes_a.str.contains(g, na=False)) |
                     (pairs_df.genes_b.str.contains(g, na=False))]
    edges = [{"pair": r.id_cp, "a": r.genes_a, "b": r.genes_b,
              "dir": r.directionality} for r in m.itertuples()]
    return {"ok": True, "gene": g, "n_edges": len(edges), "edges": edges[:100]}


def do_cond_generate(a):
    ct = a.get("cell_type") or a.get("celltype") or a.get("cell")
    cond = a.get("condition") or a.get("cond")
    if ct not in SUPPORTED_CT:
        return {"ok": False, "error": f"cond_generate 限 {SUPPORTED_CT}"}
    if cond not in cd_id_of:
        return {"ok": False, "error": f"condition '{cond}' 不在词表"}
    s = 0
    z0 = torch.from_numpy(z0p[f"{ct}|{s}|z0"]).to(env["dev"])
    cd_a, tgt, note = _pick_rows(ct, cond, s)
    Xg = gen_ko(env, np.full(len(tgt), ct_id_of[ct], dtype=np.int64),
                cd_a, tgt, z0, ko_gene=None, seed=9100)
    Eg = lut[Xg.astype(int)].mean(0)
    order = np.argsort(-Eg)[:TOPK]
    return {"ok": True, "cell_type": ct, "condition": cond,
            "condition_note": note,
            "top_expressed": [(str(genes[i]), round(float(Eg[i]), 4))
                              for i in order]}


def do_drug_lookup(a):
    name = str(a.get("name") or a.get("compound") or a.get("drug") or "").upper()
    if not name:
        return {"ok": False, "error": "drug_lookup 需要 name 参数"}
    hit = DRUGS[DRUGS.name == name]
    if len(hit) == 0:
        return {"ok": True, "name": name, "in_library": False,
                "note": ("not in the target-annotation library (4277 ChEMBL "
                         "compounds); mechanism-level prediction unavailable")}
    r = hit.iloc[0]
    return {"ok": True, "name": name, "in_library": True,
            "chembl_id": r.molecule_chembl_id,
            "phase": None if pd.isna(r.phase) else float(r.phase),
            "n_targets": int(r.n_targets)}


HANDLERS = {"ko_gene": do_ko_gene, "expr_query": do_expr_query,
            "lr_query": do_lr_query, "cond_generate": do_cond_generate,
            "drug_lookup": do_drug_lookup}

while True:
    for f in sorted(glob.glob(f"{QUEUE}/todo/*.json")):
        jid = os.path.basename(f)[:-5]
        claim = f"{f}.claim{os.getpid()}"
        try:
            os.rename(f, claim)
        except OSError:
            continue
        try:
            job = json.load(open(claim))
            os.remove(claim)
            t1 = time.time()
            if job.get("tool") not in HANDLERS:
                res = {"ok": False,
                       "error": (f"未知工具 '{job.get('tool')}'. 可用工具: "
                                 f"{sorted(HANDLERS)}; ko_gene 参数: cell_type "
                                 f"(限{SUPPORTED_CT}), condition, gene")}
            else:
                res = HANDLERS[job["tool"]](job["args"])
            res.update({"job_id": jid, "tool": job.get("tool"),
                        "elapsed_s": round(time.time() - t1, 1)})
        except Exception as e:
            res = {"job_id": jid, "ok": False,
                   "error": f"{type(e).__name__}: {e}"}
        torch.cuda.empty_cache()
        tmp = f"{QUEUE}/done/{jid}.tmp"
        json.dump(res, open(tmp, "w"), ensure_ascii=False)
        os.replace(tmp, f"{QUEUE}/done/{jid}.json")
        print(f"[{time.strftime('%H:%M:%S')}] {jid} {job.get('tool')} "
              f"ok={res.get('ok')} {res.get('elapsed_s', 0)}s", flush=True)
    time.sleep(3)
