#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""修正版 KO 生成库: z0 真实对照 + 硬掩码自抑制 (MaskGIT 迭代解码中 KO 位点全程锁空)
用法: from nidus_kofix_lib import gen_ko, make_z0
"""
import numpy as np, torch
import torch.nn.functional as F

MASK_TOKEN = 51

def make_z0(env, rows):
    """从真实细胞行号编码 z0 (与训练时扰动批一致)"""
    from nidus_pmdm_lib import read_rows
    xreal = read_rows(env, rows)
    xp = torch.from_numpy(np.pad(xreal, ((0, 0), (0, env["G_PAD"] - env["G"])))).to(env["dev"])
    with torch.no_grad():
        z0 = env["model"].z0_token(xp).mean(0, keepdim=True)
    return z0  # (1, 512)

def gen_ko(env, ct_ids, cond_ids, tgt_nnz, z0, ko_gene=None,
           T=64, cfg_w=1.5, temp=1.0, seed=0, gumbel=0.3, batch=128):
    """硬 KO MaskGIT 生成 (分批防 OOM).
    ko_gene: None=WT; 否则=面板基因索引, 该位置全程锁掩码(=0), 同时软路 pgid 也喂入.
    返回 (n, G) int8
    """
    n_all = len(ct_ids)
    outs = []
    for b0 in range(0, n_all, batch):
        b1 = min(batch, n_all - b0)
        outs.append(_gen_ko_batch(env, ct_ids[b0:b1], cond_ids[b0:b1], tgt_nnz[b0:b1],
                                  z0, ko_gene=ko_gene, T=T, cfg_w=cfg_w, temp=temp,
                                  seed=seed + b0, gumbel=gumbel))
    return np.concatenate(outs)

def _gen_ko_batch(env, ct_ids, cond_ids, tgt_nnz, z0, ko_gene=None,
                  T=64, cfg_w=1.5, temp=1.0, seed=0, gumbel=0.3):
    model, dev = env["model"], env["dev"]
    G, G_PAD = env["G"], env["G_PAD"]
    NULL_CT, NULL_COND, NULL_PERT = env["NULL_CT"], env["NULL_COND"], env["NULL_PERT"]
    n = len(ct_ids)
    torch.manual_seed(seed); np.random.seed(seed)
    pgids = np.full(n, -1 if ko_gene is None else ko_gene, dtype=np.int64)
    pert_id = NULL_PERT if ko_gene is None else 0
    x = torch.full((n, G_PAD), MASK_TOKEN, dtype=torch.long, device=dev)
    ct = torch.as_tensor(ct_ids, dtype=torch.long, device=dev)
    cd = torch.as_tensor(cond_ids, dtype=torch.long, device=dev)
    pt = torch.full((n,), pert_id, dtype=torch.long, device=dev)
    pg = torch.as_tensor(pgids, dtype=torch.long, device=dev)
    z0b = z0.expand(n, -1)
    ct_null = torch.full_like(ct, NULL_CT); cd_null = torch.full_like(cd, NULL_COND)
    pt_null = torch.full_like(pt, NULL_PERT); pg_null = torch.full_like(pg, -1)
    tgt = torch.as_tensor(tgt_nnz, dtype=torch.long, device=dev)
    cur = torch.zeros(n, dtype=torch.long, device=dev)
    ar = torch.arange(n, device=dev)
    for t in range(T, 0, -1):
        masked = x == MASK_TOKEN
        if not masked.any():
            break
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            lg_c = model(x, ct, cd, pt, z0b, pg)
            lg_u = model(x, ct_null, cd_null, pt_null, z0b, pg_null)
            logits = (1 + cfg_w) * lg_c.float() - cfg_w * lg_u.float()
        probs = F.softmax(logits / temp, -1)
        p0 = probs[..., 0]
        nz_conf = torch.log1p(-p0 + 1e-9)
        if gumbel > 0:
            nz_conf = nz_conf + torch.rand_like(nz_conf).log().neg().log().neg() * gumbel
        nz_conf[~masked] = -1e9
        if ko_gene is not None:
            nz_conf[:, ko_gene] = -1e9  # 硬 KO: 全程锁空
        need = (tgt - cur).clamp(min=0)
        q = torch.div(need + t - 1, t, rounding_mode="floor")
        q = torch.minimum(q, masked.sum(1))
        qmax = int(q.max().item())
        if qmax > 0:
            topi = nz_conf.topk(qmax, dim=1).indices
            sel = torch.arange(qmax, device=dev)[None, :] < q[:, None]
            rows_ = ar[:, None].expand(-1, qmax)[sel]
            pos = topi[sel]
            pnz = probs[rows_, pos, 1:51]
            pnz = pnz / pnz.sum(-1, keepdim=True).clamp(min=1e-9)
            lvl = torch.multinomial(pnz, 1).squeeze(-1) + 1
            x[rows_, pos] = lvl
            cur += sel.sum(1)
    x[x == MASK_TOKEN] = 0
    return x[:, :G].cpu().numpy().astype(np.int8)
