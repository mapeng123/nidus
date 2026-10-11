#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Nidus 轻量推理库: 不依赖158G token store, 只加载 ckpt + panel + z0 包"""
import numpy as np, torch, torch.nn as nn

CHUNK, MASK_TOKEN = 48, 51
D_MODEL, N_HEAD, N_LAYER, GID_DIM = 512, 8, 12, 256

class PMDDM(nn.Module):
    def __init__(self, G, G_PAD, N_CT, N_COND, N_PERT, NULL_PERT):
        super().__init__()
        self.G, self.G_PAD, self.NULL_PERT = G, G_PAD, NULL_PERT
        self.N_CHUNK = G_PAD // CHUNK
        self.gene_emb = nn.Embedding(52, D_MODEL)
        self.compress = nn.Linear(CHUNK * D_MODEL, D_MODEL)
        self.pos_emb = nn.Parameter(torch.randn(1, self.N_CHUNK + 4, D_MODEL) * 0.02)
        self.ct_emb = nn.Embedding(N_CT, D_MODEL)
        self.cond_emb = nn.Embedding(N_COND, D_MODEL)
        self.gene_id_emb = nn.Embedding(G, GID_DIM)
        self.pert_gene_mlp = nn.Sequential(nn.Linear(GID_DIM, 512), nn.GELU(), nn.Linear(512, D_MODEL))
        self.pert_free_emb = nn.Embedding(N_PERT, GID_DIM)
        self.pert_free_mlp = nn.Sequential(nn.Linear(GID_DIM, 512), nn.GELU(), nn.Linear(512, D_MODEL))
        self.delta_head = nn.Sequential(nn.Linear(D_MODEL * 3, 512), nn.GELU(),
                                        nn.Linear(512, 512), nn.GELU(), nn.Linear(512, G))
        self.z0_proj = nn.Linear(D_MODEL, D_MODEL)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(D_MODEL, N_HEAD, D_MODEL * 4, batch_first=True,
                                       norm_first=True, activation="gelu"), N_LAYER)
        self.decompress = nn.Linear(D_MODEL, CHUNK * 52)

    def z0_token(self, x0):
        e = self.gene_emb(x0.long()).view(x0.shape[0], self.N_CHUNK, CHUNK * D_MODEL)
        return self.z0_proj(self.compress(e).mean(1))

    def pert_vector(self, pert, pgid):
        ref = self.gene_emb.weight
        v = torch.zeros(pert.shape[0], D_MODEL, device=pert.device, dtype=ref.dtype)
        is_null = pert.eq(self.NULL_PERT)
        gm = (pgid >= 0) & ~is_null
        if gm.any():
            v[gm] = self.pert_gene_mlp(self.gene_id_emb(pgid[gm])).to(v.dtype)
        fm = (pgid < 0) & ~is_null
        if fm.any():
            v[fm] = self.pert_free_mlp(self.pert_free_emb(pert[fm])).to(v.dtype)
        return v

    def forward(self, x, ct, cond, pert, z0, pgid):
        B = x.shape[0]
        e = self.gene_emb(x.long()).view(B, self.N_CHUNK, CHUNK * D_MODEL)
        z = self.compress(e)
        pv = self.pert_vector(pert, pgid)
        head = torch.stack([self.ct_emb(ct), self.cond_emb(cond), pv, z0], 1)
        z = torch.cat([head, z], 1) + self.pos_emb
        z = self.encoder(z)[:, 4:]
        return self.decompress(z).view(B, self.G_PAD, 52)


def load_light(base="/root/autodl-tmp/nidus_bundle", dev="cuda", with_packs=True):
    genes = [l.strip() for l in open(f"{base}/bone_genes_v4.txt")]
    G = len(genes)
    gidx = {g: i for i, g in enumerate(genes)}
    edges = np.load(f"{base}/bone_bin_edges_v4.npy")
    lut = np.concatenate([[0.0], edges, [edges[-1]]]).astype(np.float32)
    ck = torch.load(f"{base}/pmdm_v3_oponly_step2000.pt", map_location="cpu", weights_only=False)
    ct_vocab, cond_vocab, pert_vocab = ck["ct_vocab"], ck["cond_vocab"], ck["pert_vocab"]
    N_CHUNK = (G + CHUNK - 1) // CHUNK
    G_PAD = N_CHUNK * CHUNK
    N_CT, N_COND, N_PERT = len(ct_vocab) + 1, len(cond_vocab) + 1, len(pert_vocab) + 1
    NULL_CT, NULL_COND, NULL_PERT = len(ct_vocab), len(cond_vocab), len(pert_vocab)
    model = PMDDM(G, G_PAD, N_CT, N_COND, N_PERT, NULL_PERT).to(dev)
    model.load_state_dict(ck["model"])
    model.eval()
    env = dict(model=model, genes=genes, gidx=gidx, G=G, G_PAD=G_PAD, lut=lut,
               ct_vocab=ct_vocab, cond_vocab=cond_vocab, pert_vocab=pert_vocab,
               NULL_CT=NULL_CT, NULL_COND=NULL_COND, NULL_PERT=NULL_PERT, dev=dev)
    if with_packs:
        env["z0_pack"] = np.load(f"{base}/z0_pack.npz")
        env["gmean"] = np.load(f"{base}/gene_mean_by_ct.npz")
    return env
