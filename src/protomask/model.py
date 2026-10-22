"""ProtoMask-Set decoder (REAL, trainable): a small message-passing GNN over the proposal graph
that lets overlapping/competing SAM masks exchange evidence, with text-initialised prototypes and a
per-node calibration (temperature + unknown) head. SAM and CLIP stay frozen; only this module trains
(~few M params). Ablation flags toggle the central mechanisms so the gain is attributable.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ProtoMaskNet(nn.Module):
    def __init__(self, clip_dim, text_protos, n_geom=5, hidden=256, layers=2,
                 use_graph=True, use_uncertainty=True, refine_protos=True, residual_scale=0.2):
        super().__init__()
        self.use_graph = use_graph
        self.use_uncertainty = use_uncertainty
        self.refine_protos = refine_protos
        self.residual_scale = residual_scale                        # anchor: keep correction small
        self.C, self.D = text_protos.shape
        self.register_buffer("text_protos", text_protos)            # (C, D) frozen text init
        self.in_proj = nn.Sequential(nn.Linear(clip_dim + n_geom, hidden), nn.GELU(),
                                     nn.Linear(hidden, hidden))
        # edge-conditioned message MLPs (one per layer)
        self.msg = nn.ModuleList([nn.Sequential(nn.Linear(hidden + 3, hidden), nn.GELU(),
                                                nn.Linear(hidden, hidden)) for _ in range(layers)])
        self.upd = nn.ModuleList([nn.GRUCell(hidden, hidden) for _ in range(layers)])
        self.region_head = nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(),
                                         nn.Linear(hidden, clip_dim))
        nn.init.zeros_(self.region_head[-1].weight); nn.init.zeros_(self.region_head[-1].bias)
        if refine_protos:
            self.proto_proj = nn.Sequential(nn.Linear(clip_dim, clip_dim), nn.GELU(),
                                            nn.Linear(clip_dim, clip_dim))
            nn.init.zeros_(self.proto_proj[-1].weight); nn.init.zeros_(self.proto_proj[-1].bias)
        self.scale = 100.0                                          # FIXED CLIP logit scale (anchor)
        self._res = None                                           # cache last residual for reg
        if use_uncertainty:
            self.temp_head = nn.Linear(hidden, 1)
            self.unknown_head = nn.Linear(hidden, 1)

    def protos(self):
        P = self.text_protos
        if self.refine_protos:
            P = P + 0.1 * self.proto_proj(P)                        # light, residual refine
        return F.normalize(P, dim=-1)

    def forward(self, region_emb, geom, edge_index, edge_feat):
        """region_emb (N,D) L2-normalised, geom (N,5), edge_index (2,E), edge_feat (E,3).
        Returns dict: logits (N,C), unknown (N,), region (N,D)."""
        h = self.in_proj(torch.cat([region_emb, geom], -1))
        if self.use_graph and edge_index.shape[1] > 0:
            src, dst = edge_index[0], edge_index[1]
            for msg, upd in zip(self.msg, self.upd):
                m = msg(torch.cat([h[src], edge_feat], -1))         # (E,hidden)
                w = (edge_feat[:, 0:1] + edge_feat[:, 1:2] + edge_feat[:, 2:3].clamp(min=0))  # edge weight
                agg = torch.zeros_like(h).index_add_(0, dst, m * w)
                deg = torch.zeros(h.shape[0], 1, device=h.device).index_add_(
                    0, dst, w) + 1e-6
                h = upd(agg / deg, h)
        res = self.residual_scale * torch.tanh(self.region_head(h))  # small, bounded correction
        self._res = res
        r = F.normalize(region_emb + res, dim=-1)                    # anchored residual on CLIP region
        P = self.protos()
        logits = self.scale * (r @ P.t())                           # (N,C); falls back to baseline if res~0
        if self.use_uncertainty:
            tau = F.softplus(self.temp_head(h)).squeeze(-1) + 0.5    # per-node temperature (calibration)
            logits = logits / tau.unsqueeze(-1)
            unknown = self.unknown_head(h).squeeze(-1)
        else:
            unknown = torch.full((h.shape[0],), -10.0, device=h.device)
        return {"logits": logits, "unknown": unknown, "region": r}

    def reg(self):
        """Anchor the trained part to frozen CLIP: keep the region residual and proto refinement small,
        so novel (unsupervised) classes fall back to the strong training-free baseline."""
        r = (self._res ** 2).mean() if self._res is not None else 0.0
        p = (self.proto_proj(self.text_protos) ** 2).mean() if self.refine_protos else 0.0
        return r + p
