"""Patient-baseline normalization (PBN), trained the way it is used.

The post-hoc version normalizes a frozen network's embedding by the same patient's recent reference
windows and fits a new head. PBN puts that normalization inside training: each step draws a few
training patients, embeds some of their labelled windows together with some of their own reference
(early non-ictal) windows, robust-z-scores the labelled embeddings by the reference embeddings
(median and 1.4826*MAD, clipped at +-20, the statistic used everywhere else) and classifies the
result. Gradients flow through both, so the backbone learns a representation whose departure from the
patient's own baseline carries the label. An optional patient adversary (gradient reversal) on the
normalized embedding penalizes residual patient identity.

At inference the reference is whatever the deployment provides: a patient's baseline clips, or the
stream's windows 35 to 5 min earlier (``rolling_reference``).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


def _head(module):
    h = getattr(module, "final_layer", None)
    if h is not None:
        return h
    return [m for m in module.modules() if isinstance(m, nn.Linear)][-1]


def _heads(module) -> list:
    """The task-head module(s): one final layer, or each window classifier of a ModuleList head
    (ATCNet classifies five sliding windows and averages them)."""
    h = _head(module)
    return list(h) if isinstance(h, nn.ModuleList) else [h]


def hook_heads(module, sink: list) -> list:
    """Register hooks that append each head's input to ``sink``; returns the handles."""
    return [m.register_forward_hook(lambda mm, i, o: sink.append(i[0])) for m in _heads(module)]


def join(parts: list) -> torch.Tensor:
    """Concatenate the flattened inputs of every head call of one forward pass."""
    return torch.cat([_flat(t) for t in parts], 1) if len(parts) > 1 else _flat(parts[0])


def _flat(z: torch.Tensor) -> torch.Tensor:
    if z.ndim > 2:
        z = z.reshape(z.shape[0], -1) if z[0].numel() <= 2048 else z.reshape(z.shape[0], z.shape[1], -1).mean(-1)
    return z


def robust_z(z: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    med = ref.median(0).values
    mad = (ref - med).abs().median(0).values * 1.4826 + 1e-6
    return ((z - med) / mad).clamp(-20, 20)


class _GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lam):
        ctx.lam = lam
        return x.view_as(x)

    @staticmethod
    def backward(ctx, g):
        return -ctx.lam * g, None


class PBNNet(nn.Module):
    def __init__(self, backbone: nn.Module, d_emb: int, n_patients: int = 0):
        super().__init__()
        self.backbone = backbone
        self._zs: list = []
        hook_heads(backbone, self._zs)
        self.cls = nn.Sequential(nn.Dropout(0.25), nn.Linear(d_emb, 2))
        self.adv = (nn.Sequential(nn.Linear(d_emb, 64), nn.ReLU(), nn.Linear(64, n_patients))
                    if n_patients > 1 else None)

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        self._zs.clear()
        self.backbone(x)
        return join(self._zs)

    def classify(self, zn: torch.Tensor) -> torch.Tensor:
        return self.cls(zn / 4.0)                 # z-scores to a unit-ish scale for the linear head


def trainable_patients(g: np.ndarray, ref: np.ndarray) -> list:
    """Patients with both reference and labelled windows (the adversary's classes)."""
    return [s for s in np.unique(g) if (ref & (g == s)).any() and (~ref & (g == s)).any()]


def embed_dim(backbone: nn.Module, n_ch: int, n_t: int, device: str) -> int:
    sink: list = []
    hs = hook_heads(backbone, sink)
    backbone.eval()
    with torch.no_grad():
        backbone(torch.zeros(2, n_ch, n_t, device=device))
    for h in hs:
        h.remove()
    return int(join(sink).shape[1])


def train_pbn(backbone: nn.Module, W: np.ndarray, y: np.ndarray, g: np.ndarray, ref: np.ndarray,
              device: str, epochs: int = 10, lr: float = 3e-4, n_pat: int = 4, n_task: int = 12,
              n_ref: int = 12, adv_weight: float = 0.0, seed: int = 0) -> PBNNet:
    """W (n_clips, k, C, T) clips with k sub-windows, y labels, g patient ids, ref boolean mask of the
    reference clips (used only as references). Fine-tunes ``backbone`` in place inside a PBNNet."""
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    n, k, C, T = W.shape
    pats = trainable_patients(g, ref)
    pid = {s: i for i, s in enumerate(pats)}
    net = PBNNet(backbone, embed_dim(backbone, C, T, device), len(pats) if adv_weight > 0 else 0).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    task_idx = {s: np.flatnonzero(~ref & (g == s)) for s in pats}
    ref_idx = {s: np.flatnonzero(ref & (g == s)) for s in pats}
    pos_w = float((y[~ref] == 0).sum() / max((y[~ref] == 1).sum(), 1))
    ce = nn.CrossEntropyLoss(weight=torch.tensor([1.0, pos_w], device=device, dtype=torch.float32))
    steps = max(1, int(sum(len(v) for v in task_idx.values()) * k / (n_pat * n_task)))
    total = epochs * steps
    for step in range(total):
        net.train()
        chosen = rng.choice(len(pats), size=min(n_pat, len(pats)), replace=False)
        xs, ys, sizes, ps = [], [], [], []
        for j in chosen:
            s = pats[j]
            ti = rng.choice(task_idx[s], n_task); tw = rng.integers(0, k, n_task)
            ri = rng.choice(ref_idx[s], n_ref); rw = rng.integers(0, k, n_ref)
            xs += [W[ti, tw], W[ri, rw]]; ys.append(y[ti]); sizes.append((n_task, n_ref)); ps.append(pid[s])
        x = torch.tensor(np.concatenate(xs), dtype=torch.float32, device=device)
        z = net.embed(x)
        zn, o = [], 0
        for a, b in sizes:
            zn.append(robust_z(z[o:o + a], z[o + a:o + a + b])); o += a + b
        zn = torch.cat(zn)
        yt = torch.tensor(np.concatenate(ys), dtype=torch.long, device=device)
        loss = ce(net.classify(zn), yt)
        if net.adv is not None:
            lam = adv_weight * min(1.0, 2 * step / total)
            pt = torch.tensor(np.repeat(ps, n_task), dtype=torch.long, device=device)
            loss = loss + nn.functional.cross_entropy(net.adv(_GradReverse.apply(zn, lam)), pt)
        opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), 5.0)
        opt.step()
    return net.eval()


@torch.no_grad()
def embed_np(net: PBNNet, X: np.ndarray, device: str, bs: int = 128) -> np.ndarray:
    net.eval()
    return np.concatenate([net.embed(torch.tensor(np.asarray(X[i:i + bs], np.float32), device=device)).float().cpu().numpy()
                           for i in range(0, len(X), bs)])


@torch.no_grad()
def classify_np(net: PBNNet, Zn: np.ndarray, device: str, bs: int = 1024) -> np.ndarray:
    net.eval()
    out = []
    for i in range(0, len(Zn), bs):
        logits = net.classify(torch.tensor(np.asarray(Zn[i:i + bs], np.float32), device=device))
        out.append(torch.softmax(logits, 1)[:, 1].cpu().numpy())
    return np.concatenate(out)


def rolling_reference(E: np.ndarray, t: np.ndarray, start: int, span_s: float, gap_s: float,
                      min_ref: int) -> np.ndarray:
    """Robust z of each window from ``start`` on by the windows (t - gap - span, t - gap]; zeros when
    fewer than ``min_ref`` reference windows exist (the refreshed baseline of the paper)."""
    out = np.zeros((len(E) - start, E.shape[1]), np.float32)
    lo = np.searchsorted(t, t - gap_s - span_s, side="left")
    hi = np.searchsorted(t, t - gap_s, side="right")
    cache_key, med, mad = None, None, None
    for k, i in enumerate(range(start, len(E))):
        if hi[i] - lo[i] < min_ref:
            continue
        if (lo[i], hi[i]) != cache_key:
            r = E[lo[i]:hi[i]]
            med = np.median(r, 0); mad = np.median(np.abs(r - med), 0) * 1.4826 + 1e-6
            cache_key = (lo[i], hi[i])
        out[k] = np.clip((E[i] - med) / mad, -20, 20)
    return out
