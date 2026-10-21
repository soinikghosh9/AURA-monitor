"""The input to a network's task head, as an array (the representation the paper adapts)."""
from __future__ import annotations

import numpy as np


def embed(module, X, device, bs: int = 128) -> np.ndarray:
    """Penultimate representation of every window; for multi-window heads (ATCNet) the inputs of all
    head calls are concatenated. Identical to the single-hook version for single-head networks."""
    import torch

    from bsf.models.pbn import hook_heads, join
    module.eval()
    out, sink = [], []
    hs = hook_heads(module, sink)
    try:
        with torch.no_grad():
            for i in range(0, len(X), bs):
                sink.clear()
                module(torch.tensor(np.asarray(X[i:i + bs], np.float32), device=device))
                out.append(join([t.detach().float() for t in sink]).cpu().numpy())
    finally:
        for h in hs:
            h.remove()
    return np.concatenate(out)
