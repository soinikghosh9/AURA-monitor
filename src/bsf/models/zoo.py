"""One interface over every deep EEG model we can run, so a benchmark can be genuinely unified.

The comparisons in this project have repeatedly been decided by protocol rather than architecture, so
the only way to make an architecture claim stick is to run everything through one harness: identical
clips, identical folds, identical optimiser budget, identical pooling of sub-window predictions. This
module is that harness's model side.

Three families, all behind ``build(name, n_chans, n_times, sfreq)``:

  * **braindecode** (external, pip-installed): the standard EEG baselines (EEGNet, ShallowFBCSPNet,
    Deep4Net, EEGConformer, ATCNet, ...) AND the foundation models this manuscript previously could
    only cite -- BIOT, EEGPT, CBraMod, Labram. Running them is strictly better than citing them.
  * **repo models**: the fair CNN baseline and AURA-Net, so the new work is scored on the same folds
    as everything else rather than in its own table.
  * **feature baselines** are handled by the benchmark script directly (they take the 18-D vector, not
    the waveform) and are reported alongside with their input clearly labelled.

Every entry records its parameter count and its input contract, because "which model won" is not
interpretable without "how big was it and what did it see".
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn

# braindecode models verified to construct and forward on (B, n_chans, n_times) at our rates.
# Grouped by family so the benchmark can report like against like.
BRAINDECODE_MODELS: dict[str, list[str]] = {
    "compact_conv": ["EEGNet", "EEGTCNet", "SCCNet", "FBCNet", "SincShallowNet"],
    "classic_conv": ["ShallowFBCSPNet", "Deep4Net", "EEGInceptionERP", "EEGNeX", "EEGSimpleConv"],
    "attention": ["EEGConformer", "ATCNet", "CTNet", "MSVTNet", "TIDNet"],
    "large_or_pretrained": ["SPARCNet", "ContraWR", "BIOT", "EEGPT", "CBraMod"],
}
ALL_BRAINDECODE = [m for v in BRAINDECODE_MODELS.values() for m in v]


@dataclass
class ModelSpec:
    """What a benchmark row needs to be interpretable: the model, its size, and its input contract."""
    name: str
    family: str
    module: nn.Module
    n_params: int
    input_kind: str = "raw"          # "raw" waveform vs "features"
    notes: str = ""
    extra: dict = field(default_factory=dict)


def family_of(name: str) -> str:
    for fam, members in BRAINDECODE_MODELS.items():
        if name in members:
            return fam
    return {"deep_cnn_raw": "repo_baseline", "auranet_hybrid": "repo_ours",
            "auranet_pure": "repo_ours"}.get(name, "other")


def build_braindecode(name: str, n_chans: int, n_times: int, sfreq: float,
                      n_outputs: int = 2) -> nn.Module:
    """Construct a braindecode model, absorbing the per-model API differences.

    Two need special handling and both are worth keeping rather than dropping: ``TCN`` predates the
    ``n_times`` keyword, and ``Labram`` requires canonical 10-20 channel NAMES rather than a count, so
    it can only run on cohorts whose montage we can name (scalp), not on depth electrodes.
    """
    import braindecode.models as M
    cls = getattr(M, name)
    kw = dict(n_chans=n_chans, n_outputs=n_outputs, n_times=n_times, sfreq=sfreq)
    if name == "TCN":
        kw.pop("n_times", None)
        kw.update(n_blocks=3, n_filters=32, kernel_size=7, drop_prob=0.1)
    return cls(**kw)


# Pretrained EEG foundation models, loaded with their published weights rather than constructed from
# random initialisation. The benchmark's "foundation model" rows were architectures trained from
# scratch on a few thousand clips, which tests the architecture but not the pretraining -- the part a
# foundation model is for. These entries load the actual checkpoints (cached offline under
# models/hf_cache) and expect input resampled to the 200 Hz both were pretrained at.
# ``*_scratch200`` are the matched controls: identical architecture, input and fine-tuning recipe,
# random initialisation, so the pretrained-minus-scratch contrast isolates what pretraining buys.
PT_SFREQ = 200.0
PRETRAINED = {
    "BIOT_pt": ("BIOT", "braindecode/biot-pretrained-six-datasets-18chs"),
    "CBraMod_pt": ("CBraMod", "braindecode/cbramod-pretrained"),
    "BIOT_scratch200": ("BIOT", None),
    "CBraMod_scratch200": ("CBraMod", None),
}


def build_pretrained(name: str, n_chans: int, n_times: int, n_outputs: int = 2) -> nn.Module:
    """Build a foundation model at 200 Hz, loading its published encoder weights when it has them.

    CBraMod is channel-count agnostic (convolutional positional encoding), so its checkpoint loads for
    any montage; only the task head is new. BIOT learns one identity embedding per channel and was
    pretrained with 18, so for another montage the 18 learned rows are tiled over the new channel
    count (channel k reuses row k mod 18). No montage here matches BIOT's pretraining montage, so
    which row a channel receives is arbitrary either way; tiling keeps every row a pretrained one.
    The classification head is always freshly initialised.
    """
    import os
    os.environ.setdefault("HF_HOME", str(_hf_home()))
    import braindecode.models as M
    arch, repo = PRETRAINED[name]
    cls = getattr(M, arch)
    kw = dict(n_chans=n_chans, n_outputs=n_outputs, n_times=n_times, sfreq=PT_SFREQ)
    if repo is None:
        return cls(**kw)
    if arch == "CBraMod":
        m = cls.from_pretrained(repo, **kw)
        head = m.final_layer
        for mod in head.modules():                       # fresh head: the checkpoint's is not ours
            if hasattr(mod, "reset_parameters"):
                mod.reset_parameters()
        return m
    src = cls.from_pretrained(repo).state_dict()
    m = cls(**kw)
    dst = m.state_dict()
    n_loaded = 0
    for k, v in src.items():
        if k.startswith("final_layer") or k not in dst:
            continue
        if k == "encoder.channel_tokens.weight" and v.shape != dst[k].shape:
            idx = torch.arange(dst[k].shape[0]) % v.shape[0]
            dst[k] = v[idx].clone()
        elif v.shape == dst[k].shape:
            dst[k] = v.clone()
        else:
            continue
        n_loaded += 1
    m.load_state_dict(dst)
    m._n_pretrained_tensors = n_loaded
    return m


def _hf_home():
    from pathlib import Path
    return Path(__file__).resolve().parents[3] / "models" / "hf_cache"


def build(name: str, n_chans: int, n_times: int, sfreq: float, n_outputs: int = 2,
          feat_dim: int = 0, device: str = "cpu", seed: int | None = None) -> ModelSpec:
    """Build any model in the zoo by name and report its size and input contract.

    ``seed`` seeds the weight initialisation. Without it a model's initial weights depend on however
    much of the global RNG earlier models consumed, so the same model refitted alone, or in a different
    order, starts from different weights -- a seed-to-seed spread that no config records.
    """
    if seed is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    if name in PRETRAINED:
        m = build_pretrained(name, n_chans, n_times, n_outputs).to(device)
        return ModelSpec(name, "pretrained_fm" if PRETRAINED[name][1] else "fm_scratch", m,
                         sum(p.numel() for p in m.parameters()), "raw@200Hz",
                         "published weights" if PRETRAINED[name][1] else "random init control")
    if name in ALL_BRAINDECODE or name == "TCN":
        m = build_braindecode(name, n_chans, n_times, sfreq, n_outputs).to(device)
        return ModelSpec(name, family_of(name), m, sum(p.numel() for p in m.parameters()),
                         "raw", "braindecode")
    if name == "deep_cnn_raw":
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))
        from run_deep_baseline_fair import _build_model
        m = _build_model(n_chans, device)
        return ModelSpec(name, "repo_baseline", m, sum(p.numel() for p in m.parameters()),
                         "raw", "the fair CNN baseline this project has been unable to beat")
    raise ValueError(f"unknown model {name!r}")


@torch.no_grad()
def probe(name: str, n_chans: int, n_times: int, sfreq: float, device: str = "cpu") -> dict:
    """Can this model be constructed and run at this shape? Used to skip cleanly, never silently."""
    try:
        spec = build(name, n_chans, n_times, sfreq, device=device)
        y = spec.module(torch.randn(2, n_chans, n_times, device=device))
        return {"ok": True, "out_shape": tuple(y.shape), "n_params": spec.n_params,
                "family": spec.family}
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "error": f"{type(e).__name__}: {str(e)[:100]}"}


def available(n_chans: int, n_times: int, sfreq: float, device: str = "cpu",
              names: list[str] | None = None) -> tuple[list[str], dict]:
    """Split a candidate list into runnable and skipped, with the reason recorded for each skip."""
    names = names or (ALL_BRAINDECODE + ["deep_cnn_raw"])
    ok, skipped = [], {}
    for n in names:
        r = probe(n, n_chans, n_times, sfreq, device)
        (ok.append(n) if r["ok"] else skipped.update({n: r["error"]}))
    return ok, skipped


def fit_predict(spec: ModelSpec, Wtr: np.ndarray, ytr: np.ndarray, Wte: np.ndarray, device: str,
                epochs: int = 25, lr: float = 1e-3, bs: int = 64, seed: int = 0,
                schedule: bool = True, micro_bs: int = 32, backbone_lr: float | None = None,
                return_embed: bool = False):
    """Train and predict under the SAME recipe for every model in the zoo.

    Identical optimiser, schedule, class weighting, gradient clipping and step budget: an architecture
    comparison in which the models were trained differently measures the recipe, not the architecture.
    ``micro_bs`` bounds memory via gradient accumulation, so a 25M-parameter model and a 4k-parameter
    one still receive the same effective batch.

    ``backbone_lr`` is the single, stated deviation for fine-tuning pretrained weights: every
    parameter outside the task head (``final_layer``) trains at this lower rate while the head keeps
    ``lr``. At 1e-3 throughout, the first epochs of fine-tuning overwrite the pretrained encoder, which
    would test the architecture again rather than the pretraining. The matched random-init controls
    use the same split rates, so the pretrained-minus-scratch contrast is recipe-matched.
    """
    import math

    torch.manual_seed(seed)
    model = spec.module
    if backbone_lr is None:
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    else:
        head = [p for n, p in model.named_parameters() if n.startswith("final_layer")]
        body = [p for n, p in model.named_parameters() if not n.startswith("final_layer")]
        opt = torch.optim.AdamW([{"params": body, "lr": backbone_lr},
                                 {"params": head, "lr": lr}], weight_decay=1e-4)
    cnt = np.bincount(ytr, minlength=2).astype(float)
    w = torch.tensor(cnt.sum() / (2 * np.maximum(cnt, 1)), dtype=torch.float32, device=device)
    lossf = nn.CrossEntropyLoss(weight=w)
    # a view, not a copy: the training array is multi-GB and a copy per process exhausted the host
    # commit that WDDM backs GPU allocations with when several lanes loaded at once (CUDA OOM)
    Xt = torch.from_numpy(np.ascontiguousarray(Wtr, dtype=np.float32))
    yt = torch.tensor(ytr, dtype=torch.long)
    n = len(yt)
    steps = max(1, epochs * int(np.ceil(n / bs)))
    warm = max(1, int(0.1 * steps))
    sched = (torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm
        else 0.5 * (1.0 + math.cos(math.pi * (s - warm) / max(1, steps - warm))))
        if schedule else None)

    model.train()
    rng = np.random.default_rng(seed)
    for _ in range(epochs):
        order = rng.permutation(n)
        for i in range(0, n, bs):
            b = order[i:i + bs]
            opt.zero_grad()
            for j in range(0, len(b), micro_bs):
                mb = b[j:j + micro_bs]
                xb = Xt[mb].to(device); yb = yt[mb].to(device)
                loss = lossf(model(xb), yb) * (len(mb) / len(b))
                loss.backward()
                del xb, yb
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            if sched is not None:
                sched.step()

    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(Wte), 128):
            xb = torch.tensor(Wte[i:i + 128], dtype=torch.float32, device=device)
            out.append(torch.softmax(model(xb), 1)[:, 1].cpu().numpy())
    return np.concatenate(out)


@torch.no_grad()
def frozen_embed(spec: ModelSpec, W: np.ndarray, device: str, bs: int = 64) -> np.ndarray:
    """The frozen encoder's representation: the input to the task head, for a linear probe.

    Reading a pretrained encoder through a fitted linear head is the like-for-like comparison with the
    18-D analytic vector read through logistic regression -- representation against representation,
    with the head held fixed -- and it is immune to the fine-tuning recipe entirely.
    """
    model = spec.module.eval()
    head = getattr(model, "final_layer", None)
    if head is None:
        raise ValueError(f"{spec.name} has no final_layer to probe")
    grabbed = []
    # token-level encoders (CBraMod hands its head a (B, chans, patches, d) grid) are mean-pooled
    # over tokens to one d-vector: flattening instead gives a 35k-dim probe on ~2k clips, which
    # measures the logistic regression's regularisation rather than the representation
    def _grab(m, inp, out):
        z = inp[0].detach().float()
        if z.ndim > 2:
            z = z.reshape(z.shape[0], -1, z.shape[-1]).mean(1)
        grabbed.append(z.cpu().numpy())
    h = head.register_forward_hook(_grab)
    try:
        for i in range(0, len(W), bs):
            model(torch.tensor(W[i:i + bs], dtype=torch.float32, device=device))
    finally:
        h.remove()
    return np.concatenate(grabbed)
