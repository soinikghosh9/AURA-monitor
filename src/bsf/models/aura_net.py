"""AURA-Net: a learned, anchored, montage-agnostic encoder with a survival hazard head.

Design rationale and the evidence forcing each choice are in ``docs/AURA_Net_design.md``. In short,
our own re-runs established that (a) a deep model beats the hand-crafted 18-D vector within subject
once it is given all channels at full bandwidth, (b) deep encoders nonetheless collapse cross-patient
when labels are scarce, (c) day-ahead risk is carried by seizure timing rather than instantaneous EEG,
and (d) post-hoc isotonic calibration destroys ranking, so calibration has to be trained rather than
bolted on. This module is the architecture those four findings imply.

Blocks, in order:

  1. ``SincFilterBank``   parametrised band-pass filters, 2 learnable params each. Buys spectral
                          resolution at almost no capacity cost (labels are scarce) and is readable:
                          the learned band edges are a scientific output.
  2. ``ChannelEncoder``   depthwise-separable dilated *causal* TCN, weights SHARED across channels, so
                          parameter count is independent of the montage and every channel is a training
                          example.
  3. ``ChannelPool``      multi-head attention pooling over the channel SET (+ an explicit synchrony
                          token). Permutation-invariant and variable-M, which is what lets one model
                          span 16-channel canine iEEG and 23-channel scalp. Emits per-channel weights,
                          the substrate for localization.
  4. ``AnchorLayer``      the interictal reference expressed as a differentiable layer: whiten the
                          embedding by that patient's own interictal mean/covariance. Because the
                          encoder trains *through* it, it can allocate capacity to directions that
                          survive anchoring -- the difference between post-hoc CORAL and learned
                          alignment.
  5. ``S4DCore``          diagonal state-space stack, O(1) per step, timescales seconds to weeks.
  6. ``HazardHead``       discrete-time survival: log h(t) = f_EEG(state) + g_timing(clock, history).
                          One model gives every horizon, the likelihood is a proper scoring rule (so
                          calibration is trained), extreme class imbalance is handled natively, and the
                          additive split makes "how much of today's risk is EEG vs timing" readable.

Everything is strictly causal: filters and TCN are left-padded, the SSM is a recurrence, and the anchor
statistics are estimated from past interictal windows only.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Clinical bands used to initialise the filterbank (Hz). The filters are free to move; starting them
# on the physiology means the scarce-label regime does not have to rediscover it.
CLINICAL_BANDS = [(1, 4), (4, 8), (8, 13), (13, 30), (30, 70), (70, 150)]


# --------------------------------------------------------------------------- 1. filterbank
class SincFilterBank(nn.Module):
    """Learnable band-pass filterbank; each filter carries only (f_low, bandwidth).

    A free convolution of length L costs L parameters per filter and, at 6-25 seizures per patient,
    overfits. A sinc band-pass costs 2 and cannot express anything but a band-pass, which is exactly
    the inductive bias the hand-crafted vector encodes. Implemented causally (left-padded), unlike the
    usual centred SincNet, because a forecaster may not see the future.
    """

    def __init__(self, n_filters: int = 32, kernel_size: int = 129, fs: float = 400.0,
                 min_bw_hz: float = 2.0):
        super().__init__()
        if kernel_size % 2 == 0:
            kernel_size += 1
        self.n_filters, self.kernel_size, self.fs = n_filters, kernel_size, fs
        self.min_bw = min_bw_hz
        nyq = fs / 2.0

        # init: the clinical bands first, then log-spaced filters tiling the rest up to Nyquist
        lows, bws = [], []
        for lo, hi in CLINICAL_BANDS:
            if hi < nyq:
                lows.append(lo); bws.append(hi - lo)
        n_rest = max(0, n_filters - len(lows))
        if n_rest:
            # Tile densely up to 70 Hz and sparsely above. Our band-limited deep control showed the
            # advantage is fully available <=45 Hz, and the feature ablation showed removing the HFO
            # band costs 0.000 AUC, so spending a scarce-label capacity budget uniformly up to Nyquist
            # is wasteful. A few high filters remain so the learned edges can still MOVE up and report
            # it if a cohort does carry high-frequency signal.
            n_hi = max(1, n_rest // 6)
            n_lo = n_rest - n_hi
            edges = np.geomspace(1.0, min(70.0, nyq * 0.95), n_lo + 1)
            for i in range(n_lo):
                lows.append(float(edges[i])); bws.append(float(edges[i + 1] - edges[i]))
            if nyq > 80:
                hi_edges = np.geomspace(70.0, nyq * 0.95, n_hi + 1)
                for i in range(n_hi):
                    lows.append(float(hi_edges[i])); bws.append(float(hi_edges[i + 1] - hi_edges[i]))
            else:
                for i in range(n_hi):
                    lows.append(float(edges[-2])); bws.append(float(edges[-1] - edges[-2]))
        lows, bws = lows[:n_filters], bws[:n_filters]

        # store in Hz; abs() at use-time keeps them positive without a hard constraint
        self.f_low = nn.Parameter(torch.tensor(lows, dtype=torch.float32))
        self.f_bw = nn.Parameter(torch.tensor(bws, dtype=torch.float32))
        n = torch.arange(-(kernel_size // 2), kernel_size // 2 + 1, dtype=torch.float32)
        self.register_buffer("n_", n)
        self.register_buffer("window", torch.hamming_window(kernel_size, periodic=False))

    def band_edges(self) -> tuple[np.ndarray, np.ndarray]:
        """Learned (low, high) edges in Hz -- report these; they are an interpretable output."""
        lo = self.f_low.abs()
        hi = lo + self.f_bw.abs().clamp(min=self.min_bw)
        return lo.detach().cpu().numpy(), hi.detach().cpu().numpy()

    def filters(self) -> torch.Tensor:
        nyq = self.fs / 2.0
        lo = self.f_low.abs().clamp(0.1, nyq - self.min_bw - 0.1)
        hi = (lo + self.f_bw.abs().clamp(min=self.min_bw)).clamp(max=nyq - 0.05)
        n = self.n_.to(lo.device)
        # band-pass = difference of two low-pass sincs, windowed
        def lp(fc):
            x = 2.0 * math.pi * (fc[:, None] / self.fs) * n[None, :]
            s = torch.where(n[None, :] == 0, torch.ones_like(x), torch.sin(x) / torch.where(
                x == 0, torch.ones_like(x), x))
            return 2.0 * (fc[:, None] / self.fs) * s
        k = (lp(hi) - lp(lo)) * self.window.to(lo.device)[None, :]
        k = k - k.mean(1, keepdim=True)                     # zero DC
        k = k / (k.norm(dim=1, keepdim=True) + 1e-8)
        return k.unsqueeze(1)                               # (K, 1, L)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B*M, 1, T) -> (B*M, K, T), causal (left-padded)."""
        k = self.filters()
        return F.conv1d(F.pad(x, (self.kernel_size - 1, 0)), k)


# --------------------------------------------------------------------------- 1b. transient filters
class TransientFilters(nn.Module):
    """A few UNCONSTRAINED causal FIR filters, to match transient shape rather than band content.

    The sinc bank is an oscillatory basis: every filter is a band-pass, so it describes a signal by how
    much energy sits in each band. An interictal epileptiform discharge is not an oscillation. It is a
    broadband *transient* -- a sharp deflection followed by a slow wave -- and its diagnostic content is
    its SHAPE, which no band-pass basis represents efficiently.

    This is not a stylistic objection; it is what our own feature ablation says. On canine iEEG the only
    non-redundant feature group is physiology (removing it costs 0.112 AUC, versus ~0.000 for every
    spectral and spatial group), and two of its three members -- spike rate and kurtosis -- are pure
    transient measures. So the hand-crafted vector's decisive component is precisely the one a
    filterbank front-end cannot express, and AURA-Net inherited that blind spot.

    A free FIR of length L costs L parameters and can learn a matched filter for a spike template. At
    8 filters x 65 taps that is ~520 parameters, negligible beside the label budget, and it sits
    alongside the sinc bank rather than replacing it so the interpretable band edges survive.
    """

    def __init__(self, n_free: int = 8, kernel_size: int = 65):
        super().__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv1d(1, n_free, kernel_size, bias=False)
        nn.init.normal_(self.conv.weight, std=0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B*M, 1, T) -> (B*M, n_free, T), causal (left-padded)."""
        w = self.conv.weight
        w = w - w.mean(dim=-1, keepdim=True)              # zero DC, as for the sinc filters
        w = w / (w.norm(dim=-1, keepdim=True) + 1e-8)
        return F.conv1d(F.pad(x, (self.kernel_size - 1, 0)), w)


# --------------------------------------------------------------------------- 2. per-channel encoder
class ChannelEncoder(nn.Module):
    """Band envelopes, then a causal dilated TCN over them. Weights shared across channels.

    The first version ran the TCN directly on the 400 Hz filtered signal with dilations 1-8, giving a
    receptive field of 91 samples: **0.23 s of an 8 s window, 2.8% of it**. It was simultaneously
    myopic and expensive, and it contradicted our own evidence -- the PAC control showed the physiology
    gain collapses from +0.125 to +0.007 when temporal context is truncated, so long context is
    precisely what this task needs.

    The fix is also the physiologically right one. Rectifying the band-passed signal and pooling gives
    the band-power *envelope*, which is what the hand-crafted band-power features measure, except that
    pooling preserves its temporal evolution instead of collapsing it to one number. The TCN then runs
    on the envelope, so a modest dilation stack spans the whole window:

        pool=32, 5 blocks, kernel 7  ->  RF = 187 envelope steps = 5984 samples = 15 s @ 400 Hz

    It is also ~32x cheaper, because the expensive stack no longer runs at the sampling rate.
    """

    def __init__(self, n_filters: int = 32, d_model: int = 64, n_blocks: int = 5, kernel: int = 7,
                 dropout: float = 0.1, pool: int = 32, signed: bool = True, norm: str = "batch",
                 fine: int = 4):
        super().__init__()
        self.pool, self.signed, self.norm_kind = pool, signed, norm
        # Three views per band, not one. The envelope alone discards polarity, which silently defeated
        # both the spatial mixer and the transient filters: a spike and an INVERTED spike produced
        # bit-identical encoder input, so a matched filter could not act as one. The signed mean keeps
        # slow polarity and the signed max keeps peak polarity.
        #
        # FINE BRANCH. pool=32 at 250 Hz is a 128 ms step, and spike / sharp-wave morphology lives at
        # 20-70 ms, so the coarse views destroy the very shape the transient filters exist to detect --
        # which is why ablating those filters measured +0.005. The fine branch rectifies at ``fine``
        # samples (16 ms at 250 Hz) and then summarises each coarse step by the MAX and the SPREAD of
        # its fine envelope. Those two numbers say "a sharp transient occurred inside this step" and
        # "how peaked it was", which a mean over 128 ms cannot express, and they arrive on the coarse
        # grid so the dilation stack keeps its long receptive field.
        self.fine = fine if (fine and pool % fine == 0 and fine > 1) else 0
        n_view = (3 if signed else 1) + (2 if self.fine else 0)
        n_in = n_filters * n_view
        self.n_view = n_view
        # The views enter on wildly different scales -- log1p|x| is O(1) while the signed max is
        # O(3 sigma) -- and they were being fed straight into one compressing 1x1 conv. Normalising
        # first stops the largest-scale view dominating the projection and its gradient. BatchNorm is
        # used for the same reason the blocks use it: at inference it applies fixed running statistics
        # and is therefore causal in time.
        self.view_norm = nn.BatchNorm1d(n_in)
        self.inp = nn.Conv1d(n_in, d_model, 1)
        self.blocks = nn.ModuleList()
        for b in range(n_blocks):
            d = 2 ** b
            # BatchNorm normalises over the BATCH; at eval it uses fixed running statistics, so it is
            # causal in time at inference. An earlier note here claimed AURA-Net could not adopt it
            # without breaking streaming -- that was wrong, and it cost the model a free regulariser
            # the CNN baseline was getting. LayerNorm (per timestep, no time mixing) stays available
            # for the streaming configuration. What must NEVER be used is GroupNorm(1, C), which
            # normalises over channels AND TIME jointly and leaks the future into every step.
            self.blocks.append(nn.ModuleDict({
                "dw": nn.Conv1d(d_model, d_model, kernel, dilation=d, groups=d_model),
                "pw": nn.Conv1d(d_model, d_model, 1),
                "nm": nn.BatchNorm1d(d_model) if norm == "batch" else nn.LayerNorm(d_model),
            }))
            setattr(self, f"pad{b}", (kernel - 1) * d)
        self.drop = nn.Dropout(dropout)
        self.n_blocks, self.kernel = n_blocks, kernel

    def receptive_field_s(self, fs: float = 400.0) -> float:
        steps = 1 + sum((self.kernel - 1) * (2 ** b) for b in range(self.n_blocks))
        return steps * self.pool / fs

    def hidden(self, x: torch.Tensor) -> torch.Tensor:
        """(B*M, K, T) -> (B*M, D, T/pool): the dilated stack's output as a TIME SERIES.

        Exposed because causality is a property of this sequence, not of the window summary that
        ``forward`` returns -- a mean over time legitimately changes when the future changes. The
        causality test used to re-implement the view construction to get at it, and silently broke the
        moment the views changed. One implementation, used by both.
        """
        e = self._views(x)
        h = self.inp(self.view_norm(e))
        for b, blk in enumerate(self.blocks):
            pad = getattr(self, f"pad{b}")
            z = blk["dw"](F.pad(h, (pad, 0)))               # causal: pad only on the left
            z = (blk["nm"](z) if self.norm_kind == "batch"
                 else blk["nm"](z.transpose(1, 2)).transpose(1, 2))
            z = blk["pw"](F.gelu(z))
            h = h + self.drop(z)
        return h

    def _views(self, x: torch.Tensor) -> torch.Tensor:
        """(B*M, K, T) -> (B*M, K*n_view, T/pool). Causal: rectify and pool look backward only."""
        # band envelope (magnitude), plus signed views so polarity survives
        e = torch.log1p(F.avg_pool1d(x.abs(), self.pool, stride=self.pool))
        if self.signed:
            e = torch.cat([e,
                           F.avg_pool1d(x, self.pool, stride=self.pool),      # signed slow component
                           F.max_pool1d(x, self.pool, stride=self.pool)],     # signed peak
                          dim=1)
        if self.fine:
            r = self.pool // self.fine
            f = F.avg_pool1d(x.abs(), self.fine, stride=self.fine)            # 16 ms envelope
            pk = F.max_pool1d(f, r, stride=r)                                 # sharpest fine peak
            mn = F.avg_pool1d(f, r, stride=r)
            sq = F.avg_pool1d(f * f, r, stride=r)
            sd = torch.sqrt(torch.clamp(sq - mn * mn, min=0.0) + 1e-8)        # peakedness in-step
            n = e.shape[-1]
            e = torch.cat([e, torch.log1p(pk[..., :n]), torch.log1p(sd[..., :n])], dim=1)
        return e

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B*M, K, T) -> (B*M, 2D): mean over time plus the final causal step."""
        h = self.hidden(x)
        return torch.cat([h.mean(-1), h[..., -1]], dim=-1)  # (B*M, 2D)


# --------------------------------------------------------------------------- 2b. spatial mixing
class SpatialMixer(nn.Module):
    """Learned spatial filters: R virtual channels, each a linear combination of the raw band signals.

    This closes the one structural advantage the plain CNN baseline holds. Its first layer,
    ``Conv1d(n_ch, 64, 15)``, mixes every channel on the RAW waveform, so it can learn spatial filters:
    a bipolar derivation ``ch_i - ch_j``, a Laplacian, a CSP-like projection. Those are not incidental
    to EEG -- they are how focal discharges are read, and a polarity inversion across a focus is the
    single most diagnostic spatial feature of an epileptiform spike.

    AURA-Net had no such operation. It reshaped to ``B*M`` and encoded each channel independently, then
    combined them only at the end as an attention-weighted average of per-channel embeddings -- and by
    then the envelope ``|x|`` had already discarded phase, so cross-channel polarity was unrecoverable
    in principle, not merely unlearned.

    The mixer restores it without giving up montage-agnosticism. Weights ``a[r, m]`` are computed from
    channel ``m``'s own content, never its index, so permuting the montage permutes the summands and
    leaves each virtual channel unchanged; ``R`` is fixed, so downstream shape is montage-independent.
    Weights are deliberately NOT softmaxed: a simplex cannot express a difference, and differences are
    the point.
    """

    def __init__(self, n_filters: int = 32, n_virtual: int = 8, d_key: int = 32):
        super().__init__()
        self.n_virtual = n_virtual
        self.key = nn.Sequential(nn.Linear(n_filters, d_key), nn.GELU(), nn.Linear(d_key, d_key))
        self.query = nn.Parameter(torch.randn(n_virtual, d_key) * 0.1)
        self.scale = d_key ** -0.5

    def forward(self, xf: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """(B, M, K, T) -> virtual channels (B, R, K, T) and mixing weights (B, R, M)."""
        B, M, K, T = xf.shape
        # per-channel content descriptor: log band energy. Depends on the channel's SIGNAL, not on
        # where it sits in the montage, which is what makes the result permutation-invariant.
        c = torch.log1p(xf.abs().mean(-1))                      # (B, M, K)
        # Centre the descriptor across channels. log1p(|x|) is strictly positive, so an uncentred
        # descriptor makes the mixing logits systematically single-signed at initialisation and the
        # mixer starts unable to express a difference. Centring also states the right inductive bias:
        # a spatial filter cares how a channel deviates from the montage, which is the average
        # reference every EEG pipeline applies by hand.
        c = c - c.mean(dim=1, keepdim=True)
        k = self.key(c)                                         # (B, M, d_key)
        a = torch.einsum("rd,bmd->brm", self.query, k) * self.scale
        # Unit-norm each spatial filter, as CSP and Laplacian filters are normalised. Without this the
        # raw logits are ~1e-2, so a virtual channel comes out ~25x weaker than a real one and the
        # encoder simply ignores it. Normalising fixes the SCALE while leaving the DIRECTION (including
        # the sign pattern that encodes a difference) entirely free.
        a = a / (a.norm(dim=-1, keepdim=True) + 1e-6)
        v = torch.einsum("brm,bmkt->brkt", a, xf)               # genuine linear combination, phase kept
        return v, a


# --------------------------------------------------------------------------- 3. channel-set pooling
class ChannelPool(nn.Module):
    """Permutation-invariant pooling over the channel set (Set-Transformer style PMA).

    Montage-agnosticism lives here: the model sees channels as an unordered set, so one network spans
    16-channel canine iEEG and 23-channel scalp. Returns the pooled embedding and the per-channel
    attention weights, which are the spatial map used for localization and for review displays.
    """

    def __init__(self, d_in: int, d_model: int = 128, n_heads: int = 4, n_seeds: int = 4):
        super().__init__()
        self.proj = nn.Linear(d_in, d_model)
        self.seeds = nn.Parameter(torch.randn(n_seeds, d_model) * 0.02)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.out = nn.Linear(d_model * n_seeds + 1, d_model)   # +1 for the synchrony scalar

    def forward(self, e: torch.Tensor, mask: torch.Tensor | None = None):
        """e: (B, M, d_in), mask: (B, M) True=valid -> (z (B, d_model), alpha (B, M))."""
        B, M, _ = e.shape
        h = self.proj(e)
        q = self.seeds.unsqueeze(0).expand(B, -1, -1)
        kpm = None if mask is None else ~mask
        pooled, attw = self.attn(q, h, h, key_padding_mask=kpm, need_weights=True,
                                 average_attn_weights=True)
        pooled = self.norm(pooled).flatten(1)                  # (B, n_seeds*d_model)
        # explicit synchrony: mean off-diagonal cosine similarity of channel embeddings. The
        # hand-crafted synchrony scalar was load-bearing on Kaggle, so we hand it over rather than
        # hoping attention rediscovers it.
        hn = F.normalize(h, dim=-1)
        G = hn @ hn.transpose(1, 2)
        if mask is not None:
            m2 = (mask[:, :, None] & mask[:, None, :]).float()
            G = G * m2
            denom = (m2.sum((1, 2)) - mask.sum(1).float()).clamp(min=1.0)
        else:
            denom = torch.full((B,), float(M * (M - 1)), device=e.device).clamp(min=1.0)
        sync = ((G.sum((1, 2)) - torch.diagonal(G, dim1=1, dim2=2).sum(1)) / denom).unsqueeze(1)
        z = self.out(torch.cat([pooled, sync], dim=-1))
        return z, attw.mean(1)                                  # attw: (B, n_seeds, M) -> (B, M)


# --------------------------------------------------------------------------- 4. anchor
class AnchorLayer(nn.Module):
    """Whiten the embedding by the patient's own interictal statistics (the AURA anchor, learned).

    ``fit_reference`` estimates (mu, Sigma^{-1/2}) from a patient's interictal embeddings with
    Ledoit-Wolf shrinkage; ``forward`` applies the patient's transform. Statistics come from past
    interictal windows only, so the layer is causal. ``enabled=False`` bypasses it -- required for
    spatial localization, where our own results show the patient frame the anchor removes IS the
    signal.
    """

    def __init__(self, d_model: int, shrink: float = 0.1, eps: float = 1e-3, enabled: bool = True,
                 mode: str = "auto", full_cov_ratio: float = 4.0):
        super().__init__()
        self.d, self.shrink, self.eps, self.enabled = d_model, shrink, eps, enabled
        self.mode, self.full_cov_ratio = mode, full_cov_ratio
        self._ref: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        self._used: dict[str, str] = {}
        self._pop: tuple[torch.Tensor, torch.Tensor] | None = None

    @torch.no_grad()
    def fit_population(self, z_all: torch.Tensor) -> None:
        """A cohort-level reference, used for any patient we have no reference of our own for.

        Diagonal by construction: this is a fallback, and a full covariance estimated once over a
        pooled cohort would impose one patient group's geometry on everyone. Its only job is to put a
        new patient on the same footing the training patients were whitened onto, so that the layers
        after it see the distribution they were fitted on.
        """
        z = z_all.detach().float()
        if z.ndim != 2 or z.shape[0] < 8:
            return
        mu = z.mean(0)
        sd = z.std(0).clamp_min(self.eps)
        self._pop = (mu, torch.diag(1.0 / sd))

    @torch.no_grad()
    def fit_reference(self, pid: str, z_interictal: torch.Tensor) -> None:
        """Estimate the patient's reference transform.

        ``mode='full'`` whitens by the full covariance; ``'diag'`` by per-feature scale only;
        ``'auto'`` (default) uses full covariance ONLY when the reference set is large relative to the
        embedding dimension.

        The guard is a sample-complexity argument, not a bug fix, and the distinction matters. A full
        covariance needs O(d^2) parameters from n_ref samples, and a realistic interictal reference
        here is ~20-60 clips against d=128; whitening amplifies the reference's LOW-variance
        directions, which at n_ref < d are estimation noise. Our matched causal head-to-head agrees
        empirically: first-order recolouring (0.652) beat second-order whitening (0.577) for
        forecasting.

        Honest caveat on the evidence. A collapse (logits -> 0, loss pinned at ln 2) was first observed
        WITH the full-covariance anchor, and attributed to it. That attribution was wrong: once the
        encoder's receptive field was fixed, all three anchor modes train to AUC 1.000 on the same
        overfit test, forced full covariance included. The collapse was an interaction with the myopic
        encoder, whose near-constant embeddings made the reference covariance degenerate. The diagonal
        default is retained on the sample-complexity and empirical grounds above, not because full
        covariance is broken.
        """
        z = z_interictal.detach()
        n = z.shape[0]
        if n < 3:
            return
        mu = z.mean(0)
        use_full = (self.mode == "full") or (self.mode == "auto" and n >= self.full_cov_ratio * self.d)
        if use_full:
            zc = z - mu
            S = (zc.T @ zc) / max(1, n - 1)
            S = (1 - self.shrink) * S + self.shrink * (torch.trace(S) / self.d) * torch.eye(
                self.d, device=z.device, dtype=z.dtype)
            vals, vecs = torch.linalg.eigh(S + self.eps * torch.eye(self.d, device=z.device,
                                                                    dtype=z.dtype))
            W = vecs @ torch.diag(vals.clamp(min=self.eps).rsqrt()) @ vecs.T
            self._used[pid] = "full"
        else:
            sd = z.std(0).clamp(min=1e-3)
            W = torch.diag(1.0 / sd)            # per-feature z-score: O(d) params, estimable at n<<d
            self._used[pid] = "diag"
        self._ref[pid] = (mu, W)

    def modes_used(self) -> dict[str, str]:
        return dict(self._used)

    def has(self, pid: str) -> bool:
        return pid in self._ref

    def forward(self, z: torch.Tensor, pids: list[str]) -> torch.Tensor:
        """Apply each patient's reference transform, falling back to the population one.

        The fallback is not cosmetic. Without it an unseen patient was silently left UNTRANSFORMED
        while every training patient was whitened, so the layers downstream were fitted on anchored
        embeddings and then handed unanchored ones at test time. Every benchmark here is
        cross-patient, so that mismatch applied to every test subject in the study, and it predicts
        exactly the below-chance scalp result (0.476) that a plain failure to learn would not.
        """
        if not self.enabled:
            return z
        out = z.clone()
        for pid in set(pids):
            idx = [i for i, p in enumerate(pids) if p == pid]
            if not idx:
                continue
            ref = self._ref.get(pid, self._pop)
            if ref is None:
                continue
            mu, W = ref
            sel = torch.as_tensor(idx, device=z.device)
            # device AND dtype: the per-subject references are fitted on the compute device but the
            # population one is pooled on CPU, and casting dtype alone crashes only for the fallback
            mu = mu.to(device=z.device, dtype=z.dtype)
            W = W.to(device=z.device, dtype=z.dtype)
            out[sel] = (z[sel] - mu) @ W
        return out


# --------------------------------------------------------------------------- 5. multi-timescale core
class S4DCore(nn.Module):
    """Diagonal state-space stack: O(1)/step recurrence spanning seconds to weeks."""

    def __init__(self, d_model: int = 128, n_state: int = 64, n_layers: int = 3, dt_h: float = 1.0):
        super().__init__()
        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            # log-decay init spanning ~1 step to ~1 month, so different states carry the proximal
            # precursor and the multidien drift in one model
            tau = torch.logspace(math.log10(1.0), math.log10(720.0 / max(dt_h, 1e-3)), n_state)
            self.layers.append(nn.ParameterDict({
                "log_a": nn.Parameter(torch.log(torch.expm1(1.0 / tau))),   # softplus^-1
                "B": nn.Parameter(torch.randn(d_model, n_state) * 0.1),
                "C": nn.Parameter(torch.randn(n_state, d_model) * 0.1),
                "D": nn.Parameter(torch.ones(d_model) * 0.1),
            }))
        self.norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_layers)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, T, D) -> (B, T, D), strictly causal (cumulative recurrence)."""
        h = x
        for layer, norm in zip(self.layers, self.norms):
            a = torch.exp(-F.softplus(layer["log_a"]))               # (N,) in (0,1)
            u = h @ layer["B"]                                       # (B, T, N)
            B_, T, N = u.shape
            state = torch.zeros(B_, N, device=x.device, dtype=x.dtype)
            outs = []
            for t in range(T):                                       # explicit recurrence = causal
                state = a * state + u[:, t]
                outs.append(state)
            y = torch.stack(outs, 1) @ layer["C"] + h * layer["D"]
            h = norm(h + F.gelu(y))
        return h


# --------------------------------------------------------------------------- 6. hazard head
class HazardHead(nn.Module):
    """Discrete-time survival hazard with an additive EEG / timing decomposition.

        log h(t) = f_EEG(state_t) + g_timing(clock_t, seizure history_t)

    Trained by the survival log-likelihood, which is a proper scoring rule -- so the probabilities are
    calibrated by training rather than by a post-hoc isotonic step whose ties destroyed our ranking.
    ``survival_prob`` integrates the hazard forward, giving EVERY horizon from one forward pass instead
    of a separate classifier per horizon. The additive split is what makes "today's risk is mostly
    timing" a readable quantity rather than an interpretation.
    """

    def __init__(self, d_model: int = 128, d_timing: int = 7, hidden: int = 64,
                 bias_init: float = -5.0, d_feat: int = 0):
        super().__init__()
        self.f_eeg = nn.Sequential(nn.Linear(d_model, hidden), nn.GELU(), nn.Linear(hidden, 1))
        self.g_time = nn.Sequential(nn.Linear(d_timing, hidden), nn.GELU(), nn.Linear(hidden, 1))
        # LINEAR skip from the analytic features to the logit. This is the load-bearing part of the
        # hybrid: with the learned path at zero the model reduces exactly to logistic regression on the
        # 18-D vector, so it inherits that estimator's performance as a FLOOR rather than having to
        # rediscover Welch PSD and a correlation matrix from ~66 clips. AURA-Net previously lost to
        # that linear model on 9 of 11 subjects, which is an optimisation failure, not a capacity one.
        self.f_feat = nn.Linear(d_feat, 1) if d_feat else None
        if self.f_feat is not None:
            nn.init.zeros_(self.f_feat.bias)
            # Start the LEARNED path silent, so the model begins life AS the analytic model and has to
            # earn any departure from it. Without this the floor is only a capability: at
            # initialisation an untrained encoder injects noise that swamps the linear skip, and the
            # hybrid scored BELOW both the pure model and the feature model on a smoke run. Zero-init
            # of the residual branch is the standard remedy and makes the floor hold in optimisation,
            # not just in principle.
            nn.init.zeros_(self.f_eeg[-1].weight)
            nn.init.zeros_(self.f_eeg[-1].bias)
        # -5 encodes the rare-event prior of a continuous stream (base rate ~0.5%). On a BALANCED clip
        # cohort that prior is wrong and costs most of the training budget climbing out of it, so the
        # clip benchmarks pass bias_init=0.
        self.bias = nn.Parameter(torch.tensor(float(bias_init)))

    def forward(self, state: torch.Tensor, timing: torch.Tensor | None = None,
                feats: torch.Tensor | None = None):
        le = self.f_eeg(state).squeeze(-1)
        lt = self.g_time(timing).squeeze(-1) if timing is not None else torch.zeros_like(le)
        lf = (self.f_feat(feats).squeeze(-1) if (self.f_feat is not None and feats is not None)
              else torch.zeros_like(le))
        return le + lt + lf + self.bias, le, lt          # logit-hazard and its components

    @staticmethod
    def survival_prob(logit_h: torch.Tensor, horizon: int) -> torch.Tensor:
        """P(event within the next ``horizon`` bins) = 1 - prod(1 - h). Causal: uses only t..t+H-1."""
        h = torch.sigmoid(logit_h)
        log1m = torch.log1p(-h.clamp(max=1 - 1e-6))
        c = torch.cumsum(log1m, dim=-1)
        pad = torch.zeros_like(c[..., :1])
        cshift = torch.cat([pad, c[..., :-1]], dim=-1)
        idx = torch.arange(h.shape[-1], device=h.device)
        j = (idx + horizon).clamp(max=h.shape[-1] - 1)
        return 1.0 - torch.exp(c[..., j] - cshift)

    @staticmethod
    def nll(logit_h: torch.Tensor, event: torch.Tensor) -> torch.Tensor:
        """Discrete-time survival NLL. ``event`` is 1 in the bin an onset occurs, else 0."""
        return F.binary_cross_entropy_with_logits(logit_h, event.float(), reduction="mean")


# --------------------------------------------------------------------------- full model
class AURANet(nn.Module):
    """Filterbank -> shared channel encoder -> set pooling -> anchor -> S4D -> hazard / typing."""

    def __init__(self, fs: float = 400.0, n_filters: int = 32, d_channel: int = 64,
                 d_model: int = 128, n_state: int = 64, n_layers: int = 3, d_timing: int = 7,
                 n_types: int = 2, anchor: bool = True, dt_h: float = 1.0,
                 hazard_bias_init: float = -5.0, n_virtual: int = 8, n_free: int = 8,
                 d_feat: int = 0, signed: bool = True, norm: str = "batch",
                 use_core: bool = True, pool_ms: float = 64.0, fine_ms: float = 8.0,
                 gated_fusion: bool = True):
        super().__init__()
        self.fs = fs
        self.n_virtual, self.n_free = n_virtual, n_free
        self.filt = SincFilterBank(n_filters, fs=fs)
        self.trans = TransientFilters(n_free) if n_free else None
        n_bands = n_filters + n_free
        self.mixer = SpatialMixer(n_bands, n_virtual) if n_virtual else None
        self.d_feat = d_feat
        # Pooling is specified in TIME and converted per cohort. As a fixed sample count it silently
        # made the encoder a different model on every dataset -- 128 ms at 250 Hz, 80 ms at 400 Hz,
        # 64 ms at 500 Hz -- so its resolution and receptive field were an accident of the sampling
        # rate rather than a choice about physiology.
        pool = max(2, int(round(fs * pool_ms / 1000.0)))
        fine = max(1, int(round(fs * fine_ms / 1000.0)))
        while pool % fine and fine > 1:
            fine -= 1
        self.chan = ChannelEncoder(n_bands, d_channel, signed=signed, norm=norm,
                                   pool=pool, fine=fine)
        # the analytic vector also joins the NON-linear path, so the encoder can condition on it
        # Fusion is GATED, not additive. Additively, z = z + proj(features) gives the model no way to
        # discount its own encoder when that encoder is contributing noise -- and on cross-patient
        # typing the analytic vector alone scores 0.822 while the vector PLUS the encoder scores
        # 0.729. Adding a good signal to a bad one lost 0.093 AUC, which is a fusion failure rather
        # than a representation one. A per-dimension gate lets the model fall back to the features
        # where the encoder does not earn its place, and keep the encoder where it does.
        self.feat_proj = (nn.Sequential(nn.LayerNorm(d_feat), nn.Linear(d_feat, d_model), nn.GELU())
                          if d_feat else None)
        self.feat_gate = (nn.Sequential(nn.Linear(2 * d_model, d_model), nn.Sigmoid())
                          if (d_feat and gated_fusion) else None)
        self.pool = ChannelPool(d_channel * 2, d_model)
        self.anchor = AnchorLayer(d_model, enabled=anchor)
        # The S4D core is a STREAMING recurrence. On clip cohorts the sub-windows are linspace
        # samples minutes apart, so its step size is meaningless and it was already bypassed at call
        # time -- but it was still constructed, still counted in the reported parameter budget, and
        # still received weight decay. Not building it makes the clip configuration honest about its
        # own size.
        self.core = S4DCore(d_model, n_state, n_layers, dt_h=dt_h) if use_core else None
        self.hazard = HazardHead(d_model, d_timing, bias_init=hazard_bias_init, d_feat=d_feat)
        self.typing = nn.Linear(d_model, n_types)
        # temperature for confidence-weighted sub-window pooling; init 1.0 keeps it
        # close to the mean at the start of training so it cannot destabilise the escape
        # from the p=0.5 saddle that the batching note below is about
        self.subwin_tau = nn.Parameter(torch.ones(1))

    def embed(self, x: torch.Tensor, mask: torch.Tensor | None = None):
        """x: (B, M, T) raw multichannel -> (z (B, d_model), channel attention (B, M[+R])).

        The encoder runs over the ORIGINAL channels and, when the spatial mixer is enabled, over R
        learned virtual channels appended to the set. Keeping the originals preserves the per-channel
        attention map that localization needs; adding the virtual ones supplies the spatial filtering
        the plain CNN gets from its first layer.
        """
        B, M, T = x.shape
        xr = x.reshape(B * M, 1, T)
        xf = self.filt(xr)
        if self.trans is not None:
            xf = torch.cat([xf, self.trans(xr)], dim=1)     # band content + transient shape
        xf = xf.reshape(B, M, -1, T)
        if self.mixer is not None:
            v, _ = self.mixer(xf)                               # (B, R, K, T)
            xf = torch.cat([xf, v], dim=1)
            if mask is not None:
                mask = torch.cat([mask, mask.new_ones(B, v.shape[1])], dim=1)
        n = xf.shape[1]
        e = self.chan(xf.reshape(B * n, xf.shape[2], T)).reshape(B, n, -1)
        return self.pool(e, mask)

    def forward(self, x, pids=None, mask=None, timing=None, seq_len=None):
        """Windows -> logit-hazard. ``x`` is (B, M, T); windows are consecutive in time when
        ``seq_len`` is given, so the S4D core carries state across them."""
        z, attn = self.embed(x, mask)
        if pids is not None:
            z = self.anchor(z, pids)
        if self.core is None:
            s = z
        else:
            h = z.unsqueeze(0) if seq_len is None else z.reshape(-1, seq_len, z.shape[-1])
            s = self.core(h).reshape(-1, h.shape[-1])
        logit_h, le, lt = self.hazard(s, timing)
        return {"logit_hazard": logit_h, "eeg_term": le, "timing_term": lt,
                "embedding": z, "state": s, "channel_attention": attn,
                "typing_logits": self.typing(z)}

    def interpret(self) -> dict:
        """Readable model state: which frequencies the filterbank settled on."""
        lo, hi = self.filt.band_edges()
        order = np.argsort(lo)
        return {"filter_bands_hz": [(round(float(lo[i]), 1), round(float(hi[i]), 1)) for i in order]}

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
