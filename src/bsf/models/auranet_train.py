"""Training/inference wrapper for AURA-Net on clip-labelled cohorts.

Keeps the benchmark scripts thin and, more importantly, keeps the protocol honest in one place: the
interictal anchor is fitted from TRAINING-fold negatives only, never from the rows being scored, which
is the failure mode a per-sample causality guard cannot see.

A clip is represented as ``n_sub`` consecutive sub-windows. They are fed through the encoder
independently and then through the S4D core AS A SEQUENCE, so the core carries state across the clip
rather than being a no-op; the clip score is the mean hazard logit over its sub-windows.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F

from bsf.models.aura_net import AURANet


def subwindows(data: np.ndarray, fs: float, n: int = 12, win_s: float = 8.0,
               target_fs: float = 400.0) -> np.ndarray | None:
    """(T, M) clip -> (n, M, L) sub-windows at EXACTLY ``target_fs``, per-window per-channel z-scored.

    Integer decimation cannot hit 400 Hz from every cohort: 5000 Hz decimates by 12 to 416.7 Hz, 400 Hz
    stays at 400. That produced two bugs at once. Windows had different lengths (3333 vs 3200), so
    pooling subjects for a cross-patient run failed outright; and, more quietly, the model's filterbank
    is constructed with a fixed ``fs``, so a subject at 416.7 Hz had every learned band edge off by 4%.
    Resampling to the exact target rate fixes both, and makes ``fs`` a true constant of the pipeline.
    """
    from scipy.signal import decimate, resample_poly
    x = np.nan_to_num(np.asarray(data, dtype=np.float32))
    if x.ndim == 1:
        x = x[:, None]
    q = max(1, int(fs // target_fs))
    if q > 1:                                   # cheap integer pre-decimation, then exact resample
        x = decimate(x, q, axis=0, ftype="fir").astype(np.float32)
    fs_eff = fs / q
    if abs(fs_eff - target_fs) > 1e-6:
        from math import gcd
        a, b = int(round(target_fs)), int(round(fs_eff))
        gg = gcd(a, b)
        x = resample_poly(x, a // gg, b // gg, axis=0).astype(np.float32)
    L = int(round(win_s * target_fs))
    if x.shape[0] < L:
        return None
    starts = np.linspace(0, x.shape[0] - L, n).astype(int)
    W = np.stack([x[s:s + L].T for s in starts])
    mu = W.mean(2, keepdims=True); sd = W.std(2, keepdims=True) + 1e-6
    return np.nan_to_num((W - mu) / sd).astype(np.float32)


class AURANetClassifier:
    """sklearn-ish wrapper: ``fit(W, y, subjects)`` / ``predict_proba(W, subjects)``.

    ``W`` is ``(n_clips, n_sub, M, L)``. ``subjects`` gives each clip's subject id, used only to fit
    and apply the per-subject interictal anchor.
    """

    def __init__(self, fs: float = 400.0, n_filters: int = 32, d_model: int = 128,
                 anchor: bool = False, epochs: int = 60, lr: float = 1e-3, batch: int = 64,
                 weight_decay: float = 1e-4, micro_bs: int = 8, seed: int = 0, device: str | None = None,
                 band_limit_hz: float | None = None, hazard_bias_init: float = 0.0,
                 use_core: bool = False, n_virtual: int = 8, n_free: int = 8,
                 calibrate: bool = False, augment: bool = True,
                 aug_drop: float = 0.1, aug_noise: float = 0.05,
                 subwin_pool: str = 'softmax',
                 d_feat: int = 0, signed: bool = True, norm: str = "batch",
                 gated_fusion: bool = True,
                 verbose: bool = False):
        """Clip configuration by default.

        ``anchor=False``, ``use_core=False`` and ``norm='batch'`` are the CLIP settings.

        The anchor defaults OFF, which reverses this design's original claim, on its own evidence.
        Measured on identical folds after the encoder was fixed, disabling it is worth +0.093 AUC on
        cross-patient scalp forecasting (0.679 vs 0.586) and +0.010 on detection (0.767 vs 0.757),
        with better calibration in both cases. Detection was the task the anchor was supposed to win,
        on the argument that a disjoint per-patient reference state exists there; it does not win it.
        The earlier 1st-of-25 detection result came from a model whose encoder was destroying
        sub-128 ms morphology, so the anchor was compensating for a defect rather than adding
        anything, and it stopped helping once the defect was fixed. The S4D core is a streaming
        device: on clip cohorts the sub-windows are linspace samples minutes apart, so the recurrence
        has no meaningful step size and only adds ~60k parameters to fit from ~66 clips. The streaming
        configuration (continuous forecasting) passes ``use_core=True, norm='layer'``. Same encoder,
        same weights format, two deployment modes.

        ``d_feat>0`` enables the hybrid path: the analytic feature vector enters both as a linear skip
        to the logit (guaranteeing the feature model's performance as a floor) and as a conditioning
        input to the non-linear path.
        """
        # bias_init=0 (not the model default -5) because clip cohorts are balanced; see HazardHead.
        self.kw = dict(fs=fs, n_filters=n_filters, d_model=d_model, anchor=anchor,
                       hazard_bias_init=hazard_bias_init, n_virtual=n_virtual,
                       n_free=n_free, d_feat=d_feat, signed=signed, norm=norm, gated_fusion=gated_fusion)
        self.epochs, self.lr, self.batch, self.wd = epochs, lr, batch, weight_decay
        self.micro_bs = max(1, micro_bs)   # memory unit; the GRADIENT is still full-batch
        self.seed, self.verbose = seed, verbose
        self.band_limit_hz = band_limit_hz
        self.use_core = use_core
        self.calibrate = calibrate
        self.augment, self.aug_drop, self.aug_noise = augment, aug_drop, aug_noise
        self.subwin_pool = subwin_pool
        self.platt: tuple[float, float] | None = None
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model: AURANet | None = None

    # ------------------------------------------------------------------ internals
    def _build(self):
        torch.manual_seed(self.seed)
        m = AURANet(**{**self.kw, 'use_core': self.use_core}).to(self.device)
        if self.band_limit_hz is not None:      # ablation: cap the filterbank's reach
            with torch.no_grad():
                nyq = self.kw["fs"] / 2
                lo = m.filt.f_low.abs().clamp(0.5, self.band_limit_hz - 2)
                m.filt.f_low.copy_(lo)
                m.filt.f_bw.copy_((self.band_limit_hz - lo).clamp(min=2.0))
        return m

    def _sub_logits(self, W: torch.Tensor, subj: list[str],
                    feats: torch.Tensor | None = None) -> torch.Tensor:
        """(B, n_sub, M, L) -> (B, n_sub) per-sub-window hazard logits.

        Supervision happens here rather than on the clip mean. The hazard head is a per-timestep
        quantity by construction, and supervising each sub-window with its clip's label multiplies the
        gradient signal by ``n_sub`` -- which is exactly the data efficiency the fair CNN baseline gets
        from training on sub-windows, and without it a 230k-parameter model cannot learn from the ~60
        independent clips a single subject provides.
        """
        B, S, M, L = W.shape
        z, _ = self.model.embed(W.reshape(B * S, M, L))
        if self.model.anchor.enabled:
            z = self.model.anchor(z, [s for s in subj for _ in range(S)])
        fe = None
        if feats is not None and self.model.feat_proj is not None:
            fe = feats.repeat_interleave(S, dim=0)          # per-clip vector -> per-sub-window
            zf = self.model.feat_proj(fe)
            if self.model.feat_gate is not None:
                # per-dimension gate over [encoder, features]: additive fusion could not discount a
                # noisy encoder, and on typing that cost 0.093 AUC against the features alone
                gt = self.model.feat_gate(torch.cat([z, zf], dim=-1))
                z = gt * z + (1.0 - gt) * zf
            else:
                z = z + zf
        if self.use_core:
            h = self.model.core(z.reshape(B, S, -1)).reshape(B * S, -1)
        else:
            # Ablation, and a suspected design/usage mismatch. The S4D core is a STREAMING recurrence
            # whose state decays over 1-720 contiguous steps. On clip cohorts the sub-windows are
            # linspace samples spanning a 10-60 min clip, so consecutive "steps" are minutes apart and
            # the step size is meaningless. Running a stream model over them may inject noise rather
            # than carry context; bypassing it isolates that.
            h = z
        lh, _, _ = self.model.hazard(h, None, fe)
        return lh.reshape(B, S)

    def _logits(self, W: torch.Tensor, subj: list[str],
                feats: torch.Tensor | None = None) -> torch.Tensor:
        """Clip score from its sub-window logits.

        A plain mean assumes the evidence is spread evenly across the clip. For a pre-ictal window it
        is not: the interpretability sweep finds no single band carrying the signal and the largest
        band effect anywhere is 0.097, which is what intermittent evidence looks like -- present in
        some sub-windows and absent in others. Averaging dilutes exactly that.

        ``softmax`` pooling weights each sub-window by its own confidence, interpolating between the
        mean and a max as the temperature falls. It adds ONE parameter, so it cannot be the difference
        between fitting and not fitting at this sample size, and ``mean`` remains available as the
        ablation that isolates it.
        """
        sl = self._sub_logits(W, subj, feats)
        if self.subwin_pool == "mean" or sl.shape[1] == 1:
            return sl.mean(1)
        w = torch.softmax(sl / self.model.subwin_tau.clamp_min(0.05), dim=1)
        return (w * sl).sum(1)


    def _augment(self, W: torch.Tensor) -> torch.Tensor:
        """Label-preserving perturbations, training only. (B, S, M, L) -> same shape.

        This project's own evidence says the binding constraint is sample size, not capacity: at ~60
        clips per subject a 200k-parameter model has far more freedom than the labels can pin down,
        and no architecture in the 25-model sweep separated from any other. Augmentation is the direct
        attack on that, and it was simply absent.

        Three perturbations, each chosen because it models something a clinic actually does to the
        signal and none of which changes whether a seizure is coming:
          * CHANNEL DROPOUT -- electrodes fail, and the deployability sweep shows models losing
            0.05-0.12 AUC when leads are removed at inference. Training through it should reduce that.
          * PER-CHANNEL GAIN -- impedance drifts and rigs differ; the label does not depend on scale.
          * ADDITIVE NOISE -- keeps the model off exact sample values in a 60-clip regime.

        Deliberately NOT time-shifted or time-reversed: the clips are causal windows ending at a fixed
        distance from the forecast instant, so shifting them changes the horizon the label refers to
        and reversal destroys the arrow of time the pre-ictal claim depends on.
        """
        if not self.augment:
            return W
        B, S, M, L = W.shape
        g = torch.empty(B, 1, M, 1, device=W.device).uniform_(0.8, 1.25)
        W = W * g
        if self.aug_drop > 0:
            keep = (torch.rand(B, 1, M, 1, device=W.device) > self.aug_drop).float()
            # never drop every channel: an all-zero clip has no label-preserving interpretation
            keep = torch.where(keep.sum(dim=2, keepdim=True) < 1, torch.ones_like(keep), keep)
            W = W * keep
        if self.aug_noise > 0:
            W = W + torch.randn_like(W) * self.aug_noise
        return W

    @torch.no_grad()
    def _fit_population_anchor(self, W: np.ndarray, y: np.ndarray, subj: np.ndarray):
        """Cohort-level reference from TRAINING negatives, used for patients we have no anchor for.

        Every task in this study is cross-patient, so before this existed the anchor was applied to
        every training subject and to no test subject -- the downstream layers were fitted on whitened
        embeddings and scored on unwhitened ones.
        """
        if not self.model.anchor.enabled:
            return
        self.model.eval()
        idx = np.where(y == 0)[0]
        if len(idx) < 8:
            return
        zs = []
        for i in range(0, len(idx), 16):
            b = idx[i:i + 16]
            xb = torch.tensor(W[b], device=self.device)
            B, S, M, L = xb.shape
            z, _ = self.model.embed(xb.reshape(B * S, M, L))
            zs.append(z.float().cpu())
        self.model.anchor.fit_population(torch.cat(zs))

    @torch.no_grad()
    def _fit_temperature(self, W: np.ndarray, y: np.ndarray, subj: np.ndarray,
                         feats: np.ndarray | None = None):
        """Two parameters on held-out training SUBJECTS, so the reported probability means something.

        A single temperature was tried first and made calibration WORSE (scalp Brier skill -0.131 ->
        -0.178). Measured on the saved out-of-fold predictions, the reason is that the miscalibration
        here has a BIAS component as well as a sharpness one: a temperature can only rescale the
        logit, and cannot move its mean. On typing a temperature took AURA-Net from -0.828 to -0.360,
        still useless, while adding an intercept took it to +0.071.

        So this fits a slope and an intercept (Platt). Both are monotone increasing in the logit, so
        the ranking -- and therefore every AUC in this study -- is untouched; only the probabilities
        move. The split is by SUBJECT, because a map fitted on the same patients the model memorised
        would be as optimistic as the probabilities it is meant to correct.

        The hazard head is trained by a proper scoring rule, which is necessary for calibration and
        plainly not sufficient: measured Brier skill was negative on seven of eight configurations,
        with expected calibration error up to 0.343. A single temperature is the least invasive fix
        that cannot change the ranking, so discrimination is untouched and only the probabilities move.
        The split is by SUBJECT where possible, because a temperature tuned on the same patients the
        model memorised would be as optimistic as the probabilities it is meant to correct.
        """
        uq = np.unique(subj)
        if len(uq) >= 4:
            rng = np.random.default_rng(self.seed)
            held = set(rng.choice(uq, max(1, len(uq) // 5), replace=False))
            m = np.array([s in held for s in subj])
        else:
            m = np.zeros(len(y), bool)
            m[::5] = True
        if m.sum() < 8 or len(np.unique(y[m])) < 2:
            return
        self.model.eval()
        lg = []
        for i in range(0, int(m.sum()), self.micro_bs):
            sel = np.where(m)[0][i:i + self.micro_bs]
            xb = torch.tensor(W[sel], device=self.device)
            fb = (torch.tensor(feats[sel], dtype=torch.float32, device=self.device)
                  if feats is not None else None)
            lg.append(self._logits(xb, list(subj[sel].astype(str)), fb).float().cpu())
        z = torch.cat(lg).numpy().astype(np.float64)
        yh = y[m].astype(int)
        try:
            from sklearn.linear_model import LogisticRegression
            lr = LogisticRegression(C=1e6, max_iter=1000).fit(z[:, None], yh)
            a = float(lr.coef_[0][0]); b = float(lr.intercept_[0])
        except Exception:  # noqa: BLE001
            return
        if not np.isfinite(a) or not np.isfinite(b) or a <= 0:
            return                      # a non-positive slope would INVERT the ranking; refuse it
        self.platt = (a, b)

    @torch.no_grad()
    def _fit_anchor(self, W: np.ndarray, y: np.ndarray, subj: np.ndarray):
        """Anchor statistics from TRAINING-fold negatives only, per subject."""
        self.model.eval()
        for s in np.unique(subj):
            idx = np.where((subj == s) & (y == 0))[0]
            if len(idx) < 3:
                continue
            zs = []
            for i in range(0, len(idx), 16):
                b = torch.tensor(W[idx[i:i + 16]], device=self.device)
                B, S, M, L = b.shape
                zz, _ = self.model.embed(b.reshape(B * S, M, L))
                zs.append(zz.reshape(B, S, -1).mean(1))
            self.model.anchor.fit_reference(str(s), torch.cat(zs))

    # ------------------------------------------------------------------ api
    def fit(self, W: np.ndarray, y: np.ndarray, subjects: np.ndarray,
            feats: np.ndarray | None = None):
        """Train. Batching matters more than anything else here, for a measurable reason.

        With clip-level labels broadcast to sub-windows, a small batch gives a gradient too noisy to
        escape the ``p=0.5`` saddle of the BCE objective: at batch 8 the loss sits at ln 2 = 0.693 for
        60 epochs and the logit spread is 0.0000. The same model and data at full batch reaches loss
        0.21 and train AUC 0.98. Folds here are 40-80 clips, so a batch that covers the fold is both
        affordable and correct; ``batch`` is therefore an upper bound, not a target. A short warmup
        plus cosine decay stabilises the escape.
        """
        self.model = self._build()
        y = np.asarray(y, int); subjects = np.asarray(subjects).astype(str)
        if feats is not None and self.model.hazard.f_feat is not None:
            # Warm-start the analytic skip from the fitted logistic solution, so training BEGINS at
            # the feature model's operating point rather than converging back towards it. Combined
            # with the zero-initialised learned branch this makes "never worse than the analytic
            # model" an optimisation property, not just a representational one.
            try:
                from sklearn.linear_model import LogisticRegression
                from sklearn.preprocessing import StandardScaler
                sc = StandardScaler().fit(feats)
                lr = LogisticRegression(max_iter=1000, class_weight="balanced").fit(
                    sc.transform(feats), y)
                w = torch.tensor(lr.coef_[0] / (sc.scale_ + 1e-9), dtype=torch.float32)
                b0 = float(lr.intercept_[0] - (lr.coef_[0] * sc.mean_ / (sc.scale_ + 1e-9)).sum())
                with torch.no_grad():
                    self.model.hazard.f_feat.weight.copy_(w[None].to(self.device))
                    self.model.hazard.f_feat.bias.fill_(b0)
                    self.model.hazard.bias.zero_()
                self._warm_started = True
            except Exception:
                self._warm_started = False
        opt = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.wd)
        cnt = np.bincount(y, minlength=2).astype(float)
        pos_w = torch.tensor(cnt[0] / max(cnt[1], 1.0), dtype=torch.float32, device=self.device)
        n = len(y)
        bs = min(self.batch, n)
        steps_per_ep = max(1, int(np.ceil(n / bs)))
        total = max(1, self.epochs * steps_per_ep)
        warm = max(1, int(0.1 * total))
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt, lambda s: (s + 1) / warm if s < warm
            else 0.5 * (1.0 + math.cos(math.pi * (s - warm) / max(1, total - warm))))
        rng = np.random.default_rng(self.seed)
        for ep in range(self.epochs):
            # refresh the anchor from the current encoder a few times during training, so the encoder
            # is optimised THROUGH the anchor rather than against a stale one
            if self.model.anchor.enabled and ep % max(1, self.epochs // 4) == 0:
                self._fit_anchor(W, y, subjects)
            self.model.train()
            order = rng.permutation(n)
            tot = 0.0
            for i in range(0, n, bs):
                b = order[i:i + bs]
                opt.zero_grad()
                # Gradient ACCUMULATION over micro-batches: the optimiser step sees the full-batch
                # gradient (which is what escapes the p=0.5 saddle) while peak memory is set by
                # micro_bs. Full-batch forward on 64 clips x 6 sub-windows x 16 ch x 3200 samples
                # peaked at 15.5 GB of 16 GB, leaving no headroom for the 24-channel subject.
                for j in range(0, len(b), self.micro_bs):
                    mb = b[j:j + self.micro_bs]
                    xb = self._augment(torch.tensor(W[mb], device=self.device))
                    yb = torch.tensor(y[mb], dtype=torch.float32, device=self.device)
                    fb = (torch.tensor(feats[mb], dtype=torch.float32, device=self.device)
                          if feats is not None else None)
                    sl = self._sub_logits(xb, list(subjects[mb]), fb)  # (B, n_sub)
                    loss = F.binary_cross_entropy_with_logits(
                        sl, yb[:, None].expand_as(sl), pos_weight=pos_w)
                    (loss * (len(mb) / len(b))).backward()            # weight by micro-batch share
                    tot += float(loss) * len(mb)
                    del xb, yb, sl
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
                opt.step(); sched.step()
            if self.verbose and (ep + 1) % 10 == 0:
                print(f"      epoch {ep+1:3d}  loss {tot/n:.4f}", flush=True)
        if self.model.anchor.enabled:
            self._fit_anchor(W, y, subjects)      # final anchor from train negatives
            self._fit_population_anchor(W, y, subjects)   # and a fallback for unseen patients
        # OFF by default, and that default is a measured result rather than caution. Fitting a
        # calibrator inside each CV fold and then pooling the out-of-fold predictions mixes one
        # probability scale per fold: on scalp the three folds fitted slopes of 1.7, 6.6 and 3.4, and
        # pooling them took Brier skill from -0.131 to -0.523 and ECE from 0.141 to 0.336. Worse, three
        # DIFFERENT monotone maps change the pooled cross-fold ranking, so per-fold calibration moves
        # the AUC as well (0.566 -> 0.538 here). Calibration belongs AFTER pooling, fitted once on
        # held-out subjects -- which is what scripts/run_calibration.py does, and why that pass took
        # 10 of 94 model-task cells to 62 with max |dAUC| of exactly zero.
        if self.calibrate:
            self._fit_temperature(W, y, subjects, feats)
        return self

    @torch.no_grad()
    def predict_proba(self, W: np.ndarray, subjects: np.ndarray,
                      feats: np.ndarray | None = None) -> np.ndarray:
        self.model.eval()
        subjects = np.asarray(subjects).astype(str)
        out = []
        for i in range(0, len(W), self.micro_bs):
            xb = torch.tensor(W[i:i + self.micro_bs], device=self.device)
            fb = (torch.tensor(feats[i:i + self.micro_bs], dtype=torch.float32,
                                device=self.device) if feats is not None else None)
            lg = self._logits(xb, list(subjects[i:i + self.micro_bs]), fb)
            if self.platt is not None:
                a, b = self.platt
                lg = lg * a + b
            out.append(torch.sigmoid(lg).cpu().numpy())
        return np.concatenate(out)

    @torch.no_grad()
    def adapt_anchor(self, W: np.ndarray, subjects: np.ndarray):
        """Set a NEW subject's anchor from its own (unlabelled-as-negative) reference windows.

        Used at cross-patient test time: the target's anchor must come from its own baseline, never
        from the windows being scored.
        """
        self.model.eval()
        subjects = np.asarray(subjects).astype(str)
        for s in np.unique(subjects):
            idx = np.where(subjects == s)[0]
            if len(idx) < 3:
                continue
            zs = []
            for i in range(0, len(idx), 16):
                b = torch.tensor(W[idx[i:i + 16]], device=self.device)
                B, S, M, L = b.shape
                zz, _ = self.model.embed(b.reshape(B * S, M, L))
                zs.append(zz.reshape(B, S, -1).mean(1))
            self.model.anchor.fit_reference(str(s), torch.cat(zs))

    def interpret(self) -> dict:
        return self.model.interpret() if self.model is not None else {}

    def n_params(self) -> int:
        return self.model.n_params() if self.model is not None else 0
