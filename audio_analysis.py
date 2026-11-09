"""
Audio loading + feature extraction (numpy/scipy only, vectorised, cached).

Every stem is analysed once into a matrix of shape (n_features, n_frames) at a
hop of 512 samples (~86 frames/s @ 44.1 kHz).  All features are normalised to
0..1 so they can be routed to any parameter in the modulation matrix.
"""
from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, field
from math import gcd
from typing import Callable

import numpy as np
import soundfile as sf
from scipy import signal as sps

ANALYSIS_VERSION = 3
HOP = 512
N_FFT = 2048
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".cache")

# (key, label) -- order defines the row order in the modulation matrix
FEATURES: list[tuple[str, str]] = [
    ("rms", "Volume (RMS)"),
    ("onset", "Onset / transient"),
    ("beat", "Beat pulse"),
    ("beat_phase", "Beat phase (saw)"),
    ("brightness", "Brightness (centroid)"),
    ("rolloff", "Rolloff (85%)"),
    ("flatness", "Noisiness (flatness)"),
    ("bandwidth", "Spectral bandwidth"),
    ("flux", "Spectral flux (smooth)"),
    ("pitch", "Pitch (F0, log)"),
    ("pitch_conf", "Pitch confidence"),
    ("chroma_hue", "Chroma pitch class"),
    ("tonal", "Tonalness"),
    ("sub", "Band: sub 20-60"),
    ("low", "Band: low 60-250"),
    ("lowmid", "Band: low-mid 250-1k"),
    ("mid", "Band: mid 1k-4k"),
    ("high", "Band: high 4k-10k"),
    ("air", "Band: air 10k+"),
    ("zcr", "Zero-crossing rate"),
    ("pan", "Stereo pan"),
    ("width", "Stereo width"),
]
FEATURE_KEYS = [k for k, _ in FEATURES]
FEATURE_INDEX = {k: i for i, k in enumerate(FEATURE_KEYS)}

BANDS = {"sub": (20, 60), "low": (60, 250), "lowmid": (250, 1000),
         "mid": (1000, 4000), "high": (4000, 10000), "air": (10000, 24000)}

ProgressCB = Callable[[float, str], None]


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------
def load_audio(path: str, target_sr: int | None = None) -> tuple[np.ndarray, int]:
    """Returns float32 stereo (n, 2) and sample rate."""
    data, sr = sf.read(path, dtype="float32", always_2d=True)
    if data.shape[1] == 1:
        data = np.repeat(data, 2, axis=1)
    elif data.shape[1] > 2:
        data = data[:, :2]
    if target_sr and sr != target_sr:
        g = gcd(int(sr), int(target_sr))
        data = sps.resample_poly(data, target_sr // g, sr // g, axis=0).astype(np.float32)
        sr = target_sr
    return np.ascontiguousarray(data, dtype=np.float32), int(sr)


def bandpass(stereo: np.ndarray, sr: int, lo: float, hi: float) -> np.ndarray:
    nyq = sr * 0.5
    lo = max(lo, 10.0)
    hi = min(hi, nyq * 0.98)
    if lo <= 15:
        sos = sps.butter(4, hi / nyq, btype="lowpass", output="sos")
    elif hi >= nyq * 0.97:
        sos = sps.butter(4, lo / nyq, btype="highpass", output="sos")
    else:
        sos = sps.butter(4, [lo / nyq, hi / nyq], btype="bandpass", output="sos")
    return sps.sosfiltfilt(sos, stereo, axis=0).astype(np.float32)


def band_edges(n: int, lo: float = 30.0, hi: float = 16000.0) -> list[tuple[float, float]]:
    e = np.geomspace(lo, hi, n + 1)
    e[0], e[-1] = 0.0, 24000.0
    return [(float(e[i]), float(e[i + 1])) for i in range(n)]


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def _robust_norm(x: np.ndarray, pct: float = 99.0) -> np.ndarray:
    nz = x[x > 1e-9]
    if nz.size == 0:
        return np.zeros_like(x)
    ref = np.percentile(nz, pct)
    if ref <= 1e-12:
        return np.zeros_like(x)
    return np.clip(x / ref, 0.0, 1.0)


def _log_freq_norm(f: np.ndarray, lo=50.0, hi=12000.0) -> np.ndarray:
    return np.clip(np.log2(np.maximum(f, lo) / lo) / np.log2(hi / lo), 0.0, 1.0)


def _smooth(x: np.ndarray, n: int) -> np.ndarray:
    if n <= 1:
        return x
    k = np.hanning(n + 2)[1:-1]
    k /= k.sum()
    return np.convolve(x, k, mode="same")


def _windowed_rms(x: np.ndarray, n_frames: int) -> np.ndarray:
    """RMS of a 1-D signal on the analysis frame grid (centred windows)."""
    pad = N_FFT // 2
    xp = np.pad(x.astype(np.float64), (pad, pad + N_FFT))
    cs = np.concatenate([[0.0], np.cumsum(xp * xp)])
    starts = np.arange(n_frames) * HOP
    e = cs[starts + N_FFT] - cs[starts]
    return np.sqrt(np.maximum(e, 0) / N_FFT)


def track_beats(onset: np.ndarray, fps: float) -> np.ndarray:
    """Ellis-style dynamic-programming beat tracker. Returns beat frame indices."""
    n = len(onset)
    if n < fps * 2 or onset.std() < 1e-6:
        return np.array([], dtype=int)
    o = (onset - onset.mean()) / (onset.std() + 1e-9)
    # --- tempo estimate from onset autocorrelation with a perceptual prior
    ac = np.fft.irfft(np.abs(np.fft.rfft(o, 2 * n)) ** 2)[:n]
    bpms = np.linspace(60, 190, 400)
    lags = 60.0 * fps / bpms
    vals = np.interp(lags, np.arange(n), ac)
    weight = np.exp(-0.5 * (np.log2(bpms / 120.0) / 0.9) ** 2)
    period = float(lags[np.argmax(vals * weight)])
    # --- DP
    lo, hi = max(1, int(round(period / 2))), int(round(period * 2))
    d = np.arange(lo, hi + 1)
    penalty = 100.0 * np.log(d / period) ** 2
    score = o.copy()
    back = np.full(n, -1, dtype=np.int64)
    for t in range(lo, n):
        dd = d[d <= t]
        cand = score[t - dd] - penalty[: len(dd)]
        k = int(np.argmax(cand))
        if cand[k] > 0:
            score[t] = o[t] + cand[k]
            back[t] = t - dd[k]
    tail = max(0, n - int(period))
    t = tail + int(np.argmax(score[tail:]))
    beats = []
    while t >= 0:
        beats.append(t)
        t = back[t]
    return np.array(beats[::-1], dtype=int)


# --------------------------------------------------------------------------
# Core analysis
# --------------------------------------------------------------------------
def analyze(stereo: np.ndarray, sr: int, progress: ProgressCB | None = None) -> np.ndarray:
    mono = stereo.mean(axis=1).astype(np.float32)
    n_frames = 1 + len(mono) // HOP
    out = np.zeros((len(FEATURES), n_frames), dtype=np.float32)
    F = FEATURE_INDEX

    padded = np.pad(mono, (N_FFT // 2, N_FFT // 2 + N_FFT))
    frames_view = np.lib.stride_tricks.sliding_window_view(padded, N_FFT)[::HOP][:n_frames]

    win = np.hanning(N_FFT).astype(np.float32)
    freqs = np.fft.rfftfreq(N_FFT, 1.0 / sr)
    nb = len(freqs)

    # chroma mapping matrix (bins 55 Hz .. 5 kHz)
    chroma_map = np.zeros((nb, 12), dtype=np.float32)
    valid = (freqs >= 55) & (freqs <= 5000)
    pcs = (np.round(12 * np.log2(np.maximum(freqs, 1) / 440.0) + 69).astype(int)) % 12
    chroma_map[np.where(valid)[0], pcs[valid]] = 1.0
    ang = 2 * np.pi * np.arange(12) / 12.0

    band_masks = {k: (freqs >= lo) & (freqs < hi) for k, (lo, hi) in BANDS.items()}

    tau_min = max(2, int(sr / 2000.0))
    tau_max = min(N_FFT // 2 - 2, int(sr / 50.0))

    band_raw = {k: np.zeros(n_frames, np.float32) for k in BANDS}
    flux = np.zeros(n_frames, np.float32)
    pitch_hz = np.zeros(n_frames, np.float32)
    prev_log = None
    chunk = 2048
    for c0 in range(0, n_frames, chunk):
        c1 = min(n_frames, c0 + chunk)
        fr = np.asarray(frames_view[c0:c1], dtype=np.float32)
        rms = np.sqrt(np.mean(fr * fr, axis=1))
        out[F["rms"], c0:c1] = rms
        sg = np.signbit(fr)
        out[F["zcr"], c0:c1] = np.mean(sg[:, 1:] != sg[:, :-1], axis=1)

        mag = np.abs(np.fft.rfft(fr * win, axis=1)).astype(np.float32)
        pw = mag * mag
        msum = mag.sum(axis=1) + 1e-9
        cen = (mag * freqs).sum(axis=1) / msum
        out[F["brightness"], c0:c1] = _log_freq_norm(cen)
        bw = np.sqrt((mag * (freqs[None, :] - cen[:, None]) ** 2).sum(axis=1) / msum)
        out[F["bandwidth"], c0:c1] = bw
        cps = np.cumsum(pw, axis=1)
        ro = np.argmax(cps >= 0.85 * cps[:, -1:], axis=1)
        out[F["rolloff"], c0:c1] = _log_freq_norm(freqs[ro])
        lp = np.log(pw + 1e-10)
        out[F["flatness"], c0:c1] = np.exp(lp.mean(axis=1)) / (pw.mean(axis=1) + 1e-10)
        for k, m in band_masks.items():
            band_raw[k][c0:c1] = np.sqrt(pw[:, m].sum(axis=1))

        lm = np.log1p(100.0 * mag)
        prev = np.vstack([prev_log[None, :] if prev_log is not None else lm[:1], lm[:-1]])
        flux[c0:c1] = np.maximum(0.0, lm - prev).sum(axis=1)
        prev_log = lm[-1]

        ch = pw @ chroma_map
        chs = ch.sum(axis=1) + 1e-9
        out[F["tonal"], c0:c1] = (ch.max(axis=1) / chs * 12.0 - 1.0) / 11.0
        hue = np.arctan2((ch * np.sin(ang)).sum(1), (ch * np.cos(ang)).sum(1))
        out[F["chroma_hue"], c0:c1] = (hue / (2 * np.pi)) % 1.0

        # --- YIN-style pitch via FFT autocorrelation
        X = np.fft.rfft(fr, n=2 * N_FFT, axis=1)
        r = np.fft.irfft(np.abs(X) ** 2, axis=1)[:, : N_FFT // 2]
        dfn = np.maximum(2.0 * r[:, :1] - 2.0 * r, 0.0)
        dfn[:, 0] = 0
        cums = np.cumsum(dfn[:, 1:], axis=1)
        cmnd = np.ones_like(dfn)
        cmnd[:, 1:] = dfn[:, 1:] * np.arange(1, dfn.shape[1]) / (cums + 1e-12)
        seg = cmnd[:, tau_min:tau_max]
        below = seg < 0.15
        has = below.any(axis=1)
        first = np.where(has, np.argmax(below, axis=1), np.argmin(seg, axis=1))
        idx = np.arange(seg.shape[1])[None, :]
        wmask = (idx >= first[:, None]) & (idx <= (first * 1.5 + 2)[:, None])
        tau_i = np.argmin(np.where(wmask, seg, np.inf), axis=1)
        conf = 1.0 - seg[np.arange(len(seg)), tau_i]
        tau = (tau_i + tau_min).astype(np.float64)
        # parabolic refinement
        ti = np.clip(tau_i + tau_min, 1, cmnd.shape[1] - 2)
        a, b, cc = cmnd[np.arange(len(ti)), ti - 1], cmnd[np.arange(len(ti)), ti], cmnd[np.arange(len(ti)), ti + 1]
        den = a - 2 * b + cc
        safe_mask = np.abs(den) > 1e-9
        tau = tau + np.where(safe_mask, 0.5 * (a - cc) / np.where(safe_mask, den, 1.0), 0.0)
        conf = np.where(rms > 1e-3, np.clip(conf, 0, 1), 0.0)
        pitch_hz[c0:c1] = sr / np.maximum(tau, 1.0)
        out[F["pitch_conf"], c0:c1] = conf

        if progress:
            progress(c1 / n_frames * 0.85, "analysing")

    # --- pitch: hold last confident value
    conf = out[F["pitch_conf"]]
    good = conf > 0.5
    if good.any():
        idx = np.where(good, np.arange(n_frames), 0)
        np.maximum.accumulate(idx, out=idx)
        held = pitch_hz[idx]
        held[: np.argmax(good)] = pitch_hz[np.argmax(good)]
        out[F["pitch"]] = np.clip(np.log2(np.clip(held, 50, 2000) / 50.0) / np.log2(40.0), 0, 1)

    # --- normalisation
    for k in ("rms", "zcr", "bandwidth", "flatness"):
        out[F[k]] = _robust_norm(out[F[k]])
    for k in BANDS:
        out[F[k]] = _robust_norm(band_raw[k])
    onset = _robust_norm(flux)
    out[F["onset"]] = onset
    fps = sr / HOP
    out[F["flux"]] = _robust_norm(_smooth(flux, int(fps * 0.25)))
    out[F["tonal"]] = np.clip(out[F["tonal"]], 0, 1)

    # --- stereo
    L, R = stereo[:, 0], stereo[:, 1]
    rl, rr = _windowed_rms(L, n_frames), _windowed_rms(R, n_frames)
    rm, rs = _windowed_rms((L + R) * 0.5, n_frames), _windowed_rms((L - R) * 0.5, n_frames)
    out[F["pan"]] = np.clip(0.5 + 0.5 * (rr - rl) / (rr + rl + 1e-9), 0, 1)
    out[F["width"]] = np.clip(rs / (rm + rs + 1e-9) * 2.0, 0, 1)

    # --- beats
    if progress:
        progress(0.9, "beat tracking")
    beats = track_beats(_smooth(onset, 3), fps)
    if len(beats) >= 2:
        t = np.arange(n_frames)
        nxt_i = np.searchsorted(beats, t, side="right")
        prv = beats[np.clip(nxt_i - 1, 0, len(beats) - 1)]
        nxt = beats[np.clip(nxt_i, 0, len(beats) - 1)]
        period = np.median(np.diff(beats))
        span = np.where(nxt > prv, nxt - prv, period)
        since = np.where(t >= prv, t - prv, t - prv + period)
        out[F["beat_phase"]] = np.clip(since / span, 0, 1)
        pulse = np.exp(-np.maximum(since, 0) / (0.12 * fps))
        pulse[t < beats[0]] = 0
        out[F["beat"]] = pulse
    if progress:
        progress(1.0, "done")
    return np.nan_to_num(out, copy=False)


# --------------------------------------------------------------------------
# Caching wrapper
# --------------------------------------------------------------------------
def _cache_key(*parts) -> str:
    h = hashlib.sha1("|".join(str(p) for p in parts + (ANALYSIS_VERSION, HOP, N_FFT)).encode())
    return h.hexdigest()


def file_signature(path: str) -> str:
    st = os.stat(path)
    return f"{os.path.abspath(path)}|{st.st_size}|{int(st.st_mtime)}"


def cached_analyze(key_parts: tuple, stereo_fn: Callable[[], tuple[np.ndarray, int]],
                   progress: ProgressCB | None = None) -> np.ndarray:
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = os.path.join(CACHE_DIR, _cache_key(*key_parts) + ".npy")
    if os.path.exists(path):
        try:
            return np.load(path)
        except Exception:
            pass
    stereo, sr = stereo_fn()
    feats = analyze(stereo, sr, progress)
    try:
        np.save(path, feats)
    except Exception:
        pass
    return feats


# --------------------------------------------------------------------------
# Feature bank used by the modulation engine
# --------------------------------------------------------------------------
@dataclass
class FeatureBank:
    sr: int
    stems: list[np.ndarray] = field(default_factory=list)   # each (F, N)
    mix: np.ndarray | None = None

    @property
    def fps(self) -> float:
        return self.sr / HOP

    def sample(self, idx: int, t: float) -> np.ndarray:
        """idx >= 0: stem; idx == -1: mix. Linear interpolation in time."""
        arr = self.mix if idx < 0 else (self.stems[idx] if idx < len(self.stems) else None)
        if arr is None or arr.shape[1] == 0:
            return np.zeros(len(FEATURES), np.float32)
        x = t * self.fps
        i = int(np.floor(x))
        if i < 0:
            return arr[:, 0].copy()
        if i >= arr.shape[1] - 1:
            return np.zeros(len(FEATURES), np.float32)  # past the end => silence
        f = x - i
        return arr[:, i] * (1 - f) + arr[:, i + 1] * f

