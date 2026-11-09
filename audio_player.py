"""Real-time stem playback via sounddevice with a smooth clock for visual sync."""
from __future__ import annotations

import threading
import time

import numpy as np

try:
    import sounddevice as sd
except Exception:  # pragma: no cover
    sd = None


class AudioPlayer:
    def __init__(self):
        self.sr = 44100
        self.stems: list[np.ndarray] = []     # stereo float32 arrays (only playable stems)
        self.gains: list[float] = []
        self.master = 0.8
        self.length = 0
        self._pos = 0
        self._cb_pos = 0
        self._cb_time = 0.0
        self.playing = False
        self.finished = False
        self._stream = None
        self._lock = threading.Lock()
        self._free_t0 = None     # clock when no audio is loaded
        self._free_base = 0.0
        self.latency = 0.0

    # ------------------------------------------------------------------
    def set_audio(self, stems: list[np.ndarray], sr: int):
        was = self.playing
        self.pause()
        with self._lock:
            if self._stream is not None and sr != self.sr:
                self._close()
            self.sr = sr
            self.stems = stems
            self.gains = [1.0] * len(stems)
            self.length = max((len(s) for s in stems), default=0)
            self._pos = min(self._pos, self.length)
        if was:
            self.play()

    def set_gains(self, gains: list[float]):
        with self._lock:
            self.gains = list(gains) + [1.0] * (len(self.stems) - len(gains))

    @property
    def duration(self) -> float:
        return self.length / self.sr if self.length else 0.0

    @property
    def has_audio(self) -> bool:
        return self.length > 0 and sd is not None

    # ------------------------------------------------------------------
    def _open(self):
        if self._stream is None and sd is not None:
            self._stream = sd.OutputStream(samplerate=self.sr, channels=2, dtype="float32",
                                           callback=self._callback, latency="low")
            self._stream.start()
            try:
                self.latency = float(self._stream.latency)
            except Exception:
                self.latency = 0.0

    def _close(self):
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None

    def play(self):
        self.finished = False
        if self.has_audio:
            if self._pos >= self.length:
                self._pos = 0
            self._open()
            self._cb_pos, self._cb_time = self._pos, time.perf_counter()
            self.playing = True
        else:
            self._free_t0 = time.perf_counter()
            self.playing = True

    def pause(self):
        if self.playing and not self.has_audio and self._free_t0 is not None:
            self._free_base += time.perf_counter() - self._free_t0
            self._free_t0 = None
        self.playing = False

    def stop(self):
        self.pause()
        self.seek(0.0)

    def seek(self, t: float):
        with self._lock:
            self._pos = int(max(0.0, t) * self.sr)
            if self.length:
                self._pos = min(self._pos, self.length)
            self._cb_pos, self._cb_time = self._pos, time.perf_counter()
        self._free_base = max(0.0, t)
        if self._free_t0 is not None:
            self._free_t0 = time.perf_counter()

    def time(self) -> float:
        if not self.has_audio:
            if self.playing and self._free_t0 is not None:
                return self._free_base + time.perf_counter() - self._free_t0
            return self._free_base
        if not self.playing:
            return self._pos / self.sr
        # interpolate between callbacks for a smooth clock, compensate output latency
        t = self._cb_pos / self.sr + (time.perf_counter() - self._cb_time) - self.latency
        return float(min(max(t, 0.0), self._pos / self.sr))

    # ------------------------------------------------------------------
    def _callback(self, outdata, frames, time_info, status):
        if not self.playing:
            outdata.fill(0)
            return
        with self._lock:
            p = self._pos
            out = np.zeros((frames, 2), dtype=np.float32)
            for s, g in zip(self.stems, self.gains):
                if g <= 0:
                    continue
                seg = s[p:p + frames]
                if len(seg):
                    out[:len(seg)] += seg * g
            out *= self.master
            np.clip(out, -1.0, 1.0, out=out)
            outdata[:] = out
            self._pos = p + frames
            self._cb_pos, self._cb_time = p, time.perf_counter()
            if self._pos >= self.length:
                self._pos = self.length
                self.playing = False
                self.finished = True

    def close(self):
        self.pause()
        self._close()

