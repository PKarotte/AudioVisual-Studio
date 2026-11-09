"""Offline renderer: runs the engine in its own (standalone) GL context and pipes frames to ffmpeg."""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import time

import numpy as np
import soundfile as sf

CODECS = {
    "H.264 (x264)": dict(ext=".mp4", v=["-c:v", "libx264", "-preset", "slow", "-crf", "{q}", "-pix_fmt", "yuv420p"],
                         a=["-c:a", "aac", "-b:a", "320k"]),
    "H.264 NVENC (GPU, fast)": dict(ext=".mp4", v=["-c:v", "h264_nvenc", "-preset", "p6", "-rc", "vbr", "-cq", "{q}",
                                                    "-b:v", "0", "-pix_fmt", "yuv420p"],
                                    a=["-c:a", "aac", "-b:a", "320k"]),
    "H.265 NVENC (GPU)": dict(ext=".mp4", v=["-c:v", "hevc_nvenc", "-preset", "p6", "-rc", "vbr", "-cq", "{q}",
                                              "-b:v", "0", "-pix_fmt", "yuv420p", "-tag:v", "hvc1"],
                              a=["-c:a", "aac", "-b:a", "320k"]),
    "ProRes 422 HQ": dict(ext=".mov", v=["-c:v", "prores_ks", "-profile:v", "3", "-pix_fmt", "yuv422p10le"],
                          a=["-c:a", "pcm_s24le"]),
    "ProRes 4444 (Maximum Fidelity 12-bit)": dict(ext=".mov", v=["-c:v", "prores_ks", "-profile:v", "4", "-pix_fmt", "yuva444p10le"],
                                                  a=["-c:a", "pcm_s24le"]),
    "PNG sequence": dict(ext="", v=[], a=[]),
}


def find_ffmpeg() -> str | None:
    p = shutil.which("ffmpeg")
    if p:
        return p
    # winget default location
    base = os.path.expandvars(r"%LOCALAPPDATA%\Microsoft\WinGet\Packages")
    if os.path.isdir(base):
        for root, _dirs, files in os.walk(base):
            if "ffmpeg.exe" in files:
                return os.path.join(root, "ffmpeg.exe")
    return None


def mix_audio(stems: list[np.ndarray], gains: list[float], master: float) -> np.ndarray:
    n = max((len(s) for s in stems), default=0)
    out = np.zeros((n, 2), np.float32)
    for s, g in zip(stems, gains):
        if g > 0:
            out[:len(s)] += s * g
    out *= master
    peak = float(np.abs(out).max()) if n else 0.0
    if peak > 1.0:
        out /= peak
    return out


def render(project_dict: dict, bank, duration: float, audio_stems: list[np.ndarray], gains: list[float],
           sr: int, settings: dict, progress=None, cancelled=lambda: False, log=print) -> str:
    """Blocking render. `progress(done, total, fps)` is called periodically. Returns the output path."""
    import moderngl
    from .engine import Engine
    from .project import Project

    ffmpeg = find_ffmpeg()
    if not ffmpeg:
        raise RuntimeError("ffmpeg not found. Install it (winget install Gyan.FFmpeg) or add it to PATH.")

    w, h = int(settings["width"]) // 2 * 2, int(settings["height"]) // 2 * 2
    fps = float(settings["fps"])
    codec = CODECS[settings["codec"]]
    start = max(0.0, float(settings.get("start", 0.0)))
    end = float(settings.get("end", 0.0)) or duration or 10.0
    preroll = max(0.0, float(settings.get("preroll", 0.0)))
    total = max(1, int(round((end - start) * fps)))
    out_path = settings["path"]

    ctx = moderngl.create_standalone_context(require=330)
    project = Project.from_dict(project_dict)

    # Configure simulation resolution for export
    sim_q = settings.get("sim_quality", "Auto (Match Video Height for Max Quality)")
    target_sim_res = None
    if "Auto" in sim_q:
        target_sim_res = h
    elif "1440" in sim_q:
        target_sim_res = 1440
    elif "1080" in sim_q:
        target_sim_res = 1080
    elif "768" in sim_q:
        target_sim_res = 768
    elif "512" in sim_q:
        target_sim_res = 512

    if target_sim_res:
        project.sim_values.setdefault(project.sim_name, {})["sim_res"] = target_sim_res
        cur_iter = project.sim_values[project.sim_name].get("iterations", 28)
        if target_sim_res >= 1080 and cur_iter < 35:
            project.sim_values[project.sim_name]["iterations"] = 35

    engine = Engine(ctx, project, bank, duration, render_mode=True)
    engine.set_output_size(w, h)

    tmpdir = tempfile.mkdtemp(prefix="liquid_render_")
    wav = None
    if audio_stems:
        mix = mix_audio(audio_stems, gains, project.master_volume)
        seg = mix[int(start * sr): int(end * sr)]
        if len(seg) > 0:
            wav = os.path.join(tmpdir, "audio.wav")
            sf.write(wav, seg, sr, subtype="PCM_24")

    q = str(int(settings.get("quality", 18)))
    cmd = [ffmpeg, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}",
           "-r", f"{fps}", "-i", "-"]
    if settings["codec"] == "PNG sequence":
        os.makedirs(out_path, exist_ok=True)
        cmd += ["-vf", "vflip", os.path.join(out_path, "frame_%06d.png")]
        if wav:
            shutil.copy(wav, os.path.join(out_path, "audio.wav"))
    else:
        if wav:
            cmd += ["-i", wav]
        cmd += ["-vf", "vflip"] + [a.replace("{q}", q) for a in codec["v"]]
        if wav:
            cmd += ["-map", "0:v", "-map", "1:a"] + codec["a"] + ["-shortest"]
        cmd += [out_path]
    log("ffmpeg: " + " ".join(cmd))
    errlog = open(os.path.join(tmpdir, "ffmpeg.log"), "w+")
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=errlog,
                            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    dt = 1.0 / fps
    substeps = max(1, int(round(60.0 / fps)))
    dt_sub = dt / substeps
    try:
        # pre-roll: simulate before start point (if start > 0) so the fluid is already "alive"
        if start > 0.0 and preroll > 0.0:
            t_pre_start = max(0.0, start - preroll)
            n_pre = int(round((start - t_pre_start) * fps))
            for i in range(n_pre):
                if cancelled():
                    raise InterruptedError
                f_t = t_pre_start + i * dt
                for s in range(substeps):
                    engine.frame(f_t + s * dt_sub, dt_sub)
        t0 = time.perf_counter()
        for i in range(total):
            if cancelled():
                raise InterruptedError
            f_t = start + i * dt
            for s in range(substeps):
                engine.frame(f_t + s * dt_sub, dt_sub)
            proc.stdin.write(engine.read_rgb())
            if progress and (i % 5 == 0 or i == total - 1):
                el = time.perf_counter() - t0
                progress(i + 1, total, (i + 1) / el if el > 0 else 0.0)
        proc.stdin.close()
        proc.wait()
        if proc.returncode != 0:
            errlog.seek(0)
            raise RuntimeError("ffmpeg failed:\n" + errlog.read()[-2000:])
    except (InterruptedError, BrokenPipeError) as e:
        try:
            proc.stdin.close()
        except Exception:
            pass
        proc.kill()
        if isinstance(e, BrokenPipeError):
            errlog.seek(0)
            raise RuntimeError("ffmpeg stopped:\n" + errlog.read()[-2000:])
        raise
    finally:
        errlog.close()
        engine.release()
        ctx.release()
        shutil.rmtree(tmpdir, ignore_errors=True)
    return out_path

