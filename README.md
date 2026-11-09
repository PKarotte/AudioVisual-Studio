# AudioVisual Studio 🌊🎛️

**Audio-Reactive Real-Time GPU Fluid Simulation & Offline Video Renderer**

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-3776ab.svg?logo=python)](https://python.org)
[![OpenGL 3.3+](https://img.shields.io/badge/OpenGL-3.3+-5586a4.svg?logo=opengl)](https://opengl.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

AudioVisual Studio is a Work-in-Progress open-source visual synthesizer and motion graphics tool that transforms multi-track music and audio stems into fluid dynamics. Built with Python, ModernGL (OpenGL 3.3+), and PySide6 (Qt).

---

## 🎬 Showcase

[![Inward — Audio-Reactive GPU Fluid Simulation](https://img.youtube.com/vi/UxZVmMNwRcY/maxresdefault.jpg)](https://www.youtube.com/watch?v=UxZVmMNwRcY)

> 🎥 **[Watch the full showcase on YouTube](https://www.youtube.com/watch?v=UxZVmMNwRcY)**  
> *"Inward" — Music by Daniela Helten & Pascal Karitter, visualized in real time with AudioVisual Studio.*

---

## ✨ Features

- **Multiple GPU Visual Simulations**:
  - **Eulerian Fluid Dynamics (Stable Fluids)**: Incompressible Navier-Stokes solver with semi-Lagrangian advection, Jacobi pressure projection, vorticity confinement, and 3D analytical curl-noise turbulence.
  - **2D Wave Tank & Acoustic Cymatics**: Physical shallow-water wave tank with harmonic acoustic oscillators, standing wave interference, Chladni nodal sand patterns, caustics, and specular Fresnel reflections.
  - **Physarum Slime Mold Network**: Continuous chemotaxis transport network inspired by *Physarum polycephalum*, simulating vein canalization, nutrient foraging, and multi-stem bioluminescent mycelium branching.
  - **Reaction-Diffusion (Gray-Scott)**: Turing pattern synthesis morphing between spots, coral, mitosis, labyrinth mazes, and organic cell division.
- **Background Image Underlay & Blending**:
  - Load custom background artwork (`.png`, `.jpg`, `.webp`, `.bmp`) to place behind or blend with the simulations.
  - Real-time controls for opacity, aspect ratio fitting (*Cover*, *Fit*, *Stretch*), and blend modes (*Underlay*, *Screen*, *Add*, *Multiply*).
- **Multi-Stem Audio Analysis**:
  - Automatically extracts 22 audio features per stem (RMS volume, onset transients, beat pulses, pitch F0, spectral centroid, brightness, rolloff, etc.).
  - Built-in filter bank to split any loaded stereo track into 6 frequency bands (Sub, Low, Low-Mid, Mid, High, Air).
- **Inline Modulation Controls**:
  - Easily modulate any simulation, nozzle/spawner, or post-FX parameter directly next to its slider.
  - Choose sources (assigned spawner stem, master mix audio, global LFOs, or individual stem tracks), adjust modulation depth, toggle spawner scope (*This Spawner* vs *All Spawners*), and fine-tune envelope attack/release and curve response.
  - Dedicated modulation matrix table for full macro routing overview.
- **Nozzle & Jet Spawners**:
  - Interactive placement on canvas or directly in the live viewport.
  - Directional momentum force, dye color injection, sway oscillations, orbital paths, and position jitter.
- **Cinematic Post-Processing**:
  - Real-time 3D liquid normal reconstruction with Fresnel specular reflections and depth shading.
  - Multi-pass dual-blur HDR bloom.
  - ACES filmic tonemapping, exposure, contrast, saturation, and vignette.
  - Milkdrop-style feedback motion trails with zoom and rotation.
- **Production Video Export**:
  - Offline video renderer with 60 Hz physics substepping.
  - Resolutions from 720p to 4K UHD (2160p) with native simulation grid resolution scaling.
  - Supports H.264 (libx264), GPU-accelerated NVENC (H.264/H.265), ProRes 422 HQ, ProRes 4444 (12-bit), and uncompressed PNG frame sequences.
  - Pre-roll warmup so exported videos match the live preview accurately from frame 0.

---

## 🚀 Quick Start

### 1. Requirements

- Python 3.10 or newer
- OpenGL 3.3 capable GPU
- [FFmpeg](https://ffmpeg.org/) (for video export)

### 2. Installation

Clone this repository and install the Python dependencies:

```bash
git clone https://github.com/PKarotte/AudioVisual-Studio.git
cd AudioVisual-Studio
pip install -r requirements.txt
```

### 3. Launching

On Windows, double-click `start.bat` or run:

```bash
python run.py
```

---

## 🎛️ How to Use

1. **Load Music**: Click the **Audio Stems** tab on the right sidebar and load one or more audio files (`.wav`, `.mp3`, `.flac`). You can also click **Split Loaded Track Into 6 Frequency Bands** to automatically create frequency-isolated modulation sources.
2. **Place Spawners**: In the **Spawners** tab or **Spawner Placement & Grid** view, create and position fluid nozzles. Assign each spawner to an audio stem or let them react to the master mix.
3. **Modulate Parameters**: In the **Simulation** or **Spawners** tabs, click the **⚡ Mod** dropdown below any parameter slider to map it to Volume (RMS), Beat pulse, Onset transients, or an LFO.
4. **Adjust Fluid Looks**: In the **Simulation** tab, adjust Vorticity, Turbulence, or Dye fade time. In **Post FX**, dial in liquid shading, bloom, and tonemapping.
5. **Export Video**: Click **🎬 Export Video** in the transport bar, choose your resolution (up to 4K) and codec, and hit **Start Render**.

---

## 🏛️ Architecture & Acknowledgments

- **Jos Stam**: *Stable Fluids* (SIGGRAPH 1999) — Unconditionally stable semi-Lagrangian advection algorithm.
- **Pavel Dobryakov**: WebGL Fluid simulation shader pipeline inspiration.
- **Stefan Gustavson & Ashima Arts**: 3D Simplex noise and Curl Noise implementations (MIT License).
- **Ronald Fedkiw et al.**: Vorticity confinement method for retaining small turbulent eddies.

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
