"""
2D Wave Tank & Acoustic Cymatics Simulation.

Simulates a physical shallow-water wave tank with acoustic transducer oscillators
and standing-wave cymatics (Chladni resonance). Spawners act as vibrating
oscillators and wave emitters driven by audio stem frequencies, forces, and transients.
Visualized with realistic 3D specular liquid Fresnel reflections, caustics, and nodal lines.
"""
from __future__ import annotations

import numpy as np

from ..gl_common import SPAWNER_UNIFORMS, Double, Target, pack_spawners
from ..params import Param
from .base import Simulation, register

WAVE_FS = SPAWNER_UNIFORMS + """
uniform sampler2D uWave;
uniform vec2 texelSize;
uniform float dt, waveSpeed, damping, viscosity, reflectivity;
uniform float time, baseFreq, impulseGain;

void main(){
    vec2 tx = texelSize;
    float c_h = texture(uWave, vUv).r;
    float h_l = texture(uWave, vUv - vec2(tx.x, 0.0)).r;
    float h_r = texture(uWave, vUv + vec2(tx.x, 0.0)).r;
    float h_t = texture(uWave, vUv + vec2(0.0, tx.y)).r;
    float h_b = texture(uWave, vUv - vec2(0.0, tx.y)).r;
    float h_tl = texture(uWave, vUv + vec2(-tx.x, tx.y)).r;
    float h_tr = texture(uWave, vUv + vec2(tx.x, tx.y)).r;
    float h_bl = texture(uWave, vUv + vec2(-tx.x, -tx.y)).r;
    float h_br = texture(uWave, vUv + vec2(tx.x, -tx.y)).r;

    // 9-point isotropic Laplacian (eliminates diagonal grid anisotropy)
    float lap = (0.5 * (h_l + h_r + h_t + h_b) + 0.25 * (h_tl + h_tr + h_bl + h_br) - 3.0 * c_h);

    float vel = texture(uWave, vUv).g;

    // Viscosity / high-frequency dispersion
    vel += viscosity * lap * 50.0 * dt;

    // Wave equation acceleration: d^2h/dt^2 = c^2 * lap
    float c2 = waveSpeed * waveSpeed;
    vel += c2 * lap * 120.0 * dt;

    // Acoustic transducer oscillator excitation from spawners
    for (int i = 0; i < spCount; i++) {
        float g = spFalloff(i, vUv);
        if (g > 0.0005) {
            float spFreq = baseFreq * (1.0 + float(i) * 0.28);
            float osc = sin(time * spFreq * 6.2831853 + float(i) * 1.5708);
            float force = spA[i].w * 0.01; // 0..1
            
            // Harmonic wave transducer oscillation
            float impulse = osc * force * impulseGain * 3.5;
            
            // Directional traveling wave impulse
            vec2 spDir = spB[i].xy;
            vec2 toUv = vUv - spA[i].xy;
            toUv.x *= aspect;
            float len = length(toUv);
            if (len > 1e-4) {
                float dirAlign = dot(spDir, toUv / len);
                impulse += dirAlign * force * impulseGain * 2.0;
            }
            
            vel += impulse * g * 60.0 * dt;
        }
    }

    // Viscous damping
    float damp = clamp(1.0 - damping * dt * 15.0, 0.0, 1.0);

    // Boundary conditions: reflective walls (Cymatics / Chladni) vs absorptive sponge layer
    vec2 border = min(vUv, 1.0 - vUv);
    float edge = min(border.x, border.y);
    float edgeMargin = 0.04;
    if (edge < edgeMargin) {
        float f = edge / edgeMargin;
        damp *= mix(f * f, 1.0, reflectivity);
    }

    vel *= damp;
    vel = clamp(vel, -5.0, 5.0);
    c_h += vel * dt;
    c_h = clamp(c_h, -3.0, 3.0);

    fragColor = vec4(c_h, vel, 0.0, 1.0);
}
"""

DYE_FS = SPAWNER_UNIFORMS + """
uniform sampler2D uColor;
uniform sampler2D uWave;
uniform vec2 texelSize;
uniform float dt, dyeDecay, colorRate, drift;

void main(){
    vec2 tx = texelSize;
    float h_l = texture(uWave, vUv - vec2(tx.x, 0.0)).r;
    float h_r = texture(uWave, vUv + vec2(tx.x, 0.0)).r;
    float h_t = texture(uWave, vUv + vec2(0.0, tx.y)).r;
    float h_b = texture(uWave, vUv - vec2(0.0, tx.y)).r;

    // Stokes surface drift along negative gradient of wave elevation
    vec2 waveGrad = vec2(h_r - h_l, h_t - h_b);
    vec2 flow = -waveGrad * drift * 15.0;

    // Additional spawner directional push
    for (int i = 0; i < spCount; i++) {
        flow += spB[i].xy * spA[i].w * 0.01 * spFalloff(i, vUv) * 5.0;
    }

    vec2 uvPrev = vUv - flow * dt * texelSize * 20.0;
    vec3 col = texture(uColor, uvPrev).rgb;

    // Subtle diffusion blur
    vec3 blur = 0.25 * (texture(uColor, uvPrev + vec2(tx.x, 0)).rgb + texture(uColor, uvPrev - vec2(tx.x, 0)).rgb
                      + texture(uColor, uvPrev + vec2(0, tx.y)).rgb + texture(uColor, uvPrev - vec2(0, tx.y)).rgb);
    col = mix(col, blur, 0.12);

    // Dye lifetime fading
    col *= clamp(1.0 - dyeDecay * dt, 0.0, 1.0);

    // Spawner color deposition
    for (int i = 0; i < spCount; i++) {
        float g = spFalloff(i, vUv);
        float dye = spB[i].z; // dye amount
        if (g > 0.001 && dye > 0.0) {
            float rate = clamp(g * dye * colorRate * dt * 8.0, 0.0, 1.0);
            col = mix(col, spC[i].rgb, rate);
        }
    }

    fragColor = vec4(col, 1.0);
}
"""

OUT_FS = """
uniform sampler2D uWave;
uniform sampler2D uColor;
uniform vec2 texelSize;
uniform float surfaceDepth, specular, caustics, cymatics, gain;

void main(){
    vec2 tx = texelSize;
    vec2 s = texture(uWave, vUv).rg;
    float h = s.r;
    float vel = s.g;

    float h_l = texture(uWave, vUv - vec2(tx.x, 0.0)).r;
    float h_r = texture(uWave, vUv + vec2(tx.x, 0.0)).r;
    float h_t = texture(uWave, vUv + vec2(0.0, tx.y)).r;
    float h_b = texture(uWave, vUv - vec2(0.0, tx.y)).r;

    // 3D Liquid normal vector
    float dx = (h_r - h_l) * surfaceDepth * 6.0;
    float dy = (h_t - h_b) * surfaceDepth * 6.0;
    vec3 N = normalize(vec3(-dx, -dy, 1.0));

    // Directional light from top-left
    vec3 L = normalize(vec3(-0.35, 0.45, 0.8));
    vec3 V = vec3(0.0, 0.0, 1.0);
    vec3 H = normalize(L + V);

    float NdotL = clamp(dot(N, L), 0.0, 1.0);
    float diffuse = NdotL * 0.45 + 0.55;
    float spec = pow(clamp(dot(N, H), 0.0, 1.0), 36.0) * specular;
    float fresnel = pow(1.0 - clamp(dot(N, V), 0.0, 1.0), 2.5);

    // Wave curvature caustics
    float lap = (h_l + h_r + h_t + h_b - 4.0 * h);
    float causticLight = clamp(1.0 - lap * caustics * 15.0, 0.2, 3.5);

    vec3 dye = texture(uColor, vUv).rgb;
    float dyeLum = dot(dye, vec3(0.299, 0.587, 0.114));

    // Base water body color with caustic illumination
    vec3 waterBase = vec3(0.015, 0.035, 0.06) * causticLight;
    vec3 col = mix(waterBase, dye * causticLight, clamp(dyeLum * 2.0, 0.0, 1.0));

    // Liquid shading and highlights
    col = col * diffuse + vec3(0.9, 0.95, 1.0) * (spec + fresnel * 0.35);

    // Cymatics Nodal Lines Mode:
    // Sand particles settle at vibration nodes (zero kinetic + potential energy)
    if (cymatics > 0.0) {
        float energy = h * h * 4.0 + vel * vel * 2.0;
        float nodeLine = exp(-energy * 35.0);
        vec3 sandCol = vec3(1.0, 0.94, 0.82) * 1.5;
        col = mix(col, sandCol, nodeLine * cymatics);
    }

    fragColor = vec4(col * gain, 1.0);
}
"""


@register
class WaveTankSim(Simulation):
    name = "2D Wave Tank & Cymatics"
    description = (
        "Physical 2D acoustic wave tank with standing wave cymatics (Chladni resonance). "
        "Audio stems act as acoustic oscillators creating wave interference patterns."
    )
    params = [
        Param("wave_speed", "Wave Speed", 0.1, 2.0, 0.9, group="Acoustics",
              tooltip="Speed of wave propagation across the tank"),
        Param("damping", "Damping / Viscosity", 0.001, 0.08, 0.015, group="Acoustics",
              tooltip="Dissipation rate of wave energy"),
        Param("base_freq", "Vibration Freq (Hz)", 0.2, 12.0, 3.2, group="Acoustics",
              tooltip="Base acoustic oscillator frequency"),
        Param("impulse_gain", "Impulse Strength", 0.0, 4.0, 1.6, group="Acoustics",
              tooltip="Audio force / transient excitation strength"),
        Param("viscosity", "Surface Tension", 0.0, 0.3, 0.06, group="Acoustics",
              tooltip="High-frequency capillary dispersion"),
        Param("reflectivity", "Tank Boundary Reflection", 0.0, 1.0, 0.85, group="Acoustics",
              tooltip="1.0 = Rigid walls (Chladni standing waves), 0.0 = Open absorptive tank"),
        Param("cymatics", "Cymatics Nodal Pattern", 0.0, 1.0, 0.0, group="Cymatics",
              tooltip="Visualizes acoustic standing wave nodal lines (Chladni sand patterns)"),
        Param("surface_depth", "Surface Relief (3D)", 0.1, 5.0, 1.6, group="Look",
              tooltip="Steepness of 3D wave normal shading"),
        Param("specular", "Specular Gleam", 0.0, 3.0, 1.2, group="Look",
              tooltip="Glints and specular sunlight reflections"),
        Param("caustics", "Light Caustics", 0.0, 2.0, 0.7, group="Look",
              tooltip="Refraction caustic illumination"),
        Param("drift", "Stokes Drift", 0.0, 3.0, 0.7, group="Motion",
              tooltip="Surface water drift carrying color"),
        Param("dye_decay", "Dye Fading Rate", 0.001, 0.5, 0.035, group="Color",
              tooltip="Rate at which dye clears"),
        Param("color_rate", "Color Pickup", 0.1, 5.0, 1.5, group="Color",
              tooltip="Speed at which water absorbs spawner color"),
        Param("gain", "Brightness Gain", 0.1, 4.0, 1.3, group="Look"),
        Param("sim_res", "Grid (height)", 128, 1080, 512, step=16, integer=True, modulatable=False,
              reinit=True, group="Quality", tooltip="Simulation grid resolution"),
    ]

    def __init__(self, gl):
        super().__init__(gl)
        self.p_wave = gl.pass_(WAVE_FS)
        self.p_dye = gl.pass_(DYE_FS)
        self.p_out = gl.pass_(OUT_FS)
        self.targets = []
        self._acc = 0.0

    def setup(self, out_w, out_h, values, render_mode):
        self._free()
        self.aspect = out_w / out_h
        sh = int(values.get("sim_res", 512))
        sw = max(8, int(round(sh * self.aspect)))
        self.wave = Double(self.ctx, sw, sh, 4, "f4")
        self.color = Double(self.ctx, sw, sh, 4, "f2")
        self.out = Target(self.ctx, sw, sh, 4, "f2")
        self.targets = [self.wave, self.color, self.out]
        self.width, self.height = out_w, out_h
        self.reset()

    def reset(self):
        self.wave.clear(0, 0, 0, 1)
        self.color.clear(0, 0, 0, 1)
        self._acc = 0.0

    def step(self, dt, t, v, spawners):
        if dt <= 0:
            return
        A, B, C, n = pack_spawners(spawners)
        sp = dict(spA=A, spB=B, spC=C, spCount=n, aspect=self.aspect)
        tx = self.wave.texel

        # Multi-substep physics for unconditional Courant numerical stability
        substeps = 4
        dt_sub = min(dt, 0.05) / substeps

        for sub in range(substeps):
            t_sub = t + sub * dt_sub
            self.p_wave.run(
                self.wave.write.fbo,
                dict(uWave=self.wave.read.tex),
                texelSize=tx,
                dt=dt_sub,
                waveSpeed=v["wave_speed"],
                damping=v["damping"],
                viscosity=v["viscosity"],
                reflectivity=v["reflectivity"],
                time=t_sub,
                baseFreq=v["base_freq"],
                impulseGain=v["impulse_gain"],
                **sp,
            )
            self.wave.swap()

        # Advect and inject dye
        self.p_dye.run(
            self.color.write.fbo,
            dict(uColor=self.color.read.tex, uWave=self.wave.read.tex),
            texelSize=tx,
            dt=min(dt, 0.05),
            dyeDecay=v["dye_decay"],
            colorRate=v["color_rate"],
            drift=v["drift"],
            **sp,
        )
        self.color.swap()

        # Render shaded output
        self.p_out.run(
            self.out.fbo,
            dict(uWave=self.wave.read.tex, uColor=self.color.read.tex),
            texelSize=tx,
            surfaceDepth=v["surface_depth"],
            specular=v["specular"],
            caustics=v["caustics"],
            cymatics=v["cymatics"],
            gain=v["gain"],
        )

    def output(self):
        return self.out.tex

    def _free(self):
        for t in self.targets:
            t.release()
        self.targets = []

    def release(self):
        self._free()
        for p in (self.p_wave, self.p_dye, self.p_out):
            p.release()
