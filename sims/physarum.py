"""
Physarum Polycephalum (Slime Mold / Organic Transport Network).

Simulates continuous chemotactic transport networks, vein canalization, and nutrient
foraging inspired by the slime mold Physarum polycephalum. Spawners act as audio-reactive
nutrient sources and spore emitters, causing living multi-colored bioluminescent veins
to sprout, branch, and interconnect across the screen.
"""
from __future__ import annotations

import numpy as np

from ..gl_common import SPAWNER_UNIFORMS, Double, Target, pack_spawners
from ..params import Param
from .base import Simulation, register

PHYSARUM_FS = SPAWNER_UNIFORMS + """
uniform sampler2D uState;
uniform vec2 texelSize;
uniform float dt, trailDiffuse, trailDecay, chemotaxis, canalization, decay, inject, drift;

void main(){
    vec2 tx = texelSize;
    vec4 s_c = texture(uState, vUv);
    float T = s_c.r;   // Chemoattractant / pheromone trail
    float rho = s_c.g; // Living biomass density

    // Neighbors
    float T_l = texture(uState, vUv - vec2(tx.x, 0.0)).r;
    float T_r = texture(uState, vUv + vec2(tx.x, 0.0)).r;
    float T_t = texture(uState, vUv + vec2(0.0, tx.y)).r;
    float T_b = texture(uState, vUv - vec2(0.0, tx.y)).r;

    float rho_l = texture(uState, vUv - vec2(tx.x, 0.0)).g;
    float rho_r = texture(uState, vUv + vec2(tx.x, 0.0)).g;
    float rho_t = texture(uState, vUv + vec2(0.0, tx.y)).g;
    float rho_b = texture(uState, vUv - vec2(0.0, tx.y)).g;

    // Trail gradient (chemotaxis sensory field)
    vec2 gradT = vec2(T_r - T_l, T_t - T_b) * 0.5;

    // Chemotactic advection velocity
    vec2 vel = gradT * chemotaxis * 40.0;

    // Spawner directional momentum injection
    for (int i = 0; i < spCount; i++) {
        float g = spFalloff(i, vUv);
        vel += spB[i].xy * (spA[i].w * 0.01) * drift * g * 15.0;
    }

    // Advection of biomass along velocity gradient
    vec2 prevUv = vUv - vel * dt * texelSize * 25.0;
    float rho_adv = texture(uState, prevUv).g;

    // Laplacian diffusion of trail pheromone
    float lapT = (T_l + T_r + T_t + T_b - 4.0 * T);
    // Laplacian diffusion of biomass
    float lapRho = (rho_l + rho_r + rho_t + rho_b - 4.0 * rho);

    // Trail equation: diffusion + biomass deposition - evaporation
    float dT = trailDiffuse * lapT * 12.0 + rho * 1.5 - trailDecay * T;
    T += dT * dt;

    // Vein canalization (positive feedback): active thick veins dilate and stabilize
    float canalize = canalization * rho_adv * rho_adv * (1.0 - rho_adv) * 3.5;
    float dRho = 0.3 * lapRho + canalize - decay * rho_adv;
    rho = rho_adv + dRho * dt;

    // Spawner injection (nutrients & fresh spores)
    for (int i = 0; i < spCount; i++) {
        float g = spFalloff(i, vUv);
        float force = spA[i].w * 0.01;
        float dye = spB[i].z;
        float pulse = (force * 0.6 + dye * 0.4) * inject;
        T += g * pulse * 3.0 * dt;
        rho += g * pulse * 4.0 * dt;
    }

    T = clamp(T, 0.0, 4.0);
    rho = clamp(rho, 0.0, 1.5);

    fragColor = vec4(T, rho, length(vel), 1.0);
}
"""

COLOR_FS = SPAWNER_UNIFORMS + """
uniform sampler2D uColor;
uniform sampler2D uState;
uniform vec2 texelSize;
uniform float dt, colorRate, chemotaxis, drift;

void main(){
    vec2 tx = texelSize;
    vec4 s = texture(uState, vUv);
    float T_l = texture(uState, vUv - vec2(tx.x, 0.0)).r;
    float T_r = texture(uState, vUv + vec2(tx.x, 0.0)).r;
    float T_t = texture(uState, vUv + vec2(0.0, tx.y)).r;
    float T_b = texture(uState, vUv - vec2(0.0, tx.y)).r;

    vec2 vel = vec2(T_r - T_l, T_t - T_b) * 0.5 * chemotaxis * 40.0;
    for (int i = 0; i < spCount; i++) {
        vel += spB[i].xy * (spA[i].w * 0.01) * drift * spFalloff(i, vUv) * 15.0;
    }

    vec2 uvPrev = vUv - vel * dt * texelSize * 25.0;
    vec3 col = texture(uColor, uvPrev).rgb;

    // Soft lateral diffusion
    vec3 blur = 0.25 * (texture(uColor, uvPrev + vec2(tx.x, 0)).rgb + texture(uColor, uvPrev - vec2(tx.x, 0)).rgb
                      + texture(uColor, uvPrev + vec2(0, tx.y)).rgb + texture(uColor, uvPrev - vec2(0, tx.y)).rgb);
    col = mix(col, blur, 0.08);

    // Spawner color seeding
    for (int i = 0; i < spCount; i++) {
        float g = spFalloff(i, vUv);
        float force = spA[i].w * 0.01;
        float dye = spB[i].z;
        float amt = clamp(g * (force + dye) * colorRate * dt * 8.0, 0.0, 1.0);
        col = mix(col, spC[i].rgb, amt);
    }

    fragColor = vec4(col, 1.0);
}
"""

OUT_FS = """
uniform sampler2D uState;
uniform sampler2D uColor;
uniform vec2 texelSize;
uniform float glow, coreHeat, gain;

void main(){
    vec2 tx = texelSize;
    vec4 s = texture(uState, vUv);
    float T = s.r;
    float rho = s.g;

    vec3 col = texture(uColor, vUv).rgb;

    // Smooth vein shape mask
    float veinMask = smoothstep(0.08, 0.65, rho);
    float trailGlow = smoothstep(0.05, 0.9, T) * 0.35;

    // Bioluminescent core: dense veins burn with bright incandescent light
    float hotCore = smoothstep(0.55, 1.1, rho) * coreHeat;
    vec3 coreColor = vec3(1.0, 0.98, 0.92) * hotCore * 2.0;

    // Organic composite
    vec3 finalCol = col * (veinMask + trailGlow) * glow + coreColor;

    fragColor = vec4(finalCol * gain, 1.0);
}
"""


@register
class PhysarumSim(Simulation):
    name = "Physarum (Slime Mold Network)"
    description = (
        "Continuous organic transport network based on Physarum polycephalum chemotaxis. "
        "Audio stems act as nutrient sources, sprouting pulsating bioluminescent veins."
    )
    params = [
        Param("chemotaxis", "Chemotaxis Speed", 0.2, 4.0, 1.5, group="Biology",
              tooltip="Rate at which organisms advance along nutrient trails"),
        Param("canalization", "Vein Canalization", 0.5, 4.0, 1.8, group="Biology",
              tooltip="Nonlinear vein reinforcement and branching sharpness"),
        Param("trail_diffuse", "Trail Diffusion", 0.1, 2.5, 1.0, group="Biology",
              tooltip="Diffusion speed of pheromone chemoattractant"),
        Param("trail_decay", "Trail Evaporation", 0.01, 0.4, 0.08, group="Biology",
              tooltip="Evaporation rate of unused trails"),
        Param("decay", "Biomass Decay", 0.005, 0.2, 0.035, group="Biology",
              tooltip="Natural pruning of inactive branches"),
        Param("inject", "Nutrient Injection", 0.1, 5.0, 1.8, group="Spawners",
              tooltip="Nutrient emission from audio stems"),
        Param("drift", "Spawner Drift", 0.0, 4.0, 1.2, group="Spawners",
              tooltip="Directional push along spawner heading"),
        Param("glow", "Bioluminescence Glow", 0.2, 4.0, 1.6, group="Look",
              tooltip="Intensity of bioluminescent vein emission"),
        Param("core_heat", "Hot Core Intensity", 0.0, 2.0, 0.85, group="Look",
              tooltip="White/hot light at the center of thick veins"),
        Param("color_rate", "Color Pickup", 0.1, 5.0, 1.6, group="Color",
              tooltip="How fast veins take on spawner stem colors"),
        Param("gain", "Brightness Gain", 0.1, 4.0, 1.3, group="Look"),
        Param("sim_res", "Grid (height)", 128, 1080, 480, step=16, integer=True, modulatable=False,
              reinit=True, group="Quality", tooltip="Simulation grid resolution"),
    ]

    def __init__(self, gl):
        super().__init__(gl)
        self.p_phys = gl.pass_(PHYSARUM_FS)
        self.p_col = gl.pass_(COLOR_FS)
        self.p_out = gl.pass_(OUT_FS)
        self.targets = []

    def setup(self, out_w, out_h, values, render_mode):
        self._free()
        self.aspect = out_w / out_h
        sh = int(values.get("sim_res", 480))
        sw = max(8, int(round(sh * self.aspect)))
        self.state = Double(self.ctx, sw, sh, 4, "f2")
        self.color = Double(self.ctx, sw, sh, 4, "f2")
        self.out = Target(self.ctx, sw, sh, 4, "f2")
        self.targets = [self.state, self.color, self.out]
        self.width, self.height = out_w, out_h
        self.reset()

    def reset(self):
        w, h = self.state.w, self.state.h
        s = np.zeros((h, w, 4), dtype="f2")
        rng = np.random.default_rng(42)
        # Seed a few small starter spores across the grid
        for _ in range(16):
            cx, cy, r = rng.integers(0, w), rng.integers(0, h), rng.integers(4, 12)
            s[max(0, cy - r):min(h, cy + r), max(0, cx - r):min(w, cx + r), 0] = 0.8
            s[max(0, cy - r):min(h, cy + r), max(0, cx - r):min(w, cx + r), 1] = 0.5
        self.state.read.tex.write(s.tobytes())
        self.state.write.tex.write(s.tobytes())
        self.color.clear(0.1, 0.1, 0.1, 1)

    def step(self, dt, t, v, spawners):
        if dt <= 0:
            return
        A, B, C, n = pack_spawners(spawners)
        sp = dict(spA=A, spB=B, spC=C, spCount=n, aspect=self.aspect)
        tx = self.state.texel

        # Substep for continuous smooth vein diffusion and advection
        substeps = 2
        dt_sub = min(dt, 0.05) / substeps

        for _ in range(substeps):
            self.p_phys.run(
                self.state.write.fbo,
                dict(uState=self.state.read.tex),
                texelSize=tx,
                dt=dt_sub,
                trailDiffuse=v["trail_diffuse"],
                trailDecay=v["trail_decay"],
                chemotaxis=v["chemotaxis"],
                canalization=v["canalization"],
                decay=v["decay"],
                inject=v["inject"],
                drift=v["drift"],
                **sp,
            )
            self.state.swap()

        self.p_col.run(
            self.color.write.fbo,
            dict(uColor=self.color.read.tex, uState=self.state.read.tex),
            texelSize=tx,
            dt=min(dt, 0.05),
            colorRate=v["color_rate"],
            chemotaxis=v["chemotaxis"],
            drift=v["drift"],
            **sp,
        )
        self.color.swap()

        self.p_out.run(
            self.out.fbo,
            dict(uState=self.state.read.tex, uColor=self.color.read.tex),
            texelSize=tx,
            glow=v["glow"],
            coreHeat=v["core_heat"],
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
        for p in (self.p_phys, self.p_col, self.p_out):
            p.release()
