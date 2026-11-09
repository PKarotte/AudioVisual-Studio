"""
Gray-Scott reaction-diffusion.

Spawners inject chemical B (dye) and push the pattern along their direction
(force => drift).  A separate colour field picks up spawner colours so the
growing structures inherit the colour of whatever stem "seeded" them.
Modulating feed/kill with the music morphs between spots, stripes, coral,
mitosis and worm-like regimes.
"""
from __future__ import annotations

import numpy as np

from ..gl_common import SPAWNER_UNIFORMS, Double, Target, pack_spawners
from ..params import Param
from .base import Simulation, register

RD_FS = SPAWNER_UNIFORMS + """
uniform sampler2D uState; uniform vec2 texelSize;
uniform float feed, kill, Du, Dv, h, inject, drift, swirl;
void main(){
    vec2 flow = vec2(0.0); float add = 0.0;
    for (int i = 0; i < spCount; i++) {
        float g = spFalloff(i, vUv);
        flow += spB[i].xy * spA[i].w * g;
        add += spB[i].z * g;
    }
    // swirl: rotate flow around the screen centre for slow global motion
    vec2 c = vUv - 0.5; c.x *= aspect;
    flow += vec2(-c.y, c.x) * swirl * 20.0;
    vec2 uv = vUv - flow * drift * texelSize * h * 0.05;
    vec2 s  = texture(uState, uv).xy;
    vec2 tx = texelSize;
    vec2 lap = -s
        + 0.2  * (texture(uState, uv + vec2(tx.x, 0)).xy + texture(uState, uv - vec2(tx.x, 0)).xy
               +  texture(uState, uv + vec2(0, tx.y)).xy + texture(uState, uv - vec2(0, tx.y)).xy)
        + 0.05 * (texture(uState, uv + tx).xy + texture(uState, uv - tx).xy
               +  texture(uState, uv + vec2(tx.x, -tx.y)).xy + texture(uState, uv + vec2(-tx.x, tx.y)).xy);
    float u = s.x, v = s.y;
    float uvv = u * v * v;
    float du = Du * lap.x - uvv + feed * (1.0 - u);
    float dv = Dv * lap.y + uvv - (kill + feed) * v;
    float inj = clamp(add * inject * h * 0.05, 0.0, 1.0);
    u += du * h - inj * u * 0.5;
    v += dv * h + inj * (1.0 - v);
    fragColor = vec4(clamp(u, 0.0, 1.0), clamp(v, 0.0, 1.0), 0.0, 1.0);
}
"""

COLOR_FS = SPAWNER_UNIFORMS + """
uniform sampler2D uColor; uniform sampler2D uState; uniform vec2 texelSize;
uniform float dt, colorDiffuse, colorRate, drift, swirl;
void main(){
    vec2 flow = vec2(0.0);
    for (int i = 0; i < spCount; i++) flow += spB[i].xy * spA[i].w * spFalloff(i, vUv);
    vec2 cc = vUv - 0.5; cc.x *= aspect;
    flow += vec2(-cc.y, cc.x) * swirl * 20.0;
    vec2 uv = vUv - flow * drift * texelSize * dt * 0.6;
    vec2 tx = texelSize * (1.0 + colorDiffuse * 3.0);
    vec3 col = texture(uColor, uv).rgb;
    vec3 blur = 0.25 * (texture(uColor, uv + vec2(tx.x, 0)).rgb + texture(uColor, uv - vec2(tx.x, 0)).rgb
                      + texture(uColor, uv + vec2(0, tx.y)).rgb + texture(uColor, uv - vec2(0, tx.y)).rgb);
    col = mix(col, blur, clamp(colorDiffuse, 0.0, 1.0));
    for (int i = 0; i < spCount; i++) {
        float g = spFalloff(i, vUv) * spB[i].z;
        col = mix(col, spC[i].rgb, clamp(g * colorRate * dt * 4.0, 0.0, 1.0));
    }
    fragColor = vec4(col, 1.0);
}
"""

OUT_FS = """
uniform sampler2D uState; uniform sampler2D uColor; uniform vec2 texelSize;
uniform float lo, hi, edge, gain, invert;
void main(){
    float v = texture(uState, vUv).y;
    float vr = texture(uState, vR).y, vl = texture(uState, vL).y;
    float vt = texture(uState, vT).y, vb = texture(uState, vB).y;
    float m = smoothstep(lo, hi, v);
    m = mix(m, 1.0 - m, invert);
    float e = clamp(length(vec2(vr - vl, vt - vb)) * 6.0, 0.0, 1.0);
    vec3 c = texture(uColor, vUv).rgb;
    fragColor = vec4(c * mix(m, e, edge) * gain, 1.0);
}
"""


@register
class ReactionDiffusionSim(Simulation):
    name = "Reaction-Diffusion (Gray-Scott)"
    description = ("Turing patterns. Feed/kill presets: spots 0.035/0.065, coral 0.0545/0.062, "
                   "mitosis 0.0367/0.0649, worms 0.078/0.061, maze 0.029/0.057.")
    params = [
        Param("feed", "Feed rate", 0.01, 0.1, 0.0545, group="Chemistry"),
        Param("kill", "Kill rate", 0.04, 0.075, 0.062, group="Chemistry"),
        Param("Du", "Diffusion A", 0.2, 1.4, 1.0, group="Chemistry"),
        Param("Dv", "Diffusion B", 0.1, 0.8, 0.5, group="Chemistry"),
        Param("steps", "Steps per frame", 1, 40, 14, group="Chemistry", tooltip="Growth speed"),
        Param("inject", "Spawner injection", 0, 10, 2, group="Spawners"),
        Param("drift", "Spawner drift", 0, 10, 1, group="Spawners", tooltip="Spawner force pushes the pattern"),
        Param("swirl", "Global swirl", -1, 1, 0, group="Spawners"),
        Param("color_rate", "Colour pickup", 0, 5, 1, group="Colour"),
        Param("color_diffuse", "Colour diffusion", 0, 1, 0.15, group="Colour"),
        Param("lo", "Threshold low", 0, 0.5, 0.12, group="Look"),
        Param("hi", "Threshold high", 0.05, 1, 0.35, group="Look"),
        Param("edge", "Edge mode", 0, 1, 0, group="Look", tooltip="0 = filled, 1 = outlines"),
        Param("invert", "Invert", 0, 1, 0, group="Look"),
        Param("gain", "Brightness", 0, 4, 1.4, group="Look"),
        Param("sim_res", "Grid (height)", 128, 1080, 480, step=16, integer=True, modulatable=False,
              reinit=True, group="Quality", tooltip="Higher = finer patterns"),
    ]

    def __init__(self, gl):
        super().__init__(gl)
        self.p_rd = gl.pass_(RD_FS)
        self.p_col = gl.pass_(COLOR_FS)
        self.p_out = gl.pass_(OUT_FS)
        self.targets = []
        self._acc = 0.0

    def setup(self, out_w, out_h, values, render_mode):
        self._free()
        self.aspect = out_w / out_h
        sh = int(values.get("sim_res", 480))
        sw = max(8, int(round(sh * self.aspect)))
        self.state = Double(self.ctx, sw, sh, 2, "f4")
        self.color = Double(self.ctx, sw, sh, 4, "f2")
        self.out = Target(self.ctx, sw, sh, 4, "f2")
        self.targets = [self.state, self.color, self.out]
        self.width, self.height = out_w, out_h
        self.reset()

    def reset(self):
        w, h = self.state.w, self.state.h
        s = np.zeros((h, w, 2), "f4")
        s[..., 0] = 1.0
        rng = np.random.default_rng(1)
        for _ in range(12):  # a few random seeds so something grows even without spawners
            cx, cy, r = rng.integers(0, w), rng.integers(0, h), rng.integers(3, 10)
            s[max(0, cy - r):cy + r, max(0, cx - r):cx + r, 1] = 0.5
        self.state.read.tex.write(s.tobytes())
        self.state.write.tex.write(s.tobytes())
        self.color.clear(0.4, 0.4, 0.4, 1)
        self._acc = 0.0

    def step(self, dt, t, v, spawners):
        if dt <= 0:
            return
        A, B, C, n = pack_spawners(spawners)
        sp = dict(spA=A, spB=B, spC=C, spCount=n, aspect=self.aspect)
        tx = self.state.texel
        self._acc += v["steps"] * min(dt, 0.1) * 60.0
        k = int(self._acc)
        self._acc -= k
        for _ in range(min(k, 200)):
            self.p_rd.run(self.state.write.fbo, dict(uState=self.state.read.tex), texelSize=tx,
                          feed=v["feed"], kill=v["kill"], Du=v["Du"], Dv=v["Dv"], h=1.0,
                          inject=v["inject"], drift=v["drift"], swirl=v["swirl"], **sp)
            self.state.swap()
        self.p_col.run(self.color.write.fbo, dict(uColor=self.color.read.tex, uState=self.state.read.tex),
                       texelSize=tx, dt=min(dt, 0.1) * 60.0 * v["steps"] / 14.0,
                       colorDiffuse=v["color_diffuse"], colorRate=v["color_rate"], drift=v["drift"],
                       swirl=v["swirl"], **sp)
        self.color.swap()
        self.p_out.run(self.out.fbo, dict(uState=self.state.read.tex, uColor=self.color.read.tex),
                       texelSize=tx, lo=v["lo"], hi=max(v["hi"], v["lo"] + 1e-3), edge=v["edge"],
                       gain=v["gain"], invert=v["invert"])

    def output(self):
        return self.out.tex

    def _free(self):
        for t in self.targets:
            t.release()
        self.targets = []

    def release(self):
        self._free()
        for p in (self.p_rd, self.p_col, self.p_out):
            p.release()

