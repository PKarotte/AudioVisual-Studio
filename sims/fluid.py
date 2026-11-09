"""
Stable-fluids (Jos Stam) GPU solver, in the style of Pavel Dobryakov's WebGL fluid.

Velocity is simulated on a coarse grid (`sim_res`), dye on the output-resolution
grid.  Spawners act as continuous jets: they accelerate the velocity field along
their direction and inject coloured dye.
"""
from __future__ import annotations

import math

from ..gl_common import SIMPLEX3, SPAWNER_UNIFORMS, Double, Target, pack_spawners
from ..params import Param
from .base import Simulation, register

FORCES_FS = SPAWNER_UNIFORMS + SIMPLEX3 + """
uniform sampler2D uVelocity; uniform sampler2D uDye;
uniform float dt, forceScale, buoyancy, noiseForce, noiseScale, noiseTime;
void main(){
    vec2 v = texture(uVelocity, vUv).xy;
    for (int i = 0; i < spCount; i++) {
        float g = spFalloff(i, vUv);
        v += spB[i].xy * spA[i].w * forceScale * g * dt;
    }
    if (buoyancy != 0.0) {
        vec3 d = texture(uDye, vUv).rgb;
        v.y += buoyancy * dot(d, vec3(0.299, 0.587, 0.114)) * forceScale * 0.5 * dt;
    }
    if (noiseForce > 0.0) {
        vec2 p = vec2(vUv.x * aspect, vUv.y) * noiseScale;
        v += curlNoise(p, noiseTime) * noiseForce * forceScale * 0.1 * dt;
    }
    fragColor = vec4(clamp(v, -2000.0, 2000.0), 0.0, 1.0);
}
"""

DYE_SPLAT_FS = SPAWNER_UNIFORMS + """
uniform sampler2D uDye; uniform float dt, rate, dyeMax;
void main(){
    vec3 c = texture(uDye, vUv).rgb;
    for (int i = 0; i < spCount; i++) {
        float g = spFalloff(i, vUv);
        float dyeAmt = spB[i].z;
        if (dyeAmt <= 0.0 || g <= 0.001) continue;

        vec3 target = spC[i].rgb * dyeAmt;
        float alpha = clamp(g * dt * rate * 8.0, 0.0, 1.0);
        c = mix(c, max(c, target), alpha);
    }
    fragColor = vec4(min(c, vec3(dyeMax)), 1.0);
}
"""

CURL_FS = """
uniform sampler2D uVelocity;
void main(){
    float L = texture(uVelocity, vL).y; float R = texture(uVelocity, vR).y;
    float T = texture(uVelocity, vT).x; float B = texture(uVelocity, vB).x;
    fragColor = vec4(0.5 * (R - L - T + B), 0.0, 0.0, 1.0);
}
"""

VORTICITY_FS = """
uniform sampler2D uVelocity; uniform sampler2D uCurl; uniform float curl, dt;
void main(){
    float L = texture(uCurl, vL).x; float R = texture(uCurl, vR).x;
    float T = texture(uCurl, vT).x; float B = texture(uCurl, vB).x;
    float C = texture(uCurl, vUv).x;
    vec2 force = 0.5 * vec2(abs(T) - abs(B), abs(R) - abs(L));
    force /= length(force) + 0.0001;
    force *= curl * C;
    force.y *= -1.0;
    vec2 v = texture(uVelocity, vUv).xy + force * dt;
    fragColor = vec4(clamp(v, -2000.0, 2000.0), 0.0, 1.0);
}
"""

DIVERGENCE_FS = """
uniform sampler2D uVelocity;
void main(){
    float L = texture(uVelocity, vL).x; float R = texture(uVelocity, vR).x;
    float T = texture(uVelocity, vT).y; float B = texture(uVelocity, vB).y;
    vec2 C = texture(uVelocity, vUv).xy;
    if (vL.x < 0.0) L = -C.x;
    if (vR.x > 1.0) R = -C.x;
    if (vT.y > 1.0) T = -C.y;
    if (vB.y < 0.0) B = -C.y;
    fragColor = vec4(0.5 * (R - L + T - B), 0.0, 0.0, 1.0);
}
"""

CLEAR_FS = """
uniform sampler2D uTexture; uniform float value;
void main(){ fragColor = value * texture(uTexture, vUv); }
"""

PRESSURE_FS = """
uniform sampler2D uPressure; uniform sampler2D uDivergence;
void main(){
    float L = texture(uPressure, vL).x; float R = texture(uPressure, vR).x;
    float T = texture(uPressure, vT).x; float B = texture(uPressure, vB).x;
    float d = texture(uDivergence, vUv).x;
    fragColor = vec4((L + R + B + T - d) * 0.25, 0.0, 0.0, 1.0);
}
"""

GRADIENT_FS = """
uniform sampler2D uPressure; uniform sampler2D uVelocity;
void main(){
    float L = texture(uPressure, vL).x; float R = texture(uPressure, vR).x;
    float T = texture(uPressure, vT).x; float B = texture(uPressure, vB).x;
    vec2 v = texture(uVelocity, vUv).xy - vec2(R - L, T - B);
    fragColor = vec4(v, 0.0, 1.0);
}
"""

ADVECT_VEL_FS = """
uniform sampler2D uVelocity; uniform sampler2D uSource;
uniform vec2 velTexel; uniform float dt, dissFactor;
void main(){
    vec2 coord = vUv - dt * texture(uVelocity, vUv).xy * velTexel;
    vec4 r = texture(uSource, coord);
    fragColor = vec4(clamp(r.xy * dissFactor, -2000.0, 2000.0), 0.0, 1.0);
}
"""

ADVECT_DYE_FS = """
uniform sampler2D uVelocity; uniform sampler2D uSource;
uniform vec2 velTexel; uniform float dt, dissFactor;
void main(){
    vec2 coord = vUv - dt * texture(uVelocity, vUv).xy * velTexel;
    vec3 r = texture(uSource, coord).rgb;
    vec3 decayed = r * dissFactor;
    if (dot(decayed, vec3(1.0)) < 0.003) {
        fragColor = vec4(0.0);
    } else {
        fragColor = vec4(decayed, 1.0);
    }
}
"""


@register
class FluidSim(Simulation):
    name = "Fluid (Stable Fluids)"
    description = "Incompressible Navier-Stokes with vorticity confinement, buoyancy and curl-noise turbulence."
    params = [
        Param("speed", "Simulation speed", 0, 3, 1.0, group="Dynamics"),
        Param("vel_diss", "Velocity dissipation", 0, 50, 0.25, group="Dynamics",
              tooltip="How fast motion dies out"),
        Param("dye_diss", "Dye fade time (seconds)", 0.2, 20.0, 1.5, step=0.1, group="Dynamics",
              tooltip="How long colors stay on screen before completely dying out (0.5s = short burst, 5s = long trail)"),
        Param("pressure", "Pressure retention", 0, 1, 0.8, group="Dynamics"),
        Param("curl", "Vorticity (swirl)", 0, 80, 20, group="Dynamics"),
        Param("buoyancy", "Buoyancy", -10, 10, 0, group="Forces", tooltip="Bright dye rises (+) or sinks (-)"),
        Param("noise_force", "Turbulence", 0, 10, 0, group="Forces", tooltip="Curl-noise force field"),
        Param("noise_scale", "Turbulence scale", 0.5, 20, 3, group="Forces"),
        Param("noise_speed", "Turbulence evolution", 0, 2, 0.15, group="Forces"),
        Param("force_mult", "Global force multiplier", 0, 4, 1, group="Forces"),
        Param("dye_rate", "Dye injection rate", 0, 5, 1, group="Dye"),
        Param("dye_max", "Dye saturation cap", 0.5, 10, 3, group="Dye"),
        Param("iterations", "Pressure iterations", 4, 120, 28, step=1, integer=True, modulatable=False,
              group="Quality", tooltip="More iterations = cleaner incompressibility and crisper vortices at high grid resolutions"),
        Param("sim_res", "Velocity grid (height)", 64, 2160, 384, step=32, integer=True, modulatable=False,
              reinit=True, group="Quality", tooltip="Simulation grid resolution (e.g. 512, 1080, 1440). Higher = finer swirls, filament details and crisp vortices. Coarser = broader, smoother waves."),
    ]

    def __init__(self, gl):
        super().__init__(gl)
        P = gl.pass_
        self.p_forces = P(FORCES_FS)
        self.p_dye = P(DYE_SPLAT_FS)
        self.p_curl = P(CURL_FS)
        self.p_vort = P(VORTICITY_FS)
        self.p_div = P(DIVERGENCE_FS)
        self.p_clear = P(CLEAR_FS)
        self.p_press = P(PRESSURE_FS)
        self.p_grad = P(GRADIENT_FS)
        self.p_adv_vel = P(ADVECT_VEL_FS)
        self.p_adv_dye = P(ADVECT_DYE_FS)
        self.targets = []
        self.noise_t = 0.0

    def setup(self, out_w, out_h, values, render_mode):
        self._free()
        aspect = out_w / out_h
        sh = int(values.get("sim_res", 192))
        sw = max(8, int(round(sh * aspect)))
        ctx = self.ctx
        self.vel = Double(ctx, sw, sh, 2, "f2")
        self.press = Double(ctx, sw, sh, 1, "f2")
        self.div = Target(ctx, sw, sh, 1, "f2")
        self.curlt = Target(ctx, sw, sh, 1, "f2")
        self.dye = Double(ctx, out_w, out_h, 4, "f2")
        self.targets = [self.vel, self.press, self.div, self.curlt, self.dye]
        self.aspect = aspect
        self.width, self.height = out_w, out_h

    def reset(self):
        for t in self.targets:
            if isinstance(t, Double):
                t.clear(0, 0, 0, 0)
            else:
                t.fbo.clear(0, 0, 0, 0)

    def step(self, dt, t, v, spawners):
        dt_frame = max(0.0, min(dt, 1.0 / 15.0))
        speed = float(v.get("speed", 1.0))
        dt_sim = max(0.0, min(dt * speed, 1.0 / 15.0))
        if dt_frame <= 0:
            return
        A, B, C, n = pack_spawners(spawners)
        vt = self.vel.texel
        sp = dict(spA=A, spB=B, spC=C, spCount=n, aspect=self.aspect)
        force_mult = float(v.get("force_mult", 1.0))
        force_scale = 60.0 * (self.vel.h / 128.0) * force_mult
        self.noise_t += dt_sim * float(v.get("noise_speed", 0.15))

        self.p_forces.run(self.vel.write.fbo, dict(uVelocity=self.vel.read.tex, uDye=self.dye.read.tex),
                          texelSize=vt, dt=dt_sim, forceScale=force_scale, buoyancy=float(v.get("buoyancy", 0.0)),
                          noiseForce=float(v.get("noise_force", 0.0)), noiseScale=float(v.get("noise_scale", 3.0)),
                          noiseTime=self.noise_t, **sp)
        self.vel.swap()

        curl = float(v.get("curl", 20.0))
        if curl > 0 and dt_sim > 0:
            self.p_curl.run(self.curlt.fbo, dict(uVelocity=self.vel.read.tex), texelSize=vt)
            self.p_vort.run(self.vel.write.fbo, dict(uVelocity=self.vel.read.tex, uCurl=self.curlt.tex),
                            texelSize=vt, curl=curl, dt=dt_sim)
            self.vel.swap()

        self.p_div.run(self.div.fbo, dict(uVelocity=self.vel.read.tex), texelSize=vt)
        self.p_clear.run(self.press.write.fbo, dict(uTexture=self.press.read.tex), texelSize=vt,
                         value=float(v.get("pressure", 0.8)))
        self.press.swap()
        for _ in range(int(v.get("iterations", 25))):
            self.p_press.run(self.press.write.fbo, dict(uPressure=self.press.read.tex, uDivergence=self.div.tex),
                             texelSize=vt)
            self.press.swap()
        self.p_grad.run(self.vel.write.fbo, dict(uPressure=self.press.read.tex, uVelocity=self.vel.read.tex),
                        texelSize=vt)
        self.vel.swap()

        vel_diss = max(0.0, float(v.get("vel_diss", 0.25)))
        vel_diss_factor = 1.0 / (1.0 + vel_diss * dt_sim)
        self.p_adv_vel.run(self.vel.write.fbo, dict(uVelocity=self.vel.read.tex, uSource=self.vel.read.tex),
                           texelSize=vt, velTexel=vt, dt=dt_sim, dissFactor=vel_diss_factor)
        self.vel.swap()

        # In DYE_SPLAT_FS, injection smoothly approaches spawner color so dye never accumulates into overexposed white
        dye_rate = float(v.get("dye_rate", 1.0))
        dye_max = float(v.get("dye_max", 3.0))
        self.p_dye.run(self.dye.write.fbo, dict(uDye=self.dye.read.tex), texelSize=self.dye.texel,
                       dt=dt_frame, rate=dye_rate, dyeMax=dye_max, **sp)
        self.dye.swap()

        # Dye fading: exponential decay strictly governed by fade_time (dye_diss)
        # Factor 3.8 ensures that at t = fade_time, dye drops to < 2.5%, below threshold, and completely dies
        fade_time = max(0.05, float(v.get("dye_diss", 1.5)))
        dye_diss_factor = math.exp(-3.8 * dt_frame / fade_time)
        self.p_adv_dye.run(self.dye.write.fbo, dict(uVelocity=self.vel.read.tex, uSource=self.dye.read.tex),
                           texelSize=self.dye.texel, velTexel=vt, dt=dt_sim, dissFactor=dye_diss_factor)
        self.dye.swap()

    def output(self):
        return self.dye.read.tex

    def _free(self):
        for t in self.targets:
            t.release()
        self.targets = []

    def release(self):
        self._free()
        for p in (self.p_forces, self.p_dye, self.p_curl, self.p_vort, self.p_div, self.p_clear,
                  self.p_press, self.p_grad, self.p_adv_vel, self.p_adv_dye):
            p.release()
