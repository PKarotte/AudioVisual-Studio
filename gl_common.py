"""Small helpers around moderngl: full-screen passes and ping-pong framebuffers."""
from __future__ import annotations

import numpy as np
import moderngl

QUAD_VS = """
#version 330
in vec2 in_pos;
out vec2 vUv;
out vec2 vL; out vec2 vR; out vec2 vT; out vec2 vB;
uniform vec2 texelSize;
void main() {
    vUv = in_pos * 0.5 + 0.5;
    vL = vUv - vec2(texelSize.x, 0.0);
    vR = vUv + vec2(texelSize.x, 0.0);
    vT = vUv + vec2(0.0, texelSize.y);
    vB = vUv - vec2(0.0, texelSize.y);
    gl_Position = vec4(in_pos, 0.0, 1.0);
}
"""

FS_HEADER = """
#version 330
in vec2 vUv;
in vec2 vL; in vec2 vR; in vec2 vT; in vec2 vB;
out vec4 fragColor;
"""

# Ashima / Stefan Gustavson 3D simplex noise (MIT)
SIMPLEX3 = """
vec3 mod289(vec3 x){return x-floor(x*(1.0/289.0))*289.0;}
vec4 mod289(vec4 x){return x-floor(x*(1.0/289.0))*289.0;}
vec4 permute(vec4 x){return mod289(((x*34.0)+1.0)*x);}
vec4 taylorInvSqrt(vec4 r){return 1.79284291400159-0.85373472095314*r;}
float snoise(vec3 v){
  const vec2 C=vec2(1.0/6.0,1.0/3.0); const vec4 D=vec4(0.0,0.5,1.0,2.0);
  vec3 i=floor(v+dot(v,C.yyy)); vec3 x0=v-i+dot(i,C.xxx);
  vec3 g=step(x0.yzx,x0.xyz); vec3 l=1.0-g; vec3 i1=min(g.xyz,l.zxy); vec3 i2=max(g.xyz,l.zxy);
  vec3 x1=x0-i1+C.xxx; vec3 x2=x0-i2+C.yyy; vec3 x3=x0-D.yyy;
  i=mod289(i);
  vec4 p=permute(permute(permute(i.z+vec4(0.0,i1.z,i2.z,1.0))+i.y+vec4(0.0,i1.y,i2.y,1.0))+i.x+vec4(0.0,i1.x,i2.x,1.0));
  float n_=0.142857142857; vec3 ns=n_*D.wyz-D.xzx;
  vec4 j=p-49.0*floor(p*ns.z*ns.z); vec4 x_=floor(j*ns.z); vec4 y_=floor(j-7.0*x_);
  vec4 x=x_*ns.x+ns.yyyy; vec4 y=y_*ns.x+ns.yyyy; vec4 h=1.0-abs(x)-abs(y);
  vec4 b0=vec4(x.xy,y.xy); vec4 b1=vec4(x.zw,y.zw);
  vec4 s0=floor(b0)*2.0+1.0; vec4 s1=floor(b1)*2.0+1.0; vec4 sh=-step(h,vec4(0.0));
  vec4 a0=b0.xzyw+s0.xzyw*sh.xxyy; vec4 a1=b1.xzyw+s1.xzyw*sh.zzww;
  vec3 p0=vec3(a0.xy,h.x); vec3 p1=vec3(a0.zw,h.y); vec3 p2=vec3(a1.xy,h.z); vec3 p3=vec3(a1.zw,h.w);
  vec4 norm=taylorInvSqrt(vec4(dot(p0,p0),dot(p1,p1),dot(p2,p2),dot(p3,p3)));
  p0*=norm.x; p1*=norm.y; p2*=norm.z; p3*=norm.w;
  vec4 m=max(0.6-vec4(dot(x0,x0),dot(x1,x1),dot(x2,x2),dot(x3,x3)),0.0); m=m*m;
  return 42.0*dot(m*m,vec4(dot(p0,x0),dot(p1,x1),dot(p2,x2),dot(p3,x3)));
}
vec2 curlNoise(vec2 p, float t){
  float e=0.01;
  float n1=snoise(vec3(p.x, p.y+e, t)); float n2=snoise(vec3(p.x, p.y-e, t));
  float n3=snoise(vec3(p.x+e, p.y, t)); float n4=snoise(vec3(p.x-e, p.y, t));
  return vec2((n1-n2), -(n3-n4))/(2.0*e);
}
"""

MAX_SPAWNERS = 64

# Uniform block shared by simulations for spawner data
SPAWNER_UNIFORMS = f"""
#define MAX_SP {MAX_SPAWNERS}
uniform int spCount;
uniform vec4 spA[MAX_SP];   // x, y (0..1, GL y-up), radius, force
uniform vec4 spB[MAX_SP];   // dirx, diry, dye, unused
uniform vec4 spC[MAX_SP];   // r, g, b, unused
uniform float aspect;
float spFalloff(int i, vec2 uv){{
    vec2 d = uv - spA[i].xy; d.x *= aspect;
    float r = spA[i].z;
    return exp(-dot(d,d)/(r*r));
}}
"""


class Pass:
    def __init__(self, gl: "GL", fs: str, vs: str = QUAD_VS):
        self.prog = gl.ctx.program(vertex_shader=vs, fragment_shader=fs)
        self.vao = gl.ctx.vertex_array(self.prog, [(gl.quad, "2f", "in_pos")])

    def set(self, name, value):
        u = self.prog.get(name, None)
        if u is None:
            return
        if isinstance(value, (bytes, bytearray)):
            u.write(value)
        else:
            u.value = value

    def run(self, fbo: moderngl.Framebuffer, textures: dict | None = None, **uniforms):
        fbo.use()
        unit = 0
        for name, tex in (textures or {}).items():
            if self.prog.get(name, None) is None:
                continue
            tex.use(location=unit)
            self.prog[name].value = unit
            unit += 1
        for k, v in uniforms.items():
            self.set(k, v)
        self.vao.render(moderngl.TRIANGLE_STRIP)

    def release(self):
        self.vao.release()
        self.prog.release()


class Target:
    """Texture + framebuffer."""

    def __init__(self, ctx, w, h, comps=4, dtype="f2", linear=True):
        self.w, self.h = int(w), int(h)
        self.tex = ctx.texture((self.w, self.h), comps, dtype=dtype)
        f = moderngl.LINEAR if linear else moderngl.NEAREST
        self.tex.filter = (f, f)
        self.tex.repeat_x = False
        self.tex.repeat_y = False
        self.fbo = ctx.framebuffer(color_attachments=[self.tex])
        self.fbo.clear(0, 0, 0, 0)

    @property
    def texel(self):
        return (1.0 / self.w, 1.0 / self.h)

    def release(self):
        self.fbo.release()
        self.tex.release()


class Double:
    def __init__(self, ctx, w, h, comps=4, dtype="f2", linear=True):
        self.read = Target(ctx, w, h, comps, dtype, linear)
        self.write = Target(ctx, w, h, comps, dtype, linear)

    @property
    def texel(self):
        return self.read.texel

    @property
    def w(self):
        return self.read.w

    @property
    def h(self):
        return self.read.h

    def swap(self):
        self.read, self.write = self.write, self.read

    def clear(self, *c):
        self.read.fbo.clear(*c)
        self.write.fbo.clear(*c)

    def release(self):
        self.read.release()
        self.write.release()


class GL:
    def __init__(self, ctx: moderngl.Context):
        self.ctx = ctx
        self.quad = ctx.buffer(np.array([-1, -1, 1, -1, -1, 1, 1, 1], dtype="f4").tobytes())

    def pass_(self, fs_body: str) -> Pass:
        return Pass(self, FS_HEADER + fs_body)


def pack_spawners(frames) -> tuple[bytes, bytes, bytes, int]:
    n = min(len(frames), MAX_SPAWNERS)
    A = np.zeros((MAX_SPAWNERS, 4), "f4")
    B = np.zeros((MAX_SPAWNERS, 4), "f4")
    C = np.zeros((MAX_SPAWNERS, 4), "f4")
    for i, f in enumerate(frames[:n]):
        A[i] = (f.x, f.y, max(f.radius, 1e-4), f.force)
        B[i] = (f.dx, f.dy, f.dye, getattr(f, "dye_life", 0.0))
        C[i] = (*f.color, 1.0)
    return A.tobytes(), B.tobytes(), C.tobytes(), n

