"""Post-processing shared by all simulations: shading, bloom, tonemap, grade, feedback trails."""
from __future__ import annotations

import os
import moderngl

from .gl_common import GL, Double, Target
from .params import Param

POST_PARAMS: list[Param] = [
    Param("exposure", "Exposure", 0, 5, 1.1, group="Tone"),
    Param("contrast", "Contrast", 0.2, 2.5, 1.0, group="Tone"),
    Param("saturation", "Saturation", 0, 2.5, 1.0, group="Tone"),
    Param("hue_rotate", "Hue rotate", -180, 180, 0, wrap=True, group="Tone"),
    Param("gamma", "Gamma", 0.4, 2.5, 1.0, group="Tone"),
    Param("shading", "Liquid shading (3D)", 0, 1, 0.35, group="Look"),
    Param("bloom", "Bloom", 0, 3, 0.3, group="Look"),
    Param("bloom_threshold", "Bloom threshold", 0, 2, 0.6, group="Look"),
    Param("vignette", "Vignette", 0, 1.5, 0.3, group="Look"),
    Param("feedback", "Feedback trails", 0, 0.99, 0.0, group="Feedback",
          tooltip="Keeps a fading copy of previous frames (Milkdrop style)"),
    Param("fb_zoom", "Feedback zoom", -0.05, 0.05, 0.004, group="Feedback"),
    Param("fb_rotate", "Feedback rotate (deg)", -5, 5, 0.0, group="Feedback"),
    Param("bg_value", "Background brightness", 0, 0.3, 0.0, group="Look"),
    Param("bg_opacity", "BG Image Opacity", 0.0, 1.0, 1.0, group="Background",
          tooltip="Opacity of loaded background image"),
    Param("bg_blend", "BG Blend Mode", 0, 3, 0, step=1, integer=True, group="Background",
          tooltip="0 = Underlay, 1 = Screen, 2 = Add, 3 = Multiply"),
    Param("bg_fit", "BG Fit Mode", 0, 2, 0, step=1, integer=True, group="Background",
          tooltip="0 = Cover (crop excess), 1 = Fit (letterbox), 2 = Stretch"),
]

HDR_FS = """
uniform sampler2D uSrc; uniform float shading; uniform vec2 texelSize;
void main(){
    vec3 c = texture(uSrc, vUv).rgb;
    if (shading > 0.0) {
        vec3 l = texture(uSrc, vL).rgb, r = texture(uSrc, vR).rgb;
        vec3 t = texture(uSrc, vT).rgb, b = texture(uSrc, vB).rgb;
        float dx = length(r) - length(l), dy = length(t) - length(b);
        float normScale = (texelSize.y > 0.0) ? (1.0 / (720.0 * texelSize.y)) : 1.0;
        dx *= normScale;
        dy *= normScale;
        vec3 n = normalize(vec3(dx, dy, 0.08));
        float diffuse = clamp(dot(n, normalize(vec3(-0.3, 0.4, 1.0))) + 0.35, 0.35, 1.25);
        float spec = pow(clamp(dot(reflect(-normalize(vec3(-0.3, 0.4, 1.0)), n), vec3(0,0,1)), 0.0, 1.0), 24.0);
        c = mix(c, c * diffuse + spec * 0.35 * length(c), shading);
    }
    fragColor = vec4(max(c, 0.0), 1.0);
}
"""

BRIGHT_FS = """
uniform sampler2D uSrc; uniform float threshold, exposure;
void main(){
    vec3 c = texture(uSrc, vUv).rgb * exposure;
    float br = max(c.r, max(c.g, c.b));
    float k = max(br - threshold, 0.0) / max(br, 1e-4);
    fragColor = vec4(c * k, 1.0);
}
"""

BLUR_FS = """
uniform sampler2D uSrc; uniform vec2 dir;
void main(){
    vec3 s = texture(uSrc, vUv).rgb * 0.227027;
    s += texture(uSrc, vUv + dir * 1.3846).rgb * 0.3162162;
    s += texture(uSrc, vUv - dir * 1.3846).rgb * 0.3162162;
    s += texture(uSrc, vUv + dir * 3.2308).rgb * 0.0702703;
    s += texture(uSrc, vUv - dir * 3.2308).rgb * 0.0702703;
    fragColor = vec4(s, 1.0);
}
"""

FINAL_FS = """
uniform sampler2D uHdr; uniform sampler2D uBloom1; uniform sampler2D uBloom2; uniform sampler2D uPrev;
uniform sampler2D uBg; uniform int hasBg;
uniform float bgOpacity; uniform int bgBlend; uniform int bgFit; uniform float bgAspect;
uniform float exposure, contrast, saturation, hueRot, gamma, bloom, vignette, feedback, fbZoom, fbRot, bg, aspect;

vec3 hueShift(vec3 c, float a){
    const vec3 k = vec3(0.57735);
    float ca = cos(a);
    return c * ca + cross(k, c) * sin(a) + k * dot(k, c) * (1.0 - ca);
}
vec3 aces(vec3 x){ return clamp((x*(2.51*x+0.03))/(x*(2.43*x+0.59)+0.14), 0.0, 1.0); }

vec3 sampleBg(vec2 uv) {
    if (hasBg == 0) return vec3(0.0);
    vec2 p = uv - 0.5;
    vec2 bgUv;
    if (bgFit == 0) {
        // Cover (crop excess, fill frame)
        vec2 s = vec2(1.0);
        if (aspect > bgAspect) {
            s = vec2(1.0, bgAspect / aspect);
        } else {
            s = vec2(aspect / bgAspect, 1.0);
        }
        bgUv = p * s + 0.5;
    } else if (bgFit == 1) {
        // Fit (letterboxed)
        vec2 s = vec2(1.0);
        if (aspect > bgAspect) {
            s = vec2(aspect / bgAspect, 1.0);
        } else {
            s = vec2(1.0, bgAspect / aspect);
        }
        bgUv = p * s + 0.5;
        if (bgUv.x < 0.0 || bgUv.x > 1.0 || bgUv.y < 0.0 || bgUv.y > 1.0) {
            return vec3(0.0);
        }
    } else {
        // Stretch
        bgUv = uv;
    }
    return texture(uBg, clamp(bgUv, 0.0, 1.0)).rgb;
}

void main(){
    vec3 c = texture(uHdr, vUv).rgb * exposure;
    c += (texture(uBloom1, vUv).rgb + texture(uBloom2, vUv).rgb) * bloom;
    c = aces(c);
    c = pow(c, vec3(1.0 / gamma));
    c = hueShift(c, radians(hueRot));
    float l = dot(c, vec3(0.299, 0.587, 0.114));
    c = mix(vec3(l), c, saturation);
    c = (c - 0.5) * contrast + 0.5;

    // Background Image Blending
    if (hasBg == 1 && bgOpacity > 0.0) {
        vec3 bgCol = sampleBg(vUv);
        if (bgBlend == 0) {
            // Underlay: liquid floating on top of background
            float simDensity = clamp(dot(c, vec3(0.299, 0.587, 0.114)) * 1.5, 0.0, 1.0);
            c = mix(bgCol, c, simDensity * bgOpacity) + c * (0.25 * bgOpacity);
        } else if (bgBlend == 1) {
            // Screen
            vec3 screened = 1.0 - (1.0 - c) * (1.0 - bgCol);
            c = mix(c, screened, bgOpacity);
        } else if (bgBlend == 2) {
            // Add
            c = c + bgCol * bgOpacity;
        } else if (bgBlend == 3) {
            // Multiply
            c = c * mix(vec3(1.0), bgCol, bgOpacity);
        }
    }

    c = max(c, vec3(bg));
    vec2 p = vUv - 0.5; p.x *= aspect;
    c *= clamp(1.0 - vignette * dot(p, p) * 1.2, 0.0, 1.0);
    if (feedback > 0.0) {
        float a = radians(fbRot);
        vec2 q = mat2(cos(a), -sin(a), sin(a), cos(a)) * p / (1.0 + fbZoom);
        q.x /= aspect;
        vec3 prev = texture(uPrev, q + 0.5).rgb;
        c = max(c, prev * feedback);
    }
    fragColor = vec4(clamp(c, 0.0, 1.0), 1.0);
}
"""


class PostFX:
    def __init__(self, gl: GL):
        self.gl = gl
        self.p_hdr = gl.pass_(HDR_FS)
        self.p_bright = gl.pass_(BRIGHT_FS)
        self.p_blur = gl.pass_(BLUR_FS)
        self.p_final = gl.pass_(FINAL_FS)
        self.size = None
        self.targets = []
        self.bg_tex: moderngl.Texture | None = None
        self.bg_path: str = ""
        self.bg_aspect: float = 1.0
        self.dummy_tex = gl.ctx.texture((1, 1), 4, dtype="f1", data=b"\x00\x00\x00\xff")

    def set_background(self, path: str):
        if self.bg_path == path and self.bg_tex is not None:
            return
        if self.bg_tex:
            self.bg_tex.release()
            self.bg_tex = None
        self.bg_path = path or ""
        if not path or not os.path.isfile(path):
            self.bg_aspect = 1.0
            return
        try:
            from PIL import Image, ImageOps
            with Image.open(path) as img:
                img = img.convert("RGBA")
                img = ImageOps.flip(img)
                w, h = img.size
                self.bg_aspect = w / h if h > 0 else 1.0
                data = img.tobytes()
                self.bg_tex = self.gl.ctx.texture((w, h), 4, data=data)
                self.bg_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
                self.bg_tex.repeat_x = False
                self.bg_tex.repeat_y = False
        except Exception as e:
            print(f"Error loading background image {path}: {e}")
            self.bg_tex = None
            self.bg_aspect = 1.0

    def ensure(self, w, h):
        if self.size == (w, h):
            return
        for t in self.targets:
            t.release()
        ctx = self.gl.ctx
        self.hdr = Target(ctx, w, h, 4, "f2")
        self.b1 = Double(ctx, max(2, w // 4), max(2, h // 4), 4, "f2")
        self.b2 = Double(ctx, max(2, w // 16), max(2, h // 16), 4, "f2")
        self.frame = Double(ctx, w, h, 4, "f1")
        self.targets = [self.hdr, self.b1, self.b2, self.frame]
        self.size = (w, h)

    def clear(self):
        if self.size:
            self.frame.clear(0, 0, 0, 1)

    def _blur(self, d: Double, passes=2):
        tx, ty = d.texel
        for _ in range(passes):
            self.p_blur.run(d.write.fbo, dict(uSrc=d.read.tex), dir=(tx, 0.0))
            d.swap()
            self.p_blur.run(d.write.fbo, dict(uSrc=d.read.tex), dir=(0.0, ty))
            d.swap()

    def process(self, src_tex, v: dict, dt: float = 1 / 60):
        w, h = self.size
        sw, sh = src_tex.size
        self.p_hdr.run(self.hdr.fbo, dict(uSrc=src_tex), texelSize=(1.0 / sw, 1.0 / sh),
                       shading=v["shading"])
        if v["bloom"] > 0:
            self.p_bright.run(self.b1.read.fbo, dict(uSrc=self.hdr.tex),
                              threshold=v["bloom_threshold"], exposure=v["exposure"])
            self._blur(self.b1)
            self.p_blur.run(self.b2.read.fbo, dict(uSrc=self.b1.read.tex), dir=(0.0, 0.0))
            self._blur(self.b2)
        # feedback is defined per 60 fps frame; scale it so look is fps-independent
        fb = v["feedback"] ** (dt * 60.0) if v["feedback"] > 0 else 0.0

        bg_tex = self.bg_tex if self.bg_tex is not None else self.dummy_tex
        has_bg = 1 if self.bg_tex is not None else 0
        bg_opacity = float(v.get("bg_opacity", 1.0))
        bg_blend = int(round(v.get("bg_blend", 0)))
        bg_fit = int(round(v.get("bg_fit", 0)))

        self.p_final.run(self.frame.write.fbo,
                         dict(uHdr=self.hdr.tex, uBloom1=self.b1.read.tex, uBloom2=self.b2.read.tex,
                              uPrev=self.frame.read.tex, uBg=bg_tex),
                         exposure=v["exposure"], contrast=v["contrast"], saturation=v["saturation"],
                         hueRot=v["hue_rotate"], gamma=v["gamma"], bloom=v["bloom"] if v["bloom"] > 0 else 0.0,
                         vignette=v["vignette"], feedback=fb, fbZoom=v["fb_zoom"] * dt * 60.0,
                         fbRot=v["fb_rotate"] * dt * 60.0, bg=v["bg_value"], aspect=w / h,
                         hasBg=has_bg, bgOpacity=bg_opacity, bgBlend=bg_blend, bgFit=bg_fit,
                         bgAspect=self.bg_aspect)
        self.frame.swap()
        return self.frame.read

    def release(self):
        for t in self.targets:
            t.release()
        for p in (self.p_hdr, self.p_bright, self.p_blur, self.p_final):
            p.release()
        if self.bg_tex:
            self.bg_tex.release()
            self.bg_tex = None
        if self.dummy_tex:
            self.dummy_tex.release()
            self.dummy_tex = None

