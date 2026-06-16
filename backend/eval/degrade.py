"""Parameterized degradation pipeline.

Each degradation takes (image, rng, severity) and returns (image, params).
The sample guide's variation list is the spec: phone-photo skew, blur,
shadows, glare, noise, low resolution, crumple, rubber stamps over text,
signatures. Every applied degradation and its parameters are recorded in
degradations.json so eval results can be broken down per degradation.
"""

import math
import random
from typing import Any

from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont


def _deg_rotate90(img: Image.Image, rng: random.Random, severity: float):
    angle = rng.choice([90, 180, 270])
    return img.rotate(angle, expand=True), {"angle": angle}


def _deg_skew(img: Image.Image, rng: random.Random, severity: float):
    angle = rng.uniform(2.0, 3.5) * (1 + severity) * rng.choice([-1, 1])
    out = img.rotate(angle, expand=True, fillcolor=(236, 233, 228),
                     resample=Image.BICUBIC)
    return out, {"angle_deg": round(angle, 2)}


def _deg_blur(img: Image.Image, rng: random.Random, severity: float):
    radius = 1.4 + 1.2 * severity
    return img.filter(ImageFilter.GaussianBlur(radius)), {"radius": round(radius, 2)}


def _deg_shadow(img: Image.Image, rng: random.Random, severity: float):
    out = img.convert("RGB")
    grad = Image.linear_gradient("L").resize(out.size)
    if rng.random() < 0.5:
        grad = grad.transpose(Image.FLIP_TOP_BOTTOM)
    if rng.random() < 0.5:
        grad = grad.rotate(90, expand=False)
    strength = 0.35 + 0.25 * severity
    dark = ImageEnhance.Brightness(out).enhance(1 - strength)
    out = Image.composite(dark, out, grad)
    return out, {"strength": round(strength, 2)}


def _deg_glare(img: Image.Image, rng: random.Random, severity: float):
    out = img.convert("RGB")
    glare = Image.new("L", out.size, 0)
    cx = rng.randint(out.width // 4, 3 * out.width // 4)
    cy = rng.randint(out.height // 5, out.height // 2)
    r = int(min(out.size) * (0.28 + 0.15 * severity))
    radial = Image.radial_gradient("L").resize((2 * r, 2 * r))
    spot = radial.point(lambda p: max(0, 230 - p))
    glare.paste(spot, (cx - r, cy - r))
    white = Image.new("RGB", out.size, "white")
    out = Image.composite(white, out, glare)
    return out, {"center": [cx, cy], "radius": r}


def _deg_noise(img: Image.Image, rng: random.Random, severity: float):
    sigma = 28 + 30 * severity
    noise = Image.effect_noise(img.size, sigma).convert("L")
    out = Image.blend(img.convert("RGB"), Image.merge("RGB", (noise,) * 3),
                      alpha=0.18 + 0.10 * severity)
    return out, {"sigma": round(sigma, 1)}


def _deg_lowres(img: Image.Image, rng: random.Random, severity: float):
    factor = 0.45 - 0.12 * severity
    small = img.resize((max(1, int(img.width * factor)),
                        max(1, int(img.height * factor))), Image.BILINEAR)
    return small.resize(img.size, Image.BILINEAR), {"scale": round(factor, 2)}


def _deg_crumple(img: Image.Image, rng: random.Random, severity: float):
    """Mesh warp: jitter a grid of quads to simulate paper deformation."""
    out = img.convert("RGB")
    w, h = out.size
    grid, jitter = 4, int(5 + 6 * severity)
    mesh = []
    for gy in range(grid):
        for gx in range(grid):
            x0, y0 = gx * w // grid, gy * h // grid
            x1, y1 = (gx + 1) * w // grid, (gy + 1) * h // grid
            j = lambda: rng.randint(-jitter, jitter)
            quad = (x0 + j(), y0 + j(), x0 + j(), y1 + j(),
                    x1 + j(), y1 + j(), x1 + j(), y0 + j())
            mesh.append(((x0, y0, x1, y1), quad))
    out = out.transform(out.size, Image.MESH, mesh, Image.BILINEAR,
                        fillcolor=(236, 233, 228))
    return out, {"jitter_px": jitter}


def _deg_stamp(img: Image.Image, rng: random.Random, severity: float):
    """Rubber stamp over content: the sample guide's 'stamp over text' case."""
    out = img.convert("RGB")
    layer = Image.new("RGBA", out.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    color = rng.choice([(46, 49, 146, 150), (140, 30, 40, 150)])
    cx = rng.randint(out.width // 3, 2 * out.width // 3)
    cy = rng.randint(out.height // 3, 2 * out.height // 3)
    rx, ry = 150, 64
    for k in range(3):
        d.ellipse([cx - rx + k, cy - ry + k, cx + rx - k, cy + ry - k],
                  outline=color, width=2)
    try:
        font = ImageFont.truetype("C:/Windows/Fonts/arialbd.ttf", 26)
    except OSError:
        font = ImageFont.load_default()
    text = rng.choice(["RECEIVED", "PAID", "ORIGINAL", "DUPLICATE"])
    d.text((cx, cy - 14), text, font=font, fill=color, anchor="mm")
    d.text((cx, cy + 16), "01 NOV 2024", font=font, fill=color, anchor="mm")
    layer = layer.rotate(rng.uniform(-18, 18), center=(cx, cy),
                         resample=Image.BICUBIC)
    out.paste(layer, (0, 0), layer)
    return out, {"text": text, "center": [cx, cy]}


def _deg_signature(img: Image.Image, rng: random.Random, severity: float):
    out = img.convert("RGB")
    d = ImageDraw.Draw(out)
    x0 = rng.randint(out.width // 2, out.width - 260)
    y0 = rng.randint(out.height - 240, out.height - 140)
    points = []
    x = x0
    for i in range(28):
        x += rng.randint(4, 10)
        y = y0 + int(14 * math.sin(i * 0.9) + rng.randint(-5, 5))
        points.append((x, y))
    d.line(points, fill=(25, 30, 90), width=2)
    return out, {"at": [x0, y0]}


DEGRADATIONS = {
    "rotate90": _deg_rotate90,
    "skew": _deg_skew,
    "blur": _deg_blur,
    "shadow": _deg_shadow,
    "glare": _deg_glare,
    "noise": _deg_noise,
    "lowres": _deg_lowres,
    "crumple": _deg_crumple,
    "stamp": _deg_stamp,
    "signature": _deg_signature,
}


def apply_degradations(img: Image.Image, names: list[str], rng: random.Random,
                       severity: float = 0.0) -> tuple[Image.Image, list[dict[str, Any]]]:
    applied = []
    for name in names:
        img, params = DEGRADATIONS[name](img, rng, severity)
        applied.append({"name": name, "severity": round(severity, 2),
                        "params": params})
    return img, applied
