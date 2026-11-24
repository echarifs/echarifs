#!/usr/bin/env python3
"""
Lite prep for ASCII conversion without rembg/cv2 (PIL + numpy only).

The profile photo is a studio portrait on a uniform dark backdrop, so instead
of a neural background remover we:
  1. upscale with Lanczos
  2. estimate the backdrop colour from the border pixels
  3. segment the subject by colour distance (backdrop is far more uniform than
     any remover would assume, so a simple threshold + binary close/open wins)
  4. stretch tones inside the subject: skin near white, hair dark, clothes
     lifted with gamma so they do not flood the ramp
  5. composite on white and crop a square around the head (biased up)

Output: source-prepped.png (grayscale) for make_ascii_svg.py, which wants
bright = blank, dark = dense characters.

    python scripts/prep_photo_lite.py [input.png] [output.png]
"""
import os
import sys

import numpy as np
from PIL import Image, ImageFilter

HERE = os.path.dirname(os.path.abspath(__file__))
INP = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "..", "avatar.png")
OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.join(HERE, "..", "source-prepped.png")

UPSCALE = 6           # 160px avatar -> 960px working resolution
BG_DISTANCE = 20      # colour-distance threshold separating subject from backdrop
GAMMA = 1.1          # mild lift so skin stays sparser than hair/clothes
TONE_CAP = 0.52      # skin lands in visible mid-range ramp chars, not the faintest ones

im = Image.open(INP).convert("RGB").resize((160 * UPSCALE, 160 * UPSCALE), Image.LANCZOS)
rgb = np.asarray(im).astype(np.float32)
h, w, _ = rgb.shape

# ---- backdrop estimate (border ring) --------------------------------------
border = np.concatenate(
    [rgb[:8].reshape(-1, 3), rgb[-8:].reshape(-1, 3), rgb[:, :8].reshape(-1, 3), rgb[:, -8:].reshape(-1, 3)]
)
bg = border.mean(axis=0)

# ---- segment by colour distance, clean the mask ----------------------------
dist = np.linalg.norm(rgb - bg, axis=2)
mask = (dist > BG_DISTANCE).astype(np.uint8) * 255
m = Image.fromarray(mask, "L")
# close (fill eyes/mouth holes), then open (drop backdrop speckle)
close = ImageFilter.MaxFilter(9)
open_ = ImageFilter.MinFilter(9)
mask = np.asarray(m.filter(close).filter(close).filter(open_).filter(open_)) > 128

# ---- tone curve inside the subject ----------------------------------------
# Target zones for the ramp in make_ascii_svg.py (blank at >= 0.80):
#   hair / clothes  ->  0.05-0.25   (dense characters)
#   skin / face     ->  0.45-0.78   (sparse but visible characters)
gray = rgb.mean(axis=2)
lo, hi = np.percentile(gray[mask], [5, 78])
tone = np.clip((gray - lo) / max(hi - lo, 1.0), 0.0, 1.0) ** (1.0 / GAMMA)
tone = np.clip(tone * TONE_CAP, 0.0, TONE_CAP)  # cap below the blank floor

out = np.where(mask, tone, 1.0) * 255.0
prep = Image.fromarray(out.astype(np.uint8), "L")

# ---- square crop around the head (keeps the dark clothes out of frame) -----
ys, xs = np.where(mask)
height = ys.max() - ys.min()
side = int(height * 0.8) + 40
head_cy = int(np.median(ys[ys < ys.min() + height * 0.45]))  # upper-mass centre
cx = int((xs.min() + xs.max()) // 2)
canvas = Image.new("L", (side, side), 255)
x0, y0 = cx - side // 2, max(head_cy - side // 2, 0)
canvas.paste(prep, (-x0, -y0))
canvas.save(OUT)
print(f"wrote {OUT} ({side}x{side}); subject fraction {mask.mean():.2f}")
