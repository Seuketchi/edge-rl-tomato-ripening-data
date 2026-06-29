#!/usr/bin/env python3
"""Side-by-side ripening timelapse: chamber camera (left, with telemetry HUD)
vs. table real-color (right), crossfading across the 5-day run. Data from the
perceived telemetry CSV. Outputs an MP4 via ffmpeg.
"""
import os, subprocess
import numpy as np, pandas as pd
from PIL import Image, ImageDraw, ImageFont

H = 560                      # panel height
TOP = 54                     # header band
FPS, DUR_S = 25, 10.0        # ~2 s/day over 5 days
NF = int(FPS * DUR_S)
DAYS = 5.0
FRAMES = "media/frames"
OUT = "media/ripening_timelapse.mp4"
CSV = "data/demo-2026-06-22.csv"  # real recorded Run 2 telemetry (HUD)

os.makedirs(FRAMES, exist_ok=True)

def font(sz, bold=True):
    for p in ([f"/usr/share/fonts/TTF/DejaVuSans{'-Bold' if bold else ''}.ttf",
               f"/usr/share/fonts/dejavu/DejaVuSans{'-Bold' if bold else ''}.ttf"]):
        if os.path.exists(p):
            return ImageFont.truetype(p, sz)
    return ImageFont.load_default()

def fit(img, W, Ht, bg=(245, 245, 245)):
    """Resize to height Ht, center on a (W,Ht) background."""
    w = int(round(img.width * Ht / img.height))
    im = img.resize((w, Ht), Image.LANCZOS)
    canvas = Image.new("RGB", (W, Ht), bg)
    canvas.paste(im, ((W - w) // 2, 0))
    return canvas

# ---- load images ----
cham = [Image.open(f"assets/cal_{c}.jpg").convert("RGB") for c in ("green", "turning", "red")]
tab = [Image.open(f"assets/table_day{i}.png").convert("RGB") for i in range(1, 7)]

CW = int(round(cham[0].width * H / cham[0].height))            # chamber 4:3 -> wide
TW = max(int(round(i.width * H / i.height)) for i in tab)       # table portrait -> narrow
cham = [fit(i, CW, H) for i in cham]
tab = [fit(i, TW, H) for i in tab]

def blend(imgs, pos, t):
    if t <= pos[0]:
        return imgs[0]
    if t >= pos[-1]:
        return imgs[-1]
    for k in range(len(pos) - 1):
        if pos[k] <= t <= pos[k + 1]:
            return Image.blend(imgs[k], imgs[k + 1], (t - pos[k]) / (pos[k + 1] - pos[k]))
    return imgs[-1]

# ---- telemetry ----
df = pd.read_csv(CSV, comment="#")
ed = df["elapsed_days"].to_numpy()

f_title = font(22); f_lbl = font(18); f_hud = font(19); f_hud_s = font(16)
Wc = CW + TW

for fi in range(NF):
    t = DAYS * fi / (NF - 1)
    chi = blend(cham, [0.0, 2.5, 5.0], t)
    tbi = blend(tab, [0, 1, 2, 3, 4, 5], t)

    frame = Image.new("RGB", (Wc, H + TOP), (20, 20, 24))
    frame.paste(chi, (0, TOP)); frame.paste(tbi, (CW, TOP))
    dr = ImageDraw.Draw(frame, "RGBA")

    # header
    dr.text((12, 14), "Edge-RL ripening — chamber camera vs. table (real)   ·   Jun 22–26 2026",
            font=f_title, fill=(240, 240, 245))
    dr.text((12, TOP + 6), "CHAMBER CAMERA  (green-light)", font=f_lbl, fill=(180, 255, 180))
    dr.text((CW + 12, TOP + 6), "TABLE — real color", font=f_lbl, fill=(255, 235, 180))
    dr.line([(CW, TOP), (CW, TOP + H)], fill=(20, 20, 24), width=4)

    # telemetry HUD (bottom-left of chamber panel)
    r = df.iloc[int(np.argmin(np.abs(ed - min(t, ed.max()))))]
    lines = [
        (r["datetime_pht"][:10] + f"   ·   Day {int(min(t,4.99))+1}/5", f_hud, (235, 235, 235)),
        (f"temp {r['temp']:.1f}°C   setpoint {r['setpoint']:.1f}°C", f_hud_s, (235, 235, 235)),
        (f"X = {r['chromatic_x']:.3f}   [{r['class']}]", f_hud, (255, 180, 90)),
        (f"action: {r['action']}", f_hud_s, (235, 235, 235)),
    ]
    bx, by = 10, TOP + H - 132
    dr.rectangle([bx, by, bx + 320, TOP + H - 10], fill=(0, 0, 0, 150))
    yy = by + 8
    for txt, fnt, col in lines:
        dr.text((bx + 12, yy), txt, font=fnt, fill=col)
        yy += fnt.size + 8

    frame.save(f"{FRAMES}/f{fi:04d}.png")

print(f"rendered {NF} frames {Wc}x{H+TOP}")
subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS),
                "-i", f"{FRAMES}/f%04d.png", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2", OUT], check=True)
print("wrote", OUT)
