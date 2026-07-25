# SPDX-License-Identifier: Apache-2.0
# Part of rsymbolic2, Copyright 2026 Toshihiro Iguchi.
"""Regenerate the web GUI's icon set (web/app/).

A development tool, not part of any build or test: the outputs are committed, so this
only runs when the icon artwork changes.

    pip install pillow
    python scripts/generate_favicon.py

Output paths are resolved from this file's location, so the working directory does not
matter. Writes favicon.svg, favicon.ico (multi-resolution), favicon-16x16.png,
favicon-32x32.png and apple-touch-icon.png, plus the 1024px master icon-1024.png — a
regenerable intermediate that .gitignore keeps out of the deployed site.
"""

import math
import struct
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter

# Repository root -> web/app; independent of the working directory.
OUT_DIR = Path(__file__).resolve().parents[1] / "web" / "app"

# Sizes packed into favicon.ico. See save_ico() for why the base image must be the largest.
ICO_SIZES = [16, 32, 48, 64, 128, 256]
# The icon badge is a white squircle (see generate_master_image), which is also the
# background the opaque apple-touch-icon is flattened onto.
BADGE_BG = (255, 255, 255)

def catmull_rom_spline(pts, num_samples=300):
    if len(pts) < 2:
        return pts
    p = [pts[0]] + pts + [pts[-1]]
    curve = []
    for i in range(1, len(p) - 2):
        p0, p1, p2, p3 = p[i-1], p[i], p[i+1], p[i+2]
        for t_i in range(num_samples):
            t = t_i / float(num_samples)
            t2 = t * t
            t3 = t2 * t
            x = 0.5 * ((2*p1[0]) + (-p0[0] + p2[0])*t + (2*p0[0] - 5*p1[0] + 4*p2[0] - p3[0])*t2 + (-p0[0] + 3*p1[0] - 3*p2[0] + p3[0])*t3)
            y = 0.5 * ((2*p1[1]) + (-p0[1] + p2[1])*t + (2*p0[1] - 5*p1[1] + 4*p2[1] - p3[1])*t2 + (-p0[1] + 3*p1[1] - 3*p2[1] + p3[1])*t3)
            curve.append((x, y))
    curve.append(pts[-1])
    return curve

def draw_smooth_ribbon(draw, points, color, radius):
    spline = catmull_rom_spline(points, num_samples=300)
    for i in range(len(spline) - 1):
        x0, y0 = spline[i]
        x1, y1 = spline[i+1]
        dist = math.hypot(x1 - x0, y1 - y0)
        steps = max(1, int(dist * 2))
        for s_idx in range(steps):
            t = s_idx / steps
            cx = x0 + (x1 - x0) * t
            cy = y0 + (y1 - y0) * t
            draw.ellipse([cx - radius, cy - radius, cx + radius, cy + radius], fill=color)

def generate_master_image(size=1024):
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    s = size / 1024.0

    # Palette
    bg_border_col  = (215, 225, 235, 255) # #D7E1EB
    grid_outer_col = (135, 155, 175, 255) # #879BAF
    grid_inner_col = (195, 210, 225, 255) # #C3D2E1
    pin_col        = (145, 165, 185, 255)
    pin_base_shadow= (175, 192, 208, 200)
    
    blue_ribbon = (24, 119, 242, 255)   # #1877F2
    blue_node   = (15, 105, 225, 255)
    red_ribbon  = (238, 43, 67, 255)   # #EE2B43
    red_node    = (225, 30, 55, 255)

    # 1. Outer Squircle Icon Badge
    margin = 36 * s
    rect = [margin, margin, size - margin, size - margin]
    corner_radius = 210 * s

    # Soft Drop Shadow
    shadow_img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    shadow_draw = ImageDraw.Draw(shadow_img)
    shadow_draw.rounded_rectangle(
        [margin + 4*s, margin + 14*s, size - margin - 4*s, size - margin + 6*s],
        radius=corner_radius,
        fill=(0, 0, 0, 32)
    )
    shadow_img = shadow_img.filter(ImageFilter.GaussianBlur(radius=18 * s))
    img.alpha_composite(shadow_img)

    # White Icon Badge Fill
    draw.rounded_rectangle(rect, radius=corner_radius, fill=(255, 255, 255, 255))
    draw.rounded_rectangle(rect, radius=corner_radius, outline=bg_border_col, width=int(6 * s))

    # 2. Isometric 3D Grid Plane
    left_pt  = (90 * s, 545 * s)
    top_pt   = (512 * s, 260 * s)
    right_pt = (934 * s, 545 * s)
    bot_pt   = (512 * s, 830 * s)

    def lerp(p1, p2, t):
        return (p1[0] + (p2[0] - p1[0]) * t, p1[1] + (p2[1] - p1[1]) * t)

    # Fill grid interior
    draw.polygon([left_pt, top_pt, right_pt, bot_pt], fill=(245, 248, 252, 220))

    # Inner mesh (4x4 cells)
    N = 4
    for i in range(1, N):
        t = i / N
        s1 = lerp(left_pt, bot_pt, t)
        e1 = lerp(top_pt, right_pt, t)
        draw.line([s1, e1], fill=grid_inner_col, width=int(6 * s))

        s2 = lerp(left_pt, top_pt, t)
        e2 = lerp(bot_pt, right_pt, t)
        draw.line([s2, e2], fill=grid_inner_col, width=int(6 * s))

    # Seamless curved joint grid border frame
    draw.line([left_pt, top_pt, right_pt, bot_pt, left_pt], fill=grid_outer_col, width=int(14 * s), joint="curve")

    # 3. Pin Pillars (px, py, gx, gy, col)
    pins = [
        # Blue pins
        (205 * s, 410 * s, 205 * s, 595 * s, blue_node),  # Leftmost blue
        (345 * s, 615 * s, 345 * s, 675 * s, blue_node),  # Bottom trough blue
        (535 * s, 395 * s, 535 * s, 560 * s, blue_node),  # Center rising blue
        (725 * s, 195 * s, 725 * s, 435 * s, blue_node),  # High peak blue
        (795 * s, 410 * s, 795 * s, 515 * s, blue_node),  # Right drop blue

        # Red pins
        (315 * s, 500 * s, 315 * s, 565 * s, red_node),   # Left mid red
        (355 * s, 270 * s, 355 * s, 510 * s, red_node),   # High peak red
        (475 * s, 475 * s, 475 * s, 590 * s, red_node),   # Center slope red
        (665 * s, 590 * s, 665 * s, 640 * s, red_node),   # Lower right red
    ]

    # Draw pin base ovals & vertical stems
    for px, py, gx, gy, col in pins:
        rx, ry = 18 * s, 9 * s
        draw.ellipse([gx - rx, gy - ry, gx + rx, gy + ry], fill=pin_base_shadow, outline=pin_col, width=int(3 * s))
        draw.line([(gx, gy), (px, py)], fill=pin_col, width=int(7 * s))

    # 4. Ribbon curves
    blue_ctrl = [
        (190 * s, 375 * s),
        (205 * s, 410 * s),
        (345 * s, 615 * s),
        (535 * s, 395 * s),
        (725 * s, 195 * s),
        (795 * s, 410 * s),
        (815 * s, 520 * s)
    ]

    red_ctrl = [
        (295 * s, 540 * s),
        (315 * s, 500 * s),
        (355 * s, 270 * s),
        (475 * s, 475 * s),
        (665 * s, 590 * s),
        (775 * s, 690 * s)
    ]

    ribbon_radius = 18 * s

    draw_smooth_ribbon(draw, blue_ctrl, blue_ribbon, ribbon_radius)
    draw_smooth_ribbon(draw, red_ctrl, red_ribbon, ribbon_radius)

    # 5. Node Dots
    node_radius = 28 * s
    for px, py, gx, gy, col in pins:
        draw.ellipse(
            [px - node_radius, py - node_radius, px + node_radius, py + node_radius],
            fill=col,
            outline=(255, 255, 255, 255),
            width=int(5 * s)
        )

    return img

def generate_svg():
    svg_content = '''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 512 512" width="100%" height="100%">
  <defs>
    <filter id="shadow" x="-10%" y="-10%" width="120%" height="120%">
      <feDropShadow dx="0" dy="4" stdDeviation="6" flood-color="#000000" flood-opacity="0.15"/>
    </filter>
  </defs>

  <!-- Background Squircle -->
  <rect x="18" y="18" width="476" height="476" rx="105" ry="105" fill="#FFFFFF" stroke="#D0DCED" stroke-width="3" filter="url(#shadow)"/>

  <!-- 3D Grid Plane -->
  <polygon points="45,272.5 256,130 467,272.5 256,415" fill="#F5F8FC" fill-opacity="0.85" stroke="#879BF7" stroke-width="6" stroke-linejoin="round"/>
  
  <!-- Grid Lines -->
  <line x1="97.75" y1="308.125" x2="308.75" y2="165.625" stroke="#CBD5E1" stroke-width="3"/>
  <line x1="150.5" y1="343.75" x2="361.5" y2="201.25" stroke="#CBD5E1" stroke-width="3"/>
  <line x1="203.25" y1="379.375" x2="414.25" y2="236.875" stroke="#CBD5E1" stroke-width="3"/>

  <line x1="97.75" y1="236.875" x2="308.75" y2="379.375" stroke="#CBD5E1" stroke-width="3"/>
  <line x1="150.5" y1="201.25" x2="361.5" y2="343.75" stroke="#CBD5E1" stroke-width="3"/>
  <line x1="203.25" y1="165.625" x2="414.25" y2="308.125" stroke="#CBD5E1" stroke-width="3"/>

  <!-- Pins Base & Shafts -->
  <!-- Blue Pins -->
  <ellipse cx="102.5" cy="297.5" rx="9" ry="4.5" fill="#B4C3D2" stroke="#91A5B9" stroke-width="1.5"/>
  <line x1="102.5" y1="297.5" x2="102.5" y2="205" stroke="#91A5B9" stroke-width="3.5"/>

  <ellipse cx="172.5" cy="337.5" rx="9" ry="4.5" fill="#B4C3D2" stroke="#91A5B9" stroke-width="1.5"/>
  <line x1="172.5" y1="337.5" x2="172.5" y2="307.5" stroke="#91A5B9" stroke-width="3.5"/>

  <ellipse cx="267.5" cy="280" rx="9" ry="4.5" fill="#B4C3D2" stroke="#91A5B9" stroke-width="1.5"/>
  <line x1="267.5" y1="280" x2="267.5" y2="197.5" stroke="#91A5B9" stroke-width="3.5"/>

  <ellipse cx="362.5" cy="217.5" rx="9" ry="4.5" fill="#B4C3D2" stroke="#91A5B9" stroke-width="1.5"/>
  <line x1="362.5" y1="217.5" x2="362.5" y2="97.5" stroke="#91A5B9" stroke-width="3.5"/>

  <ellipse cx="397.5" cy="257.5" rx="9" ry="4.5" fill="#B4C3D2" stroke="#91A5B9" stroke-width="1.5"/>
  <line x1="397.5" y1="257.5" x2="397.5" y2="205" stroke="#91A5B9" stroke-width="3.5"/>

  <!-- Red Pins -->
  <ellipse cx="157.5" cy="282.5" rx="9" ry="4.5" fill="#B4C3D2" stroke="#91A5B9" stroke-width="1.5"/>
  <line x1="157.5" y1="282.5" x2="157.5" y2="250" stroke="#91A5B9" stroke-width="3.5"/>

  <ellipse cx="177.5" cy="255" rx="9" ry="4.5" fill="#B4C3D2" stroke="#91A5B9" stroke-width="1.5"/>
  <line x1="177.5" y1="255" x2="177.5" y2="135" stroke="#91A5B9" stroke-width="3.5"/>

  <ellipse cx="237.5" cy="295" rx="9" ry="4.5" fill="#B4C3D2" stroke="#91A5B9" stroke-width="1.5"/>
  <line x1="237.5" y1="295" x2="237.5" y2="237.5" stroke="#91A5B9" stroke-width="3.5"/>

  <ellipse cx="332.5" cy="320" rx="9" ry="4.5" fill="#B4C3D2" stroke="#91A5B9" stroke-width="1.5"/>
  <line x1="332.5" y1="320" x2="332.5" y2="295" stroke="#91A5B9" stroke-width="3.5"/>

  <!-- Smooth Blue Ribbon Path -->
  <path d="M 92.5 185 C 102.5 205 132.5 307.5 172.5 307.5 C 220 307.5 235 230 267.5 197.5 C 310 155 330 97.5 362.5 97.5 C 385 97.5 397.5 205 412.5 265" fill="none" stroke="#1877F2" stroke-width="18" stroke-linecap="round" stroke-linejoin="round"/>

  <!-- Smooth Red Ribbon Path -->
  <path d="M 147.5 270 C 157.5 250 162.5 135 177.5 135 C 200 135 215 210 237.5 237.5 C 275 285 315 295 332.5 295 C 360 295 380 340 392.5 350" fill="none" stroke="#EE2B43" stroke-width="18" stroke-linecap="round" stroke-linejoin="round"/>

  <!-- Node Dots (Blue) -->
  <circle cx="102.5" cy="205" r="13" fill="#0F69E1" stroke="#FFFFFF" stroke-width="2.5"/>
  <circle cx="172.5" cy="307.5" r="13" fill="#0F69E1" stroke="#FFFFFF" stroke-width="2.5"/>
  <circle cx="267.5" cy="197.5" r="13" fill="#0F69E1" stroke="#FFFFFF" stroke-width="2.5"/>
  <circle cx="362.5" cy="97.5" r="13" fill="#0F69E1" stroke="#FFFFFF" stroke-width="2.5"/>
  <circle cx="397.5" cy="205" r="13" fill="#0F69E1" stroke="#FFFFFF" stroke-width="2.5"/>

  <!-- Node Dots (Red) -->
  <circle cx="157.5" cy="250" r="13" fill="#E11E37" stroke="#FFFFFF" stroke-width="2.5"/>
  <circle cx="177.5" cy="135" r="13" fill="#E11E37" stroke="#FFFFFF" stroke-width="2.5"/>
  <circle cx="237.5" cy="237.5" r="13" fill="#E11E37" stroke="#FFFFFF" stroke-width="2.5"/>
  <circle cx="332.5" cy="295" r="13" fill="#E11E37" stroke="#FFFFFF" stroke-width="2.5"/>
</svg>'''
    (OUT_DIR / "favicon.svg").write_text(svg_content, encoding="utf-8")


def save_ico(master_img, path):
    """Write a multi-resolution .ico.

    Pillow's ICO writer silently DROPS any requested size larger than the image it is
    called on (IcoImagePlugin skips `size > im.size`), so saving from the 16px frame
    produced a one-frame 16x16 file however many sizes were listed — a blurry upscale
    wherever Windows wants a large icon. The base must therefore be the largest size,
    with the smaller frames supplied through append_images.
    """
    frames = {sz: master_img.resize((sz, sz), Image.Resampling.LANCZOS) for sz in ICO_SIZES}
    base = frames[max(ICO_SIZES)]
    base.save(
        path,
        format="ICO",
        sizes=[(sz, sz) for sz in ICO_SIZES],
        append_images=[frames[sz] for sz in ICO_SIZES if sz != max(ICO_SIZES)],
    )
    assert_ico_frames(path, len(ICO_SIZES))
    return frames


def assert_ico_frames(path, expected):
    """Fail loudly if the .ico did not get every frame (the trap save_ico() documents)."""
    with open(path, "rb") as f:
        reserved, kind, count = struct.unpack("<HHH", f.read(6))
    if (reserved, kind) != (0, 1) or count != expected:
        raise RuntimeError(
            f"{path.name}: expected {expected} icon frames, got {count} "
            f"(reserved={reserved}, type={kind})"
        )


def save_apple_touch_icon(master_img, path, size=180):
    """Write the iOS home-screen icon, flattened onto the badge background.

    iOS does not honour alpha here: a transparent icon is composited onto black, which
    turns the badge's rounded corners into dark wedges. It applies its own corner mask,
    so the correct asset is a full-bleed opaque square.
    """
    icon = master_img.resize((size, size), Image.Resampling.LANCZOS)
    flat = Image.new("RGB", icon.size, BADGE_BG)
    flat.paste(icon, mask=icon.split()[3])
    flat.save(path, "PNG")


def main():
    if not OUT_DIR.is_dir():
        raise SystemExit(f"expected the web GUI directory at {OUT_DIR}")
    generate_svg()

    master_img = generate_master_image(1024)
    master_img.save(OUT_DIR / "icon-1024.png", "PNG")

    frames = save_ico(master_img, OUT_DIR / "favicon.ico")
    for sz in (16, 32):
        frames[sz].save(OUT_DIR / f"favicon-{sz}x{sz}.png", "PNG")
    save_apple_touch_icon(master_img, OUT_DIR / "apple-touch-icon.png")

    print(f"Regenerated icon assets in {OUT_DIR}:")
    for name in ("favicon.svg", "favicon.ico", "favicon-16x16.png",
                 "favicon-32x32.png", "apple-touch-icon.png", "icon-1024.png"):
        print(f"  {name}")


if __name__ == "__main__":
    main()
