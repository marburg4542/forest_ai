"""Generate the PWA icon set.

    python tools/make_icons.py

Drawn rather than shipped as binary blobs so the design can be changed by
editing numbers here.  Output goes to web/static/icons/ and is committed —
the running app needs it and the files are only a few kB each.
"""

from __future__ import annotations

import pathlib

from PIL import Image, ImageDraw

OUT = pathlib.Path(__file__).resolve().parent.parent / "web" / "static" / "icons"

BG = (20, 49, 31)          # deep forest green
TREE = (74, 192, 107)      # the accent green used by the interface
TREE_DARK = (43, 138, 71)
TRUNK = (120, 92, 64)
SCAN = (255, 255, 255, 70)  # faint point-cloud speckle

# a fixed speckle pattern, in unit coordinates, so the icon is reproducible
SPECKLE = [
    (0.13, 0.30), (0.20, 0.52), (0.11, 0.68), (0.24, 0.79), (0.17, 0.88),
    (0.87, 0.30), (0.80, 0.50), (0.89, 0.66), (0.76, 0.80), (0.84, 0.88),
    (0.34, 0.90), (0.66, 0.91), (0.50, 0.95), (0.30, 0.20), (0.70, 0.21),
]


def draw_icon(size: int, inset: float = 0.0, rounded: bool = True) -> Image.Image:
    """One icon.  `inset` shrinks the artwork for maskable safe zones."""
    ss = 4                                   # supersample, then downscale
    n = size * ss
    img = Image.new("RGBA", (n, n), (0, 0, 0, 0))
    d = ImageDraw.Draw(img, "RGBA")

    if rounded:
        d.rounded_rectangle([0, 0, n - 1, n - 1], radius=int(n * 0.22), fill=BG)
    else:
        d.rectangle([0, 0, n - 1, n - 1], fill=BG)

    for fx, fy in SPECKLE:
        r = n * 0.011
        cx, cy = fx * n, fy * n
        d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=SCAN)

    # artwork box, centred, shrunk by `inset` on every side
    m = inset
    x0, x1 = n * (0.5 - (0.5 - 0.16) * (1 - m)), n * (0.5 + (0.5 - 0.16) * (1 - m))
    y0, y1 = n * (0.5 - (0.5 - 0.10) * (1 - m)), n * (0.5 + (0.5 - 0.10) * (1 - m))
    w, h = x1 - x0, y1 - y0
    cx = (x0 + x1) / 2

    # trunk
    tw = w * 0.13
    d.rounded_rectangle([cx - tw / 2, y0 + h * 0.66, cx + tw / 2, y1],
                        radius=int(tw * 0.3), fill=TRUNK)

    # three stacked canopy tiers, widest at the bottom
    tiers = [(0.00, 0.34, 0.52), (0.20, 0.56, 0.76), (0.42, 0.80, 1.00)]
    for i, (top, bot, spread) in enumerate(tiers):
        half = w * 0.5 * spread
        d.polygon([(cx, y0 + h * top),
                   (cx - half, y0 + h * bot),
                   (cx + half, y0 + h * bot)],
                  fill=TREE if i % 2 == 0 else TREE_DARK)

    return img.resize((size, size), Image.LANCZOS)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    written = []
    for size in (192, 512):
        p = OUT / f"icon-{size}.png"
        draw_icon(size).save(p)
        written.append(p)
    # maskable icons get cropped to a circle by some launchers, so the artwork
    # has to sit inside the inner 80% and the background must reach the edges
    p = OUT / "icon-maskable-512.png"
    draw_icon(512, inset=0.30, rounded=False).save(p)
    written.append(p)

    p = OUT / "favicon.png"
    draw_icon(64).save(p)
    written.append(p)

    # Windows shortcuts need a real .ico; one file holding every size the shell
    # asks for (taskbar, alt-tab, explorer at each zoom level)
    p = OUT / "forest_ai.ico"
    draw_icon(256).save(p, sizes=[(16, 16), (24, 24), (32, 32), (48, 48),
                                  (64, 64), (128, 128), (256, 256)])
    written.append(p)

    for p in written:
        print(f"  {p.relative_to(OUT.parent.parent.parent)}  {p.stat().st_size/1024:.1f} kB")


if __name__ == "__main__":
    main()
