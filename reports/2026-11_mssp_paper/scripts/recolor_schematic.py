"""Recolour the NeurIPS workflow schematic (figures/differentiable-oed-double-loop-draft.pptx) to the paper palette.

One hue per loop, on its frame, title and arrows only; fills are faint tints, box text stays black:
outer design loop = oslo deep blue, lab / measurement = Crameri cork green, inner FWI loop = neutral grey.
Sources and receivers stay outside the colour map so they remain very visible: sources strong red, receivers vivid gold
(black outline on the legend star; the receiver dots in the two velocity maps are enlarged by one pixel).
The two embedded velocity maps (grey scale) are mapped onto oslo; the rendered maths labels are recoloured pixel by pixel.

Works on the XML and media files (no python-pptx); needs numpy, pillow, cmcrameri, LibreOffice (soffice), pdftocairo
and Ghostscript. Equations use PowerPoint's fallback pictures; Helvetica Neue is replaced by Liberation Sans (same metrics).
Usage:  python reports/2026-11_mssp_paper/scripts/recolor_schematic.py
Output: figures/F1_loop.{pptx,pdf} (PDF cropped to the drawing)
"""
from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path

import numpy as np
from cmcrameri import cm
from PIL import Image

HERE = Path(__file__).resolve().parents[1]   # paper directory
SRC = HERE / "figures" / "differentiable-oed-double-loop-draft.pptx"
OUT = HERE / "figures" / "F1_loop"

XML_MAP = {                      # old srgb -> new srgb
    "3D8DFF": "26578C",          # outer loop frame, arrows, labels -> oslo deep blue
    "2F6FC9": "163A5D",          # receiver-update box text and frame -> oslo navy
    "F2FAFE": "F3F6FA", "F2FAFF": "F3F6FA",   # outer-loop fill -> faint tint of the oslo blue
    "CAE2C2": "EEF4EC",          # lab box fill -> faint tint of the cork green
    "ECECEC": "F2F2F2",          # inner-loop fill -> faint neutral grey
    "B8BCC4": "A6A6A6",          # inner-loop frame and box lines -> neutral grey
    "FF0000": "D7191C",          # sources: strong red (outside the map)
    "FFFF00": "FFC800",          # receivers: vivid gold (outside the map)
}
SCHEME_MAP = {"accent6": "2A6D28"}            # lab / measurement: Crameri cork green (frame, title, data arrow)
PIX_KEYS = [("2F6FC9", "163A5D"), ("3D8DFF", "26578C")]
FILL_OLD, FILL_NEW = "F2FAFF", "F3F6FA"
RED, GOLD = np.array([215, 25, 28]), np.array([255, 200, 0])

hx = lambda s: np.array([int(s[i:i + 2], 16) for i in (0, 2, 4)], float)


def recolor_xml(t):
    # equations: use PowerPoint's rendered fallback pictures (LibreOffice drops Office Math)
    t = re.sub(r"<mc:AlternateContent[^>]*>\s*<mc:Choice[^>]*>.*?</mc:Choice>\s*<mc:Fallback[^>]*>(.*?)</mc:Fallback>\s*</mc:AlternateContent>",
               r"\1", t, flags=re.S)
    t = t.replace('typeface="Helvetica Neue"', 'typeface="Liberation Sans"')    # Helvetica metrics (installed)
    for a, b in XML_MAP.items():
        t = re.sub(f'srgbClr val="{a}"', f'srgbClr val="{b}"', t, flags=re.I)
    for a, b in SCHEME_MAP.items():
        t = re.sub(rf'<a:schemeClr val="{a}"\s*/>', f'<a:srgbClr val="{b}"/>', t)
        t = re.sub(rf'<a:schemeClr val="{a}">(.*?)</a:schemeClr>', rf'<a:srgbClr val="{b}">\1</a:srgbClr>', t)
    return t


def recolor_label(a):
    """Rendered labels: blends of white / light fill with the old blues -> the same blends with the new colours."""
    rgb, al = a[..., :3].astype(float), a[..., 3:]
    out = rgb.copy()
    for bg_old, bg_new in (("FFFFFF", "FFFFFF"), (FILL_OLD, FILL_NEW)):
        b0, b1 = hx(bg_old), hx(bg_new)
        for k_old, k_new in PIX_KEYS:
            k0, k1 = hx(k_old), hx(k_new)
            d = k0 - b0
            w = np.clip(((rgb - b0) @ d) / (d @ d), 0, 1)
            res = np.linalg.norm(rgb - (b0 + w[..., None] * d), axis=-1)
            sel = (res < 18) & (w > 0.04)
            out[sel] = (b1 + w[..., None] * (k1 - b1))[sel]
        near = np.linalg.norm(rgb - b0, axis=-1) < 4
        out[near] = b1
    red = (rgb[..., 0] > 150) & (rgb[..., 1] < 90) & (rgb[..., 2] < 90)
    out[red] = RED
    return np.concatenate([np.clip(out, 0, 255), al], -1).astype(np.uint8)


def recolor_map(a):
    """Velocity maps: grey pixels -> oslo; red sources -> strong red; yellow receivers -> gold, one pixel larger."""
    rgb = a[..., :3].astype(float)
    sat = rgb.max(-1) - rgb.min(-1)
    grey = sat < 30
    lut = (cm.oslo(np.linspace(0, 1, 256))[:, :3] * 255)
    out = rgb.copy()
    g = rgb.mean(-1).astype(int)
    out[grey] = lut[g[grey]]
    red = (rgb[..., 0] > 150) & (rgb[..., 1] < 100) & (rgb[..., 2] < 100)
    yel = (rgb[..., 0] > 120) & (rgb[..., 1] > 120) & (rgb[..., 2] < 90) & ~grey
    big = yel.copy()
    for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0)):
        big |= np.roll(np.roll(yel, dy, 0), dx, 1)
    ring = np.zeros_like(big)
    for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0)):
        ring |= np.roll(np.roll(big, dy, 0), dx, 1)
    ring &= ~big
    rbig = red.copy()
    for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0)):
        rbig |= np.roll(np.roll(red, dy, 0), dx, 1)
    rring = np.zeros_like(rbig)
    for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0)):
        rring |= np.roll(np.roll(rbig, dy, 0), dx, 1)
    out[(ring | rring) & ~big & ~rbig] = (255, 255, 255)
    out[big] = GOLD
    out[rbig] = RED
    return np.concatenate([np.clip(out, 0, 255), a[..., 3:]], -1).astype(np.uint8)


def main():
    tmp = Path(tempfile.mkdtemp())
    with zipfile.ZipFile(SRC) as z:
        z.extractall(tmp)
    for f in list(tmp.rglob("*.xml")):
        t = f.read_text(encoding="utf-8")
        n = recolor_xml(t)
        if n != t:
            f.write_text(n, encoding="utf-8")
    for f in sorted((tmp / "ppt" / "media").glob("*.png")):
        a = np.asarray(Image.open(f).convert("RGBA"))
        is_map = a.shape[0] > 400 and a.shape[1] > 500
        Image.fromarray(recolor_map(a) if is_map else recolor_label(a)).save(f)
    pptx = OUT.with_suffix(".pptx")
    with zipfile.ZipFile(SRC) as zin, zipfile.ZipFile(pptx, "w", zipfile.ZIP_DEFLATED) as zout:
        for item in zin.infolist():
            zout.write(tmp / item.filename, item.filename)
    subprocess.run(["soffice", "--headless", "--convert-to", "pdf", "--outdir", str(tmp), str(pptx)], check=True,
                   capture_output=True)
    raw = tmp / (pptx.stem + ".pdf")
    subprocess.run(["pdftocairo", "-png", "-r", "72", "-singlefile", str(raw), str(tmp / "page")], check=True)
    img = np.asarray(Image.open(tmp / "page.png").convert("L"))
    ys, xs = np.nonzero(img < 250)
    hpt = img.shape[0]
    x0, x1, y0, y1 = xs.min(), xs.max() + 1, hpt - (ys.max() + 1), hpt - ys.min()     # PDF points, origin bottom left
    pad = 3
    boxed = tmp / "boxed.pdf"
    subprocess.run(["gs", "-q", "-dBATCH", "-dNOPAUSE", "-sDEVICE=pdfwrite", f"-sOutputFile={boxed}",
                    "-c", f"[/CropBox [{x0 - pad} {y0 - pad} {x1 + pad} {y1 + pad}] /PAGES pdfmark", "-f", str(raw)], check=True)
    subprocess.run(["gs", "-q", "-dBATCH", "-dNOPAUSE", "-sDEVICE=pdfwrite", "-dUseCropBox",       # crop = media box
                    f"-sOutputFile={OUT.with_suffix('.pdf')}", str(boxed)], check=True)
    shutil.rmtree(tmp)
    print(OUT.with_suffix(".pdf"))


if __name__ == "__main__":
    main()
