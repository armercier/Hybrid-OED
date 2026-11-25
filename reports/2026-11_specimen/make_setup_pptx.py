"""Editable PowerPoint version of appendix figure A2 (lab setup): the photo LDV_photos/jpg/IMG_6341.jpg fills the
slide, and every overlay is a separate shape to move by hand: three green laser beams (straight connectors; drag their
end points), the scan point (small green dot), five labels (text boxes) with white leader lines (connectors).
Starting positions = those of make_setup_photo.py. After editing, export with:
  soffice --headless --convert-to pdf figures/source/A2_setup_editable.pptx
Needs python-pptx (not in the hybridOED env; run with a venv that has it).
Usage:  <venv>/bin/python reports/2026-11_specimen/make_setup_pptx.py
Output: reports/2026-11_mssp_paper/figures/source/A2_setup_editable.pptx
"""
from __future__ import annotations

from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE
from pptx.enum.text import MSO_AUTO_SIZE, PP_ALIGN
from pptx.util import Emu, Pt

REPO = Path(__file__).resolve().parents[2]
PHOTO = REPO / "reports" / "2026-11_mssp_paper" / "LDV_photos" / "jpg" / "IMG_6341.jpg"
OUT = REPO / "reports" / "2026-11_mssp_paper" / "figures" / "source" / "A2_setup_editable.pptx"
W, H = Emu(6858000), Emu(9144000)            # 7.5 x 10 in, the photo's 3:4 aspect
BEAMS_FROM = [(0.366, 0.308), (0.521, 0.296), (0.723, 0.276)]
BEAM_TO = (0.50, 0.625)
LABELS = [("3-D scanning laser vibrometer heads", (0.42, 0.24), (0.18, 0.06)),
          ("robot", (0.78, 0.47), (0.84, 0.39)),
          ("PMMA plate, 10 mm", (0.62, 0.62), (0.62, 0.80)),
          ("scan area", (0.56, 0.63), (0.64, 0.52)),
          ("foam support", (0.30, 0.90), (0.08, 0.96))]
GREEN, WHITE, INK = RGBColor(0x3C, 0xF0, 0x3C), RGBColor(0xFF, 0xFF, 0xFF), RGBColor(0, 0, 0)


def P(fx, fy):
    return Emu(int(fx * W)), Emu(int(fy * H))


def line(sl, a, b, color, width, name):
    c = sl.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, *P(*a), *P(*b))
    c.line.color.rgb = color; c.line.width = Pt(width); c.name = name
    return c


def main():
    prs = Presentation()
    prs.slide_width, prs.slide_height = W, H
    sl = prs.slides.add_slide(prs.slide_layouts[6])
    pic = sl.shapes.add_picture(str(PHOTO), 0, 0, W, H); pic.name = "photo (background, locked)"
    from lxml import etree
    cnv = pic._element.find(".//{http://schemas.openxmlformats.org/presentationml/2006/main}cNvPicPr")
    locks = cnv.find("{http://schemas.openxmlformats.org/drawingml/2006/main}picLocks")
    if locks is None:
        locks = etree.SubElement(cnv, "{http://schemas.openxmlformats.org/drawingml/2006/main}picLocks")
    for k in ("noMove", "noResize", "noRot", "noSelect", "noGrp"):
        locks.set(k, "1")
    for i, a in enumerate(BEAMS_FROM):
        line(sl, a, BEAM_TO, GREEN, 1.0, f"laser beam {i + 1}")
    d = 0.008
    dot = sl.shapes.add_shape(MSO_SHAPE.OVAL, *P(BEAM_TO[0] - d / 2, BEAM_TO[1] - d * 0.75 / 2), Emu(int(d * W)), Emu(int(d * 0.75 * W)))
    dot.fill.solid(); dot.fill.fore_color.rgb = GREEN; dot.line.fill.background(); dot.name = "scan point"
    for text, tip, box in LABELS:
        line(sl, box, tip, WHITE, 1.0, f"leader: {text}")
        tb = sl.shapes.add_textbox(*P(*box), Pt(len(text) * 6.6 + 14), Pt(20))
        tb.name = f"label: {text}"
        tf = tb.text_frame; tf.word_wrap = False; tf.auto_size = MSO_AUTO_SIZE.SHAPE_TO_FIT_TEXT
        tb.fill.solid(); tb.fill.fore_color.rgb = WHITE
        p = tf.paragraphs[0]; p.text = text; p.alignment = PP_ALIGN.LEFT
        p.runs[0].font.size = Pt(12); p.runs[0].font.color.rgb = INK; p.runs[0].font.name = "Liberation Sans"
    OUT.parent.mkdir(parents=True, exist_ok=True)
    prs.save(OUT)
    print(OUT)


if __name__ == "__main__":
    main()
