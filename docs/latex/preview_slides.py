"""Create an approximate contact sheet for layout QA (not a PowerPoint renderer)."""

from __future__ import annotations

import textwrap
import xml.etree.ElementTree as ET
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from build_slides import EMU, H, IMAGES, W, build_slides

NS = {"p": "http://schemas.openxmlformats.org/presentationml/2006/main",
      "a": "http://schemas.openxmlformats.org/drawingml/2006/main"}
SCALE = 96
FONT = "C:/Windows/Fonts/segoeui.ttf"
BOLD = "C:/Windows/Fonts/segoeuib.ttf"


def render(slide):
    canvas = Image.new("RGB", (round(W * SCALE), round(H * SCALE)), "white")
    draw = ImageDraw.Draw(canvas)
    root = ET.fromstring(slide.xml())
    tree = root.find(".//p:spTree", NS)
    for item in tree:
        tag = item.tag.split("}")[-1]
        if tag not in {"sp", "pic"}:
            continue
        off = item.find(".//a:xfrm/a:off", NS)
        ext = item.find(".//a:xfrm/a:ext", NS)
        if off is None or ext is None:
            continue
        x = round(int(off.attrib["x"]) / EMU * SCALE)
        y = round(int(off.attrib["y"]) / EMU * SCALE)
        w = round(int(ext.attrib["cx"]) / EMU * SCALE)
        h = round(int(ext.attrib["cy"]) / EMU * SCALE)
        if tag == "pic":
            name = item.find(".//p:cNvPr", NS).attrib["name"]
            im = Image.open(IMAGES / name).convert("RGB").resize((w, h), Image.Resampling.LANCZOS)
            canvas.paste(im, (x, y))
            continue
        fill = item.find("./p:spPr/a:solidFill/a:srgbClr", NS)
        if fill is not None:
            draw.rectangle((x, y, x + w, y + h), fill="#" + fill.attrib["val"])
        body = item.find("./p:txBody", NS)
        if body is None:
            continue
        margin = round(int(body.find("a:bodyPr", NS).attrib.get("lIns", "0")) / EMU * SCALE)
        cy = y
        for paragraph in body.findall("a:p", NS):
            run = paragraph.find("a:r", NS)
            if run is None:
                continue
            rp = run.find("a:rPr", NS)
            text = run.find("a:t", NS).text or ""
            pt = int(rp.attrib["sz"]) / 100
            font = ImageFont.truetype(BOLD if rp.attrib.get("b") == "1" else FONT,
                                      max(6, round(pt * SCALE / 72)))
            col = rp.find("a:solidFill/a:srgbClr", NS)
            color = "#" + col.attrib["val"] if col is not None else "#17233B"
            max_width = w - 2 * margin
            words = text.split()
            lines = []
            cur = ""
            for word in words:
                trial = (cur + " " + word).strip()
                if draw.textlength(trial, font=font) > max_width and cur:
                    lines.append(cur)
                    cur = word
                else:
                    cur = trial
            lines.append(cur)
            for line in lines:
                draw.text((x + margin, cy), line, fill=color, font=font)
                cy += round(pt * SCALE / 72 * 1.24)
    return canvas


def main():
    slides = build_slides()
    thumbs = []
    for s in slides:
        im = render(s)
        im.thumbnail((400, 225), Image.Resampling.LANCZOS)
        thumbs.append(im)
    sheet = Image.new("RGB", (4 * 414, ((len(thumbs) + 3) // 4) * 248), "#E8EDF3")
    d = ImageDraw.Draw(sheet)
    for i, im in enumerate(thumbs):
        col, row = i % 4, i // 4
        x, y = col * 414 + 7, row * 248 + 7
        sheet.paste(im, (x, y))
        d.text((x + 2, y + 226), f"{i+1:02d} · {slides[i].title}", fill="#17233B")
    out = Path(__file__).with_name("slides_contact_sheet.png")
    sheet.save(out)
    print(out)


if __name__ == "__main__":
    main()
