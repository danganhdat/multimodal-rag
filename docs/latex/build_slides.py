"""Build a self-contained PowerPoint from report.tex and its original figures.

Uses only Python's standard library plus Pillow, so it also works where
python-pptx and Office automation are unavailable.
"""

from __future__ import annotations

import hashlib
import re
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from xml.sax.saxutils import escape

from PIL import Image


ROOT = Path(__file__).resolve().parent
IMAGES = ROOT / "images"
OUTPUT = ROOT / "video_rag_eval_slides.pptx"
EMU = 914400
W, H = 13.333, 7.5

INK = "17233B"
MUTED = "596579"
BLUE = "2563EB"
TEAL = "0F9E90"
PALE = "F3F7FC"
LINE = "D8E2EF"
WHITE = "FFFFFF"
AMBER = "D88A17"
RED = "C24152"


def xattr(value: object) -> str:
    return escape(str(value), {'"': "&quot;"})


def emu(value: float) -> int:
    return round(value * EMU)


def rgb(color: str) -> str:
    return f'<a:solidFill><a:srgbClr val="{color}"/></a:solidFill>'


class Slide:
    def __init__(self, section: str, title: str, number: int):
        self.number = number
        self.section = section
        self.title = title
        self.parts: list[str] = []
        self.images: list[tuple[str, str]] = []
        self.next_id = 2
        self.rect(0, 0, W, H, WHITE)
        self.rect(0, 0, 0.10, H, BLUE)
        self.text(section.upper(), 0.48, 0.24, 8.0, 0.25, 10, TEAL, bold=True)
        self.text(title, 0.48, 0.55, 12.3, 0.63, 28, INK, bold=True)
        self.rect(0.48, 1.27, 12.35, 0.015, LINE)
        self.text("VIDEO RAG  /  AIC 2025", 0.49, 7.19, 6.5, 0.18, 8, MUTED)
        self.text(f"{number:02d}", 12.16, 7.14, 0.55, 0.25, 10, MUTED, align="r")

    def _id(self) -> int:
        out = self.next_id
        self.next_id += 1
        return out

    def rect(self, x: float, y: float, w: float, h: float, fill: str,
             radius: bool = False, stroke: str | None = None) -> None:
        sid = self._id()
        geom = "roundRect" if radius else "rect"
        line = f'<a:ln w="{emu(0.012)}">{rgb(stroke)}</a:ln>' if stroke else '<a:ln><a:noFill/></a:ln>'
        self.parts.append(
            f'<p:sp><p:nvSpPr><p:cNvPr id="{sid}" name="Shape {sid}"/>'
            '<p:cNvSpPr/><p:nvPr/></p:nvSpPr><p:spPr>'
            f'<a:xfrm><a:off x="{emu(x)}" y="{emu(y)}"/>'
            f'<a:ext cx="{emu(w)}" cy="{emu(h)}"/></a:xfrm>'
            f'<a:prstGeom prst="{geom}"><a:avLst/></a:prstGeom>{rgb(fill)}{line}'
            '</p:spPr></p:sp>'
        )

    def text(self, value: str, x: float, y: float, w: float, h: float,
             size: float = 18, color: str = INK, bold: bool = False,
             align: str = "l", margin: float = 0.0) -> None:
        sid = self._id()
        paragraphs = []
        for line in value.split("\n"):
            prop = (f'<a:rPr lang="vi-VN" sz="{round(size*100)}"'
                    + (' b="1"' if bold else '') + '>'
                    + rgb(color) + '<a:latin typeface="Aptos"/></a:rPr>')
            paragraphs.append(
                f'<a:p><a:pPr algn="{align}"/>'
                f'<a:r>{prop}<a:t>{xattr(line or " ")}</a:t></a:r>'
                '<a:endParaRPr lang="vi-VN"/></a:p>'
            )
        self.parts.append(
            f'<p:sp><p:nvSpPr><p:cNvPr id="{sid}" name="Text {sid}"/>'
            '<p:cNvSpPr txBox="1"/><p:nvPr/></p:nvSpPr><p:spPr>'
            f'<a:xfrm><a:off x="{emu(x)}" y="{emu(y)}"/>'
            f'<a:ext cx="{emu(w)}" cy="{emu(h)}"/></a:xfrm>'
            '<a:prstGeom prst="rect"><a:avLst/></a:prstGeom>'
            '<a:noFill/><a:ln><a:noFill/></a:ln></p:spPr>'
            f'<p:txBody><a:bodyPr wrap="square" anchor="t" lIns="{emu(margin)}" '
            f'rIns="{emu(margin)}" tIns="0" bIns="0"/><a:lstStyle/>'
            + ''.join(paragraphs) + '</p:txBody></p:sp>'
        )

    def image(self, filename: str, x: float, y: float, w: float, h: float,
              fit: str = "contain") -> None:
        path = IMAGES / filename
        if not path.is_file():
            raise FileNotFoundError(path)
        iw, ih = Image.open(path).size
        if fit == "contain":
            scale = min(w / iw, h / ih)
            pw, ph = iw * scale, ih * scale
            x += (w - pw) / 2
            y += (h - ph) / 2
            w, h = pw, ph
        sid = self._id()
        rid = f"rId{len(self.images) + 2}"
        self.images.append((rid, filename))
        self.parts.append(
            f'<p:pic><p:nvPicPr><p:cNvPr id="{sid}" name="{xattr(filename)}" '
            f'descr="Original figure: {xattr(filename)}"/>'
            '<p:cNvPicPr><a:picLocks noChangeAspect="1"/></p:cNvPicPr><p:nvPr/></p:nvPicPr>'
            f'<p:blipFill><a:blip r:embed="{rid}"/><a:stretch><a:fillRect/></a:stretch></p:blipFill>'
            f'<p:spPr><a:xfrm><a:off x="{emu(x)}" y="{emu(y)}"/>'
            f'<a:ext cx="{emu(w)}" cy="{emu(h)}"/></a:xfrm>'
            '<a:prstGeom prst="rect"><a:avLst/></a:prstGeom>'
            '<a:ln><a:noFill/></a:ln></p:spPr></p:pic>'
        )

    def card(self, title: str, body: str, x: float, y: float, w: float,
             h: float, accent: str = BLUE, body_size: int = 15) -> None:
        self.rect(x, y, w, h, PALE, radius=True, stroke=LINE)
        self.rect(x, y, 0.055, h, accent)
        self.text(title, x + 0.22, y + 0.18, w - 0.42, 0.38, 17, accent, bold=True)
        self.text(body, x + 0.22, y + 0.68, w - 0.45, h - 0.78, body_size, INK)

    def chip(self, label: str, x: float, y: float, w: float,
             fill: str = PALE, color: str = BLUE) -> None:
        self.rect(x, y, w, 0.37, fill, radius=True)
        self.text(label, x + 0.12, y + 0.07, w - 0.22, 0.21, 11, color, bold=True)

    def bar(self, label: str, value: float, maxval: float, x: float, y: float,
            width: float, color: str = BLUE, display: str | None = None) -> None:
        self.text(label, x, y, 1.32, 0.28, 13, INK, bold=True)
        self.rect(x + 1.45, y + 0.06, width - 2.12, 0.18, LINE, radius=True)
        self.rect(x + 1.45, y + 0.06, max(0.02, (width - 2.12) * value / maxval),
                  0.18, color, radius=True)
        self.text(display or f"{value:.3f}", x + width - 0.55, y - 0.01,
                  0.55, 0.28, 13, color, bold=True, align="r")

    def xml(self) -> str:
        group = (
            '<p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr>'
            '<p:grpSpPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="0" cy="0"/>'
            '<a:chOff x="0" y="0"/><a:chExt cx="0" cy="0"/></a:xfrm></p:grpSpPr>'
        )
        return (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<p:sld xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" '
            'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" '
            'xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">'
            '<p:cSld><p:spTree>' + group + ''.join(self.parts) + '</p:spTree></p:cSld>'
            '<p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr></p:sld>'
        )


def relationships(items: list[tuple[str, str, str]]) -> str:
    return ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            + ''.join(f'<Relationship Id="{rid}" Type="{typ}" Target="{target}"/>'
                      for rid, typ, target in items) + '</Relationships>')


def build_slides() -> list[Slide]:
    slides: list[Slide] = []

    def new(section: str, title: str) -> Slide:
        slide = Slide(section, title, len(slides) + 1)
        slides.append(slide)
        return slide

    s = new("Mở đầu", "Video RAG đa phương thức")
    s.text("Truy hồi khoảnh khắc • trả lời có bằng chứng", 0.55, 1.67, 11.8, 0.8,
           29, INK, bold=True)
    s.rect(0.55, 2.72, 1.1, 0.08, TEAL)
    s.text("Từ 873 video đến benchmark 20 QA; đánh giá retrieval, VQA và ReAct agent.",
           0.55, 3.12, 11.5, 1.0, 22, MUTED)
    s.text("Trang Kỳ Anh · Lê Nguyễn Bảo Hân · Đặng Anh Đạt",
           0.55, 4.22, 11.8, 0.38, 16, INK, bold=True)
    s.text("GVHD: TS. Phạm Đỗ Kim Chi · UIT, ĐHQG TP.HCM",
           0.55, 4.64, 11.8, 0.35, 14, MUTED)
    s.chip("EVAL", 0.55, 5.49, 1.3)
    s.chip("ABLATION", 1.99, 5.49, 1.68)
    s.chip("ERROR ANALYSIS", 3.80, 5.49, 2.4)
    s.text("Nguồn: docs/latex/report.tex · hình gốc: docs/latex/images", 0.55, 6.47,
           11.0, 0.3, 13, MUTED)

    s = new("Bài toán", "Từ video dài đến truy hồi frame-level")
    s.text("Câu hỏi về một khoảnh khắc trong hàng trăm video", 0.63, 1.59,
           12.0, 0.54, 21, INK, bold=True)
    s.card("Video-level", "Truy hồi video → temporal grounding.\nVideo encoder nặng; moment ngắn dễ bị pha loãng.",
           0.60, 2.35, 5.33, 2.38, RED, 17)
    s.text("→", 6.14, 3.16, 0.72, 0.55, 30, BLUE, bold=True, align="ctr")
    s.card("Frame-level", "Shot boundary → keyframe đại diện.\nMỗi frame có ảnh, OCR, objects và timestamp.",
           7.03, 2.35, 5.67, 2.38, TEAL, 17)
    s.rect(0.61, 5.29, 12.02, 1.08, PALE, radius=True)
    s.text("Mục tiêu: tìm đúng frame có bằng chứng, rồi mới để VLM trả lời.",
           0.89, 5.63, 11.4, 0.40, 20, INK, bold=True)

    s = new("Dữ liệu", "Shot boundary → keyframe")
    s.image("transnetv2.png", 0.62, 1.52, 5.0, 5.28)
    s.card("Corpus", "873 video L21–L30\n168.749 keyframe sau xử lý\nFrame-level index + OCR + objects + timestamp",
           6.0, 1.78, 6.48, 2.22, BLUE, 17)
    s.card("TransNetV2", "Chọn frame đại diện tại ranh giới shot.\nLọc frame trắng/đen trước khi lập chỉ mục.",
           6.0, 4.28, 6.48, 1.91, TEAL, 17)

    s = new("Kiến trúc", "5 tầng: offline nặng, online nhẹ")
    s.image("diagram_1.png", 0.52, 1.75, 12.27, 2.11)
    s.card("Offline", "Shot detection → keyframe → SigLIP2 / OCR / objects → Milvus + BM25",
           0.63, 4.19, 5.92, 1.57, BLUE, 16)
    s.card("Online", "Parse → hybrid retrieval → temporal matching → rerank → VQA / agent",
           6.74, 4.19, 5.92, 1.57, TEAL, 16)

    s = new("Phương pháp", "Hiểu truy vấn tiếng Việt")
    s.image("diagram_2.png", 0.52, 1.55, 12.28, 3.62)
    s.card("Hai tín hiệu song song", "LLM parse sự kiện/quan hệ thời gian; regex lấy OCR literal và object.",
           0.62, 5.42, 6.03, 1.1, BLUE, 14)
    s.card("Fallback", "LLM lỗi → dịch truy vấn và tạo một sự kiện để vẫn tìm kiếm được.",
           6.83, 5.42, 5.84, 1.1, TEAL, 14)

    s = new("Phương pháp", "Truy hồi lai và RRF")
    s.image("diagram_3.png", 0.64, 1.45, 12.05, 5.49)

    s = new("Phương pháp", "Temporal chain matching")
    s.image("diagram_4.png", 0.55, 1.75, 12.22, 1.58)
    s.card("Điều kiện", "Cùng video • đúng thứ tự pts_time • gap ≤ ngưỡng.",
           0.62, 3.79, 3.87, 1.90, BLUE, 17)
    s.card("Điểm chain", "Mean relevance × coverage; giữ top 20 chain và khử trùng.",
           4.72, 3.79, 3.87, 1.90, TEAL, 17)
    s.card("Giới hạn", "Greedy có thể chọn sai khi cảnh tương tự lặp lại trong video.",
           8.82, 3.79, 3.87, 1.90, AMBER, 17)

    s = new("Phương pháp", "Manual và Agentic: chung core, khác luồng")
    s.rect(0.61, 1.52, 12.10, 0.86, PALE, radius=True, stroke=LINE)
    s.text("SHARED CORE", 0.83, 1.78, 1.69, 0.28, 13, BLUE, bold=True)
    s.text("SigLIP2Encoder · MilvusSearchService · pipeline.run_search()",
           2.51, 1.68, 9.76, 0.32, 17, INK, bold=True)
    s.text("Feedback up/down · traces và spans", 2.52, 2.06, 9.36, 0.22, 12, MUTED)

    def mode_node(label: str, detail: str, x: float, y: float,
                  width: float, accent: str) -> None:
        s.rect(x, y, width, 0.94, WHITE, radius=True, stroke=LINE)
        s.rect(x, y, 0.055, 0.94, accent)
        s.text(label, x + 0.13, y + 0.18, width - 0.24, 0.29,
               13, accent, bold=True)
        s.text(detail, x + 0.13, y + 0.55, width - 0.24, 0.23,
               11, MUTED)

    s.rect(0.61, 2.65, 12.10, 1.56, PALE, radius=True, stroke=LINE)
    s.text("MANUAL", 0.79, 2.98, 1.10, 0.31, 16, BLUE, bold=True)
    s.text("/api/v1/vqa", 0.79, 3.38, 1.12, 0.24, 11, MUTED)
    manual_steps = [
        ("S1–S2 · QE v2", "QueryPlan"),
        ("S3 · search", "variants + RRF"),
        ("S4 · boost", "metadata"),
        ("S5 · temporal", "chuỗi sự kiện"),
        ("S6 · rerank", "MLLM listwise"),
        ("S7 · VQA", "one-shot answer"),
    ]
    for idx, (label, detail) in enumerate(manual_steps):
        xx = 1.91 + idx * 1.78
        mode_node(label, detail, xx, 2.97, 1.55, BLUE)
        if idx < len(manual_steps) - 1:
            s.text("→", xx + 1.60, 3.23, 0.16, 0.29, 19, BLUE, bold=True,
                   align="ctr")

    s.rect(0.61, 4.49, 12.10, 1.56, PALE, radius=True, stroke=LINE)
    s.text("AGENTIC", 0.79, 4.82, 1.11, 0.31, 16, TEAL, bold=True)
    s.text("/api/v1/chat", 0.79, 5.22, 1.12, 0.24, 11, MUTED)
    agent_steps = [
        ("ReAct reasoning", "chọn tool tiếp theo"),
        ("search / events", "dense · RRF · temporal"),
        ("look / neighbors", "VLM · crop · context"),
        ("final_answer", "guard → SSE stream"),
    ]
    for idx, (label, detail) in enumerate(agent_steps):
        xx = 2.04 + idx * 2.68
        mode_node(label, detail, xx, 4.80, 2.30, TEAL)
        if idx < len(agent_steps) - 1:
            s.text("→", xx + 2.38, 5.06, 0.22, 0.29, 21, TEAL, bold=True,
                   align="ctr")

    s.text("Agentic tự search và xem frame; one-shot Manual chạy pipeline S1–S7 một lần.",
           0.81, 6.44, 11.85, 0.38, 17, INK)

    s = new("Phương pháp", "One-shot VQA: ảnh + OCR")
    s.image("diagram_5.png", 0.54, 1.62, 12.21, 3.22)
    s.card("Evidence", "Chọn keyframe từ retrieval; đưa ảnh JPEG và OCR text vào cùng prompt.",
           0.61, 5.08, 5.98, 1.37, BLUE, 15)
    s.card("Guard", "Thiếu evidence → không trả lời; không suy đoán thông tin ngoài frame.",
           6.80, 5.08, 5.91, 1.37, TEAL, 15)

    s = new("Phương pháp", "Agent ReAct: search → look → guard")
    s.image("diagram_6.png", 0.69, 1.44, 11.97, 5.57)

    s = new("Tổng quan", "Kết quả chính của hệ thống")
    s.card("20 QA / 416 qrels", "Silver benchmark\n14 test • 6 dev\n128 / 873 video được quét",
           0.55, 1.68, 3.82, 3.55, TEAL, 19)
    s.card("0,694", "Visual Context Recall\nE8 ReAct agent\nBaseline E1: 0,528",
           4.59, 1.68, 3.82, 3.55, BLUE, 19)
    s.card("0,929", "Faithfulness nhị phân\nE8 ReAct agent\nLatency p50: 54,5 s",
           8.63, 1.68, 3.82, 3.55, AMBER, 19)
    s.text("Đánh đổi: agent tìm và kiểm chứng tốt hơn, nhưng tốn nhiều lượt VLM tuần tự.",
           0.60, 5.77, 12.0, 0.65, 20, INK)

    s = new("Đánh giá", "Tạo benchmark silver")
    s.image("diagram_7.png", 0.55, 1.63, 12.2, 1.95)
    s.card("Khử trùng trước lấy mẫu",
           "Gom frame gần trùng bằng hash, pHash, SSIM, SigLIP2 và OCR; chỉ chọn frame đại diện.",
           0.62, 3.91, 3.86, 1.98, BLUE, 14)
    s.card("Hai lượt judge từ ảnh",
           "Mỗi lượt tự trả lời từ ảnh, không thấy đáp án chuẩn; bất đồng hoặc không đạt → loại.",
           4.74, 3.91, 3.86, 1.98, TEAL, 14)
    s.card("Kiểm tra answer leak",
           "Mô tả/câu hỏi không được lộ đáp án; 2 proposal bị gắn cờ leaks_answer.",
           8.86, 3.91, 3.86, 1.98, AMBER, 14)
    s.text("128 video → 120 frame mẫu → 43 proposal → 20 QA đóng băng → 416 qrels silver.",
           0.68, 6.10, 12.0, 0.38, 17, INK, bold=True)
    s.text("Chưa có probe hỏi model không kèm ảnh; leak gate hiện chỉ kiểm tra thông tin lộ trong prompt.",
           0.68, 6.56, 12.0, 0.28, 12, RED)

    s = new("Đánh giá", "Từ nhãn frame đến metric")
    s.image("diagram_8.png", 0.55, 1.56, 12.18, 3.83)
    s.text("Qrels + accepted ranges dùng cho retrieval; evidence images dùng cho visual judges.",
           0.66, 5.73, 12.0, 0.60, 20, INK)

    s = new("Đánh giá", "Bốn metric cốt lõi")
    s.card("Visual Context Recall", "Tỷ lệ claim được các frame context hỗ trợ.",
           0.55, 1.57, 6.03, 1.14, BLUE, 13)
    s.card("Context Precision", "Average precision theo thứ hạng frame trong context.",
           6.77, 1.57, 6.03, 1.14, TEAL, 13)
    s.card("Faithfulness", "RAGAS MultiModalFaithfulness: verdict nhị phân.",
           0.55, 2.88, 6.03, 1.14, AMBER, 13)
    s.card("Answer Relevance", "RAGAS AnswerRelevancy: độ đúng trọng tâm, không đo fact.",
           6.77, 2.88, 6.03, 1.14, RED, 13)
    # diagram_9 is byte-identical to diagram_8; retain both named source figures.
    s.image("diagram_9.png", 1.56, 4.34, 10.2, 2.36)

    s = new("Đánh giá", "Giao thức và giới hạn diễn giải")
    s.card("Chia tập", "20 QA: 6 dev / 14 test. Split theo video và nhóm near-duplicate.",
           0.55, 1.65, 5.96, 1.70, BLUE, 16)
    s.card("Qrels hữu hạn", "Unjudged ≠ negative. Pool 416 qrels không bao phủ toàn bộ corpus.",
           6.76, 1.65, 5.96, 1.70, TEAL, 16)
    s.card("Judge", "Generator và judge dùng cùng checkpoint với prompt độc lập; có correlated bias.",
           0.55, 3.58, 5.96, 1.70, AMBER, 16)
    s.card("Phạm vi", "E1–E8 đánh giá dense SigLIP2; chưa đo BM25 hybrid và MLLM rerank production.",
           6.76, 3.58, 5.96, 1.70, RED, 16)
    s.text("So sánh ablation là bằng chứng pilot, chưa đủ để kết luận mạnh trên toàn bộ 873 video.",
           0.63, 5.90, 12.0, 0.52, 18, INK)
    s.text("Metric bổ sung: Recall@k · MRR · nDCG · Answer Correctness · Negative Rejection · Latency p50/p95.",
           0.63, 6.43, 12.0, 0.34, 13, MUTED)

    s = new("Ablation", "E1–E8: mỗi bước kiểm tra một thành phần")
    variants = [
        ("E1", "Dense SigLIP2", "baseline"),
        ("E2", "+ query variants + RRF", "retrieval"),
        ("E3", "+ object soft boost", "metadata"),
        ("E4", "+ SuperGlobal", "rerank"),
        ("E5", "+ visual crops", "context"),
        ("E6", "+ temporal evidence", "context"),
        ("E7", "+ shot summaries", "context"),
        ("E8", "ReAct agent", "agent"),
    ]
    for idx, (code, name, group) in enumerate(variants):
        col, row = idx % 2, idx // 2
        xx, yy = 0.58 + col * 6.17, 1.61 + row * 1.24
        s.rect(xx, yy, 5.95, 1.06, PALE, radius=True, stroke=LINE)
        s.chip(code, xx + 0.17, yy + 0.32, 0.72,
               fill=BLUE if code == "E8" else "E4EDFF",
               color=WHITE if code == "E8" else BLUE)
        s.text(name, xx + 1.07, yy + 0.20, 3.95, 0.35, 16, INK, bold=True)
        s.text(group, xx + 1.08, yy + 0.59, 3.2, 0.25, 11, MUTED)

    s = new("Ablation", "Retrieval: RRF giúp, SuperGlobal hại")
    s.text("Recall@10", 0.66, 1.64, 4.0, 0.35, 19, INK, bold=True)
    for i, (name, value, color) in enumerate([
        ("E1 Dense", 0.392, MUTED), ("E2 +RRF", 0.458, TEAL),
        ("E3 +Object", 0.458, BLUE), ("E4 +SG", 0.362, RED),
    ]):
        s.bar(name, value, 0.5, 0.68, 2.18 + i * 0.72, 5.84, color)
    s.text("MRR@10", 7.05, 1.64, 4.0, 0.35, 19, INK, bold=True)
    for i, (name, value, color) in enumerate([
        ("E1 Dense", 0.417, MUTED), ("E2 +RRF", 0.536, TEAL),
        ("E3 +Object", 0.536, BLUE), ("E4 +SG", 0.326, RED),
    ]):
        s.bar(name, value, 0.6, 7.07, 2.18 + i * 0.72, 5.57, color)
    s.rect(0.61, 5.37, 12.04, 1.1, PALE, radius=True)
    s.text("E1→E2: Recall@10 +17%, MRR@10 +29%. E4 giảm MRR@10 khoảng 0,21.",
           0.85, 5.70, 11.5, 0.48, 19, INK, bold=True)

    s = new("Ablation", "VQA: agent cải thiện chất lượng context")
    s.text("Config", 0.72, 1.64, 1.1, 0.25, 12, MUTED, bold=True)
    s.text("Context recall", 2.19, 1.64, 3.1, 0.25, 12, MUTED, bold=True)
    s.text("Context precision", 5.78, 1.64, 3.2, 0.25, 12, MUTED, bold=True)
    s.text("Faithfulness", 9.51, 1.64, 2.8, 0.25, 12, MUTED, bold=True)
    values = [
        ("E1", .528, .417, .786), ("E2", .472, .476, .571),
        ("E3", .639, .476, .692), ("E4", .333, .108, 1.0),
        ("E5", .639, .476, .786), ("E6", .569, .276, .714),
        ("E7", .639, .476, .714), ("E8", .694, .603, .929),
    ]
    for i, (name, recall, precision, faith) in enumerate(values):
        yy = 2.11 + i * 0.49
        if i % 2 == 0:
            s.rect(0.61, yy - 0.05, 12.03, 0.43, PALE)
        s.text(name, 0.75, yy, 0.8, 0.27, 14, BLUE if name == "E8" else INK,
               bold=name == "E8")
        for xx, value, barw, col in [
            (2.19, recall, 1.8, BLUE), (5.78, precision, 1.8, TEAL),
            (9.51, faith, 1.64, AMBER),
        ]:
            s.rect(xx, yy + 0.10, barw, 0.11, LINE, radius=True)
            s.rect(xx, yy + 0.10, barw * value, 0.11, col, radius=True)
            s.text(f"{value:.3f}", xx + barw + 0.12, yy, 0.6, 0.25, 13, INK)
    s.text("E8: context recall 0,694 · precision 0,603 · faithfulness 0,929.\nAnswer correctness 0,250 · negative rejection 1,0.",
           0.69, 6.18, 11.9, 0.70, 15, INK, bold=True)

    s = new("Ablation", "Đánh đổi chất lượng ↔ độ trễ")
    s.text("Latency p50 (giây)", 0.67, 1.60, 5.0, 0.35, 19, INK, bold=True)
    for i, (name, value, color) in enumerate([
        ("E2 +RRF", 9.8, BLUE), ("E5 +Crops", 42.7, TEAL),
        ("E7 +Summary", 52.9, AMBER), ("E8 Agent", 54.5, RED),
    ]):
        s.bar(name, value, 60, 0.70, 2.18 + i * 0.78, 7.26, color,
              display=f"{value:.1f}s")
    s.card("Đọc kết quả cùng nhau", "E5: Answer Relevance 0,796 nhưng chậm 42,7s.\nE8: evidence tốt nhất, latency 54,5s.",
           8.38, 1.89, 4.13, 3.25, TEAL, 16)
    s.text("Answer correctness chỉ 0,167–0,250; pilot 14 test QA và phép so khớp tiếng Việt nghiêm ngặt.",
           0.69, 5.91, 12.0, 0.65, 18, INK)

    s = new("Error analysis", "Lỗi xảy ra ở nhiều tầng")
    errors = [
        ("Retrieval", "Vocabulary mismatch; dense trượt cảnh diễn đạt khác."),
        ("OCR miss", "OCR extraction thiếu hoặc sai chữ nên BM25 không match."),
        ("Selection", "Top-3 gần điểm chọn sai frame dù đã tìm đúng video."),
        ("Translation", "VI→EN mất sắc thái hoặc đổi bối cảnh tìm kiếm."),
        ("Temporal", "Cảnh lặp làm greedy chọn sai thứ tự event."),
        ("Generation", "VLM suy đoán khi evidence không đủ rõ."),
        ("Unsupported script", "VLM không đọc được chữ Hán-Nôm, thư pháp, chữ viết tay."),
    ]
    for i, (heading, body) in enumerate(errors):
        col, row = i % 2, i // 2
        xx, yy = 0.59 + col * 6.12, 1.55 + row * 1.27
        s.card(heading, body, xx, yy, 5.93, 1.14,
               [BLUE, TEAL, AMBER, RED, BLUE, TEAL, AMBER][i], 12)

    s = new("Error analysis", "Hai ca lỗi cho thấy giới hạn thực tế")
    s.card("Ca 1 · thơ tại đình Nguyễn Trung Trực", "Đúng frame nhưng Florence-2 và VLM không đọc được chữ thư pháp Hán-Nôm. Agent look/crop vẫn không lấy được nội dung.\n→ Cần OCR chuyên biệt hoặc từ chối có căn cứ.",
           0.59, 1.72, 5.94, 3.84, RED, 17)
    s.card("Ca 2 · công thức có 200g thịt xay", "Manual: OCR “200g” match nhiều công thức, chọn nhầm top-3.\nAgent: search → look L30_V078/031 → đọc “Nhân bánh cuốn”.\n→ Bước verify bằng ảnh giúp sửa lỗi chọn frame.",
           6.76, 1.72, 5.94, 3.84, TEAL, 17)
    s.text("Lỗi cuối cùng có thể đến từ retrieval, evidence selection hoặc khả năng đọc của VLM.",
           0.67, 6.04, 12.0, 0.53, 18, INK)

    s = new("Phụ lục số liệu", "Retrieval ablation · đầy đủ E1–E8")
    retrieval_rows = [
        ("E1", "Dense baseline", ".392", ".578", ".701", ".417", ".512"),
        ("E2", "Variants + RRF", ".458", ".612", ".727", ".536", ".587"),
        ("E3", "Object boost", ".458", ".607", ".727", ".536", ".589"),
        ("E4", "SuperGlobal", ".362", ".612", ".727", ".326", ".427"),
        ("E5", "Runtime crops", ".458", ".612", ".727", ".536", ".587"),
        ("E6", "Temporal evidence", ".458", ".612", ".727", ".536", ".587"),
        ("E7", "Shot summaries", ".458", ".612", ".727", ".536", ".587"),
        ("E8", "ReAct agent", ".458", ".612", ".727", ".536", ".587"),
    ]
    rcols = [("ID", .72, .55), ("Cấu hình", 1.40, 3.58),
             ("R@10", 5.10, 1.12), ("R@50", 6.48, 1.12),
             ("R@300", 7.85, 1.12), ("MRR@10", 9.22, 1.23),
             ("nDCG@10", 10.77, 1.43)]
    s.rect(.60, 1.63, 12.11, .50, INK)
    for label, xx, ww in rcols:
        s.text(label, xx, 1.75, ww, .27, 13, WHITE, bold=True)
    for i, row in enumerate(retrieval_rows):
        yy = 2.18 + i * .50
        if i % 2 == 0:
            s.rect(.60, yy, 12.11, .48, PALE)
        for j, (_, xx, ww) in enumerate(rcols):
            s.text(row[j], xx, yy + .11, ww, .28, 14,
                   TEAL if row[0] == "E8" else INK,
                   bold=(j == 0 or row[0] == "E8"))
    s.text("14 test QA · range-aware matching · E5–E8 dùng retrieval đầu vào giống E2.",
           .72, 6.48, 11.7, .37, 14, MUTED)

    s = new("Phụ lục số liệu", "VQA ablation · đầy đủ E1–E8")
    vqa_rows = [
        ("E1", ".250", ".786", ".528", ".417", ".655", "1.0", "21.9"),
        ("E2", ".250", ".571", ".472", ".476", ".680", "1.0", "9.8"),
        ("E3", ".250", ".692", ".639", ".476", ".715", "1.0", "10.4"),
        ("E4", ".167", "1.000", ".333", ".108", ".748", "1.0", "8.8"),
        ("E5", ".167", ".786", ".639", ".476", ".796", "1.0", "42.7"),
        ("E6", ".250", ".714", ".569", ".276", ".740", ".5", "11.9"),
        ("E7", ".250", ".714", ".639", ".476", ".673", "1.0", "52.9"),
        ("E8", ".250", ".929", ".694", ".603", ".750", "1.0", "54.5"),
    ]
    vcols = [("ID", .70, .55), ("Correct", 1.42, 1.13),
             ("Faith", 2.88, .95), ("Visual CR", 4.15, 1.22),
             ("CPrec AP", 5.74, 1.34), ("Relevance", 7.42, 1.35),
             ("Neg. rej.", 9.20, 1.33), ("p50 (s)", 10.98, 1.25)]
    s.rect(.60, 1.63, 12.11, .50, INK)
    for label, xx, ww in vcols:
        s.text(label, xx, 1.75, ww, .27, 12, WHITE, bold=True)
    for i, row in enumerate(vqa_rows):
        yy = 2.18 + i * .50
        if i % 2 == 0:
            s.rect(.60, yy, 12.11, .48, PALE)
        for j, (_, xx, ww) in enumerate(vcols):
            s.text(row[j], xx, yy + .11, ww, .28, 14,
                   TEAL if row[0] == "E8" else INK,
                   bold=(j == 0 or row[0] == "E8"))
    s.text("Faith = RAGAS multimodal binary · Visual CR = Visual Context Recall · Correct = Answer Correctness.",
           .72, 6.48, 11.8, .37, 13, MUTED)

    s = new("Kết luận", "Điều cần giữ và bước tiếp theo")
    s.card("Giữ", "Frame-level indexing; dense + BM25 + RRF; evidence guard; agent verify khi câu hỏi khó.",
           0.60, 1.69, 5.90, 2.45, TEAL, 18)
    s.card("Cải thiện", "Mở rộng benchmark >300 QA; human audit; OCR chuyên biệt; chọn temporal evidence có kiểm soát.",
           6.77, 1.69, 5.90, 2.45, BLUE, 18)
    s.text("Thông điệp chính", 0.65, 5.02, 4.0, 0.4, 16, MUTED, bold=True)
    s.text("Retrieval tạo trần chất lượng; agent nâng chất lượng evidence bằng cách kiểm chứng frame.",
           0.65, 5.51, 11.95, 0.8, 23, INK, bold=True)

    return slides


def create_pptx(slides: list[Slide]) -> None:
    pns = "http://schemas.openxmlformats.org/presentationml/2006/main"
    ans = "http://schemas.openxmlformats.org/drawingml/2006/main"
    rns = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
    relbase = "http://schemas.openxmlformats.org/officeDocument/2006/relationships/"

    slide_ids = ''.join(f'<p:sldId id="{256+i}" r:id="rId{2+i}"/>'
                        for i in range(len(slides)))
    presentation = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        f'<p:presentation xmlns:a="{ans}" xmlns:r="{rns}" xmlns:p="{pns}">'
        '<p:sldMasterIdLst><p:sldMasterId id="2147483648" r:id="rId1"/></p:sldMasterIdLst>'
        f'<p:sldIdLst>{slide_ids}</p:sldIdLst>'
        f'<p:sldSz cx="{emu(W)}" cy="{emu(H)}" type="screen16x9"/>'
        '<p:notesSz cx="6858000" cy="9144000"/></p:presentation>'
    )
    pres_rels = [("rId1", relbase + "slideMaster", "slideMasters/slideMaster1.xml")]
    pres_rels += [(f"rId{2+i}", relbase + "slide", f"slides/slide{i+1}.xml")
                  for i in range(len(slides))]

    group = (
        '<p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr>'
        '<p:grpSpPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="0" cy="0"/>'
        '<a:chOff x="0" y="0"/><a:chExt cx="0" cy="0"/></a:xfrm></p:grpSpPr>'
    )
    master = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        f'<p:sldMaster xmlns:a="{ans}" xmlns:r="{rns}" xmlns:p="{pns}">'
        f'<p:cSld name="Blank"><p:spTree>{group}</p:spTree></p:cSld>'
        '<p:clrMap accent1="accent1" accent2="accent2" accent3="accent3" '
        'accent4="accent4" accent5="accent5" accent6="accent6" '
        'bg1="lt1" bg2="lt2" folHlink="folHlink" hlink="hlink" '
        'tx1="dk1" tx2="dk2"/>'
        '<p:sldLayoutIdLst><p:sldLayoutId id="2147483649" r:id="rId1"/>'
        '</p:sldLayoutIdLst><p:txStyles><p:titleStyle/><p:bodyStyle/>'
        '<p:otherStyle/></p:txStyles></p:sldMaster>'
    )
    layout = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        f'<p:sldLayout xmlns:a="{ans}" xmlns:r="{rns}" xmlns:p="{pns}" '
        f'type="blank" preserve="1"><p:cSld name="Blank"><p:spTree>{group}'
        '</p:spTree></p:cSld><p:clrMapOvr><a:masterClrMapping/>'
        '</p:clrMapOvr></p:sldLayout>'
    )
    theme_colors = [("dk1", "000000"), ("lt1", WHITE), ("dk2", INK),
                    ("lt2", PALE), ("accent1", BLUE), ("accent2", TEAL),
                    ("accent3", AMBER), ("accent4", RED), ("accent5", "7C6EC6"),
                    ("accent6", "5C829B"), ("hlink", BLUE), ("folHlink", "7C6EC6")]
    theme = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        f'<a:theme xmlns:a="{ans}" name="Minimal Video RAG"><a:themeElements>'
        '<a:clrScheme name="Video RAG">'
        + ''.join(f'<a:{name}><a:srgbClr val="{color}"/></a:{name}>'
                  for name, color in theme_colors)
        + '</a:clrScheme><a:fontScheme name="Aptos">'
        '<a:majorFont><a:latin typeface="Aptos Display"/><a:ea typeface=""/>'
        '<a:cs typeface=""/></a:majorFont><a:minorFont><a:latin typeface="Aptos"/>'
        '<a:ea typeface=""/><a:cs typeface=""/></a:minorFont></a:fontScheme>'
        '<a:fmtScheme name="Minimal"><a:fillStyleLst>'
        '<a:solidFill><a:schemeClr val="accent1"/></a:solidFill>'
        '<a:solidFill><a:schemeClr val="accent2"/></a:solidFill>'
        '<a:solidFill><a:schemeClr val="lt1"/></a:solidFill>'
        '</a:fillStyleLst><a:lnStyleLst>'
        '<a:ln w="9525"><a:solidFill><a:schemeClr val="accent1"/></a:solidFill></a:ln>'
        '<a:ln w="19050"><a:solidFill><a:schemeClr val="accent2"/></a:solidFill></a:ln>'
        '<a:ln w="28575"><a:solidFill><a:schemeClr val="dk2"/></a:solidFill></a:ln>'
        '</a:lnStyleLst><a:effectStyleLst>'
        '<a:effectStyle><a:effectLst/></a:effectStyle>'
        '<a:effectStyle><a:effectLst/></a:effectStyle>'
        '<a:effectStyle><a:effectLst/></a:effectStyle>'
        '</a:effectStyleLst><a:bgFillStyleLst>'
        '<a:solidFill><a:schemeClr val="lt1"/></a:solidFill>'
        '<a:solidFill><a:schemeClr val="lt2"/></a:solidFill>'
        '<a:solidFill><a:schemeClr val="accent1"/></a:solidFill>'
        '</a:bgFillStyleLst>'
        '</a:fmtScheme></a:themeElements><a:objectDefaults/><a:extraClrSchemeLst/>'
        '</a:theme>'
    )

    media_names = sorted({name for slide in slides for _, name in slide.images})
    ct = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Default Extension="xml" ContentType="application/xml"/>'
        '<Default Extension="png" ContentType="image/png"/>'
        '<Override PartName="/ppt/presentation.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.presentation.main+xml"/>'
        '<Override PartName="/ppt/slideMasters/slideMaster1.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slideMaster+xml"/>'
        '<Override PartName="/ppt/slideLayouts/slideLayout1.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slideLayout+xml"/>'
        '<Override PartName="/ppt/theme/theme1.xml" ContentType="application/vnd.openxmlformats-officedocument.theme+xml"/>'
        '<Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>'
        '<Override PartName="/docProps/app.xml" ContentType="application/vnd.openxmlformats-officedocument.extended-properties+xml"/>'
        + ''.join(f'<Override PartName="/ppt/slides/slide{i+1}.xml" '
                  'ContentType="application/vnd.openxmlformats-officedocument.presentationml.slide+xml"/>'
                  for i in range(len(slides))) + '</Types>'
    )
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    core = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" '
        'xmlns:dc="http://purl.org/dc/elements/1.1/" '
        'xmlns:dcterms="http://purl.org/dc/terms/" '
        'xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">'
        '<dc:title>Video RAG đa phương thức — Eval, Ablation, Error Analysis</dc:title>'
        '<dc:creator>Video RAG team</dc:creator>'
        f'<dcterms:created xsi:type="dcterms:W3CDTF">{now}</dcterms:created>'
        '</cp:coreProperties>'
    )
    app = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties" '
        'xmlns:vt="http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes">'
        '<Application>Microsoft Office PowerPoint</Application>'
        f'<Slides>{len(slides)}</Slides></Properties>'
    )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(OUTPUT, "w", compression=zipfile.ZIP_DEFLATED,
                         compresslevel=6) as z:
        def put(name: str, content: str | bytes) -> None:
            z.writestr(name, content.encode("utf-8") if isinstance(content, str) else content)

        put("[Content_Types].xml", ct)
        put("_rels/.rels", relationships([
            ("rId1", relbase + "officeDocument", "ppt/presentation.xml"),
            ("rId2", "http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties", "docProps/core.xml"),
            ("rId3", relbase + "extended-properties", "docProps/app.xml"),
        ]))
        put("docProps/core.xml", core)
        put("docProps/app.xml", app)
        put("ppt/presentation.xml", presentation)
        put("ppt/_rels/presentation.xml.rels", relationships(pres_rels))
        put("ppt/slideMasters/slideMaster1.xml", master)
        put("ppt/slideMasters/_rels/slideMaster1.xml.rels", relationships([
            ("rId1", relbase + "slideLayout", "../slideLayouts/slideLayout1.xml"),
            ("rId2", relbase + "theme", "../theme/theme1.xml"),
        ]))
        put("ppt/slideLayouts/slideLayout1.xml", layout)
        put("ppt/slideLayouts/_rels/slideLayout1.xml.rels", relationships([
            ("rId1", relbase + "slideMaster", "../slideMasters/slideMaster1.xml"),
        ]))
        put("ppt/theme/theme1.xml", theme)
        for i, slide in enumerate(slides, 1):
            put(f"ppt/slides/slide{i}.xml", slide.xml())
            links = [("rId1", relbase + "slideLayout", "../slideLayouts/slideLayout1.xml")]
            links += [(rid, relbase + "image", f"../media/{name}")
                      for rid, name in slide.images]
            put(f"ppt/slides/_rels/slide{i}.xml.rels", relationships(links))
        for name in media_names:
            put(f"ppt/media/{name}", (IMAGES / name).read_bytes())


def verify(slides: list[Slide]) -> None:
    import xml.etree.ElementTree as ET
    import posixpath

    expected = {f"ppt/media/{p.name}" for p in IMAGES.glob("*.png")}
    with zipfile.ZipFile(OUTPUT) as z:
        assert z.testzip() is None
        actual = {n for n in z.namelist() if n.startswith("ppt/media/")}
        assert actual == expected, (expected - actual, actual - expected)
        assert len([n for n in z.namelist() if re.fullmatch(r"ppt/slides/slide\d+\.xml", n)]) == len(slides)
        for name in z.namelist():
            if name.endswith(".xml") or name.endswith(".rels"):
                ET.fromstring(z.read(name))
            if name.endswith(".rels"):
                if name == "_rels/.rels":
                    base = ""
                else:
                    folder, relfile = name.rsplit("/_rels/", 1)
                    base = posixpath.dirname(f"{folder}/{relfile[:-5]}")
                for link in ET.fromstring(z.read(name)):
                    target = link.attrib["Target"]
                    if not target.startswith("http"):
                        resolved = posixpath.normpath(posixpath.join(base, target))
                        assert resolved in z.namelist(), (name, target, resolved)
        for name in expected:
            source = IMAGES / Path(name).name
            assert hashlib.sha256(z.read(name)).digest() == hashlib.sha256(source.read_bytes()).digest()


if __name__ == "__main__":
    deck = build_slides()
    create_pptx(deck)
    verify(deck)
    print(f"Created {OUTPUT} ({len(deck)} slides, {len(list(IMAGES.glob('*.png')))} original image files)")
