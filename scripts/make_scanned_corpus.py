"""
生成「扫描件」语料，用于 OCR 能力对比
====================================
把若干源文档渲染成「图片型 PDF」（无文本层），模拟企业里的纸质扫描件。

对比实验（scripts/ocr_compare.py）：
  - 无 OCR：PyMuPDF 直接解析 → 提取不到文字（0 字）
  - 有 OCR：qwen-vl-max 逐页识别 → 可提取文字并入库

用法：
  python scripts/make_scanned_corpus.py
输出：
  data/corpus_scanned/*.pdf
"""

import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "data" / "corpus"
OUT = ROOT / "data" / "corpus_scanned"

SOURCES = [
    ("规章制度", "考勤管理制度_2024版.md"),
    ("规章制度", "差旅费标准_海外地区.md"),
    ("规章制度", "事假管理规定.md"),
    ("规章制度", "差旅费报销规范.md"),
    ("技术文档", "用户中心API接入文档.md"),
    ("操作流程SOP", "服务器巡检SOP.md"),
    ("员工手册", "员工手册_总部.md"),
]

LINES_PER_PAGE = 26
LINE_H = 40
FONT_SIZE = 24


def find_font() -> str:
    for f in ["C:/Windows/Fonts/msyh.ttc", "C:/Windows/Fonts/simsun.ttc",
              "C:/Windows/Fonts/simhei.ttf"]:
        if Path(f).exists():
            return f
    raise SystemExit("找不到中文字体（msyh.ttc / simsun.ttc / simhei.ttf）")


def render_pdf(text: str, out_path: Path, font) -> int:
    """把文本渲染成多页图片型 PDF，返回页数"""
    lines = [ln for ln in text.split("\n") if ln.strip()]
    pages = [lines[i:i + LINES_PER_PAGE] for i in range(0, len(lines), LINES_PER_PAGE)]
    images = []
    for page_lines in pages:
        W, H = 1000, 40 + LINE_H * len(page_lines)
        img = Image.new("RGB", (W, H), "white")
        draw = ImageDraw.Draw(img)
        y = 20
        for ln in page_lines:
            draw.text((24, y), ln, fill="black", font=font)
            y += LINE_H
        images.append(img)
    if not images:
        return 0
    images[0].save(str(out_path), "PDF", resolution=150,
                   save_all=True, append_images=images[1:])
    return len(images)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    font = ImageFont.truetype(find_font(), FONT_SIZE)
    count = 0
    for doc_type, name in SOURCES:
        src = CORPUS / doc_type / name
        if not src.exists():
            print(f"  跳过（不存在）：{doc_type}/{name}")
            continue
        text = src.read_text(encoding="utf-8")
        out = OUT / f"{src.stem}_扫描件.pdf"
        pages = render_pdf(text, out, font)
        print(f"  生成：{out.name}（{pages} 页）")
        count += 1
    print(f"\n完成：{count} 个扫描件 -> {OUT}")


if __name__ == "__main__":
    main()
