"""
OCR 能力对比实验
================
对 data/corpus_scanned/ 下的扫描件 PDF 做对比：
  - 无 OCR：PyMuPDF 直接解析（提取不到文字）
  - 有 OCR：qwen-vl-max 逐页识别

比较两者能提取的文本量，以及关键短语是否被正确识别。

用法：
  python scripts/make_scanned_corpus.py   # 先生成扫描件
  python scripts/ocr_compare.py           # 再对比
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from SmartQuery.rag.ingest import load_file

SCANNED = Path(__file__).resolve().parents[1] / "data" / "corpus_scanned"

# 每个扫描件期望被识别出的关键短语
EXPECTED = {
    "考勤管理制度_2024版_扫描件": "年休假7天",
    "差旅费标准_海外地区_扫描件": "1200",
    "事假管理规定_扫描件": "每月累计不得超过3天",
    "差旅费报销规范_扫描件": "10个工作日",
    "用户中心API接入文档_扫描件": "1000",
    "服务器巡检SOP_扫描件": "每日一次",
    "员工手册_总部_扫描件": "9:00-18:00",
}


def pymupdf_text(path: Path) -> str:
    """无 OCR：直接解析 PDF 文本层"""
    import fitz
    doc = fitz.open(str(path))
    text = "".join(page.get_text() for page in doc)
    doc.close()
    return text


def main() -> None:
    pdfs = sorted(SCANNED.glob("*.pdf"))
    if not pdfs:
        print(f"未找到扫描件，请先运行 scripts/make_scanned_corpus.py（{SCANNED}）")
        return

    print(f"{'扫描件':<34}{'无OCR字数':>10}{'有OCR字数':>10}{'关键短语':>10}")
    print("-" * 64)
    total_no, total_ocr, hit = 0, 0, 0
    for p in pdfs:
        raw = pymupdf_text(p).strip()
        try:
            docs = load_file(str(p), doc_type="规章制度")
            ocr = "".join(d.page_content for d in docs)
        except Exception as e:
            ocr = ""
            print(f"  OCR 失败 {p.name}: {e}")
        key = EXPECTED.get(p.stem, "")
        ok = (key in ocr) if key else None
        if ok:
            hit += 1
        total_no += len(raw)
        total_ocr += len(ocr)
        mark = "OK" if ok else ("MISS" if ok is False else "-")
        print(f"{p.stem:<34}{len(raw):>10}{len(ocr):>10}{mark:>10}")

    print("-" * 64)
    print(f"合计：无 OCR {total_no} 字 / 有 OCR {total_ocr} 字；关键短语命中 {hit}/{len(pdfs)}")


if __name__ == "__main__":
    main()
