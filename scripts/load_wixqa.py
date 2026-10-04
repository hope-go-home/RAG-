"""
WixQA 语料转换脚本
==================
把 HuggingFace 的 Wix/WixQA 数据集转成本项目可入库的目录结构：

  data/wixqa_corpus/<article_type>/<safe_title>__<id8>.md   # 每篇文章一个文件，首行 # 标题
  data/wixqa_corpus/_meta.json                             # 文件名 -> {id,url,title,article_type}
  SmartQuery/evaluation/golden_wixqa.json                  # question + gold_sources（文章文件名）

用法：
  python scripts/load_wixqa.py                    # 全量（语料 + expertwritten 问答）
  python scripts/load_wixqa.py --limit 200        # 只转前 200 篇（快速验证）
  python scripts/load_wixqa.py --qa synthetic     # 用 synthetic 问答（默认 expertwritten）
  python scripts/load_wixqa.py --qa both          # 两套问答都生成
"""

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data" / "wixqa_corpus"
GOLDEN = ROOT / "SmartQuery" / "evaluation" / "golden_wixqa.json"
DATASET = "Wix/WixQA"


def _as_dataset(ds):
    """load_dataset 可能返回 Dataset 或 DatasetDict，统一取单个 split"""
    from datasets import DatasetDict
    if isinstance(ds, DatasetDict):
        return ds[list(ds.keys())[0]]
    return ds


def safe_name(title: str, article_id: str) -> str:
    base = re.sub(r"[^\w\u4e00-\u9fff-]+", "_", title or "").strip("_")[:80] or "article"
    return f"{base}__{article_id[:8]}.md"


def load_qa(config: str):
    from datasets import load_dataset
    return _as_dataset(load_dataset(DATASET, config))


def main() -> None:
    ap = argparse.ArgumentParser(description="WixQA 语料转换")
    ap.add_argument("--limit", type=int, default=0, help="只转换前 N 篇（0=全部）")
    ap.add_argument("--qa", default="expertwritten",
                    choices=["expertwritten", "synthetic", "both", "none"],
                    help="生成哪套问答标注")
    args = ap.parse_args()

    from datasets import load_dataset

    print("下载语料配置 wix_kb_corpus ...")
    kb = _as_dataset(load_dataset(DATASET, "wix_kb_corpus"))

    rows = list(kb)
    if args.limit:
        rows = rows[: args.limit]
    print(f"待转换文章数：{len(rows)}")

    meta: dict[str, dict] = {}
    id_to_file: dict[str, str] = {}
    by_type: dict[str, int] = {}

    for row in rows:
        aid = row["id"]
        article_type = row.get("article_type") or "article"
        title = (row.get("title") or "").strip()
        contents = (row.get("contents") or "").strip()
        if not contents:
            continue
        fname = safe_name(title, aid)
        type_dir = OUT_DIR / article_type
        type_dir.mkdir(parents=True, exist_ok=True)
        # 首行写 Markdown 标题，便于分块器识别并作为父块上下文
        (type_dir / fname).write_text(f"# {title}\n\n{contents}\n", encoding="utf-8")
        meta[fname] = {"id": aid, "url": row.get("url", ""),
                       "title": title, "article_type": article_type}
        id_to_file[aid] = fname
        by_type[article_type] = by_type.get(article_type, 0) + 1

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print("文章类型分布：")
    for t, n in sorted(by_type.items(), key=lambda x: -x[1]):
        print(f"  {t}: {n}")
    print(f"已写入 {len(meta)} 篇到 {OUT_DIR}")

    if args.qa == "none":
        return

    configs = []
    if args.qa in ("expertwritten", "both"):
        configs.append("wixqa_expertwritten")
    if args.qa in ("synthetic", "both"):
        configs.append("wixqa_synthetic")

    golden = []
    for cfg in configs:
        print(f"下载问答配置 {cfg} ...")
        qa = load_qa(cfg)
        total, kept = 0, 0
        for row in qa:
            total += 1
            gold_files = [id_to_file[a] for a in row.get("article_ids", []) if a in id_to_file]
            if not gold_files:
                continue
            kept += 1
            doc_type = meta[gold_files[0]]["article_type"] if gold_files else ""
            golden.append({
                "level": cfg,
                "doc_type": doc_type,
                "question": row["question"],
                "gold_sources": gold_files,
                "reference": row.get("answer", ""),
            })
        print(f"  {cfg}: {total} 条 -> 有效 {kept} 条")

    GOLDEN.parent.mkdir(parents=True, exist_ok=True)
    GOLDEN.write_text(json.dumps(golden, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"评测标注已写入 {GOLDEN}（共 {len(golden)} 题）")


if __name__ == "__main__":
    main()
