"""评测回归门禁
============
对比最新评测报告与基线，若任一指标跌破「基线 - 容忍度」则以非 0 退出，
用于 CI / 发布前阻断检索质量劣化。

用法：
  python scripts/eval_gate.py
  python scripts/eval_gate.py --tol 0.05

若无报告文件（例如 CI 未连 Milvus），则跳过并返回 0。
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser(description="检索评测回归门禁")
    ap.add_argument("--report", default=str(ROOT / "SmartQuery/evaluation/report/retrieval_eval.json"))
    ap.add_argument("--baseline", default=str(ROOT / "SmartQuery/evaluation/baseline.json"))
    ap.add_argument("--tol", type=float, default=0.03, help="允许的下降容忍度")
    args = ap.parse_args()

    report = Path(args.report)
    baseline = Path(args.baseline)
    if not report.exists():
        print(f"[skip] 无评测报告：{report}（CI 未跑评测时跳过）")
        return 0
    if not baseline.exists():
        print(f"[skip] 无基线文件：{baseline}")
        return 0

    cur = json.loads(report.read_text(encoding="utf-8")).get("summary", {})
    base = json.loads(baseline.read_text(encoding="utf-8"))

    failed = False
    for strat, metrics in base.items():
        for key, bval in metrics.items():
            cval = cur.get(strat, {}).get(key)
            if cval is None:
                continue
            if cval < bval - args.tol:
                print(f"[FAIL] {strat}.{key}: 当前 {cval} < 基线 {bval}（容忍 {args.tol}）")
                failed = True
            else:
                print(f"[ok]   {strat}.{key}: {cval} >= {bval - args.tol}")
    print("评测门禁：" + ("未通过" if failed else "通过"))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
