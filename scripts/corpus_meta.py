"""语料元数据：文件 → 所属部门。

部门映射由 scripts/build_corpus.py 生成到 data/corpus/_departments.json。
用于权限隔离：提问时只检索「当前用户部门 + 公共」的文档。
未登记的文件默认归为「公共」。
"""

import json
from pathlib import Path

_MAP_FILE = Path(__file__).resolve().parents[1] / "data" / "corpus" / "_departments.json"

DEPARTMENT_MAP: dict[str, str] = {}
if _MAP_FILE.exists():
    try:
        DEPARTMENT_MAP = json.loads(_MAP_FILE.read_text(encoding="utf-8"))
    except Exception:
        DEPARTMENT_MAP = {}


def department_of(filename: str) -> str:
    """按文件名查所属部门，未登记默认「公共」"""
    return DEPARTMENT_MAP.get(filename, "公共")
