"""SmartQuery 包入口。

离线模式必须在任何会导入 huggingface_hub 的模块之前决定：
langchain_community / transformers / FlagEmbedding 都会在导入时读取
HF_HUB_OFFLINE 并缓存为常量，之后再设置环境变量将不生效。
放在包 __init__ 里可保证先于所有子模块执行。
"""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

_hf_home = os.environ.get("HF_HOME", "")
_hub = Path(_hf_home) / "hub" if _hf_home else None
if _hub and (_hub / "models--BAAI--bge-m3").exists():
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
