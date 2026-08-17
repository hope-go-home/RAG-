"""统一日志配置：控制台 + 按天滚动的文件日志，供全项目复用

用法：
    from SmartQuery.backend.logger import get_logger
    logger = get_logger(__name__)
    logger.info("xxx")
"""
import logging
import os
import sys
from logging.handlers import TimedRotatingFileHandler

LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "logs")
os.makedirs(LOG_DIR, exist_ok=True)

FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

_configured = False


def _init_root():
    """只初始化一次根 logger，子 logger 直接继承"""
    global _configured
    if _configured:
        return
    _configured = True

    root = logging.getLogger()
    root.setLevel(logging.INFO)
    if root.handlers:
        return

    # 控制台输出（stderr，避免污染 stdout 的流式输出）
    console = logging.StreamHandler(sys.stderr)
    console.setFormatter(logging.Formatter(FORMAT, DATE_FORMAT))
    root.addHandler(console)

    # 文件输出：logs/app.log，按天滚动，保留 30 天
    file_handler = TimedRotatingFileHandler(
        os.path.join(LOG_DIR, "app.log"),
        when="midnight",
        backupCount=30,
        encoding="utf-8",
    )
    file_handler.setFormatter(logging.Formatter(FORMAT, DATE_FORMAT))
    root.addHandler(file_handler)


def get_logger(name: str = "smartquery") -> logging.Logger:
    _init_root()
    return logging.getLogger(name)