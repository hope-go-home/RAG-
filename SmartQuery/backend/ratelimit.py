"""简易限流：进程内固定窗口计数（无 Redis 时的兜底方案）。

生产多副本环境应换 Redis（见 README「企业级能力」）。此处保证单实例可用。
"""
import threading
import time


class FixedWindowLimiter:
    def __init__(self, limit: int, window: int):
        self.limit = limit
        self.window = window
        self._buckets: dict[str, tuple[float, int]] = {}
        self._lock = threading.Lock()

    def allow(self, key: str) -> bool:
        now = time.time()
        with self._lock:
            start, count = self._buckets.get(key, (now, 0))
            if now - start >= self.window:
                start, count = now, 0
            count += 1
            self._buckets[key] = (start, count)
            return count <= self.limit


# 各接口限流器（可按需调）
chat_limiter = FixedWindowLimiter(limit=30, window=60)     # 30 次/分/用户
login_limiter = FixedWindowLimiter(limit=20, window=60)    # 20 次/分/IP
upload_limiter = FixedWindowLimiter(limit=20, window=60)   # 20 次/分/用户
feedback_limiter = FixedWindowLimiter(limit=60, window=60)  # 60 次/分/用户


def client_ip(request) -> str:
    return request.client.host if request.client else "unknown"
