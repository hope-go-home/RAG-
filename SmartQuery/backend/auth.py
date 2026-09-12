"""认证与授权：密码哈希、JWT 签发/校验、FastAPI 依赖

设计要点：
- 部门与角色写入 JWT，由服务端解析，前端无法伪造
- `get_current_user`：解析 Bearer token → 返回用户信息
- `require_admin`：仅管理员可访问
"""

from datetime import datetime, timedelta, timezone

import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from passlib.context import CryptContext

from SmartQuery.backend.config import JWT_SECRET, JWT_EXPIRE_MINUTES
from SmartQuery.backend.logger import get_logger

logger = get_logger(__name__)

ALGORITHM = "HS256"

_pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
_bearer = HTTPBearer(auto_error=False)


def hash_password(password: str) -> str:
    return _pwd_context.hash(password)


def verify_password(password: str, password_hash: str) -> bool:
    try:
        return _pwd_context.verify(password, password_hash)
    except Exception:
        return False


def create_access_token(user: dict) -> str:
    """user: {id, username, department, role}"""
    now = datetime.now(timezone.utc)
    payload = {
        "sub": str(user["id"]),
        "username": user["username"],
        "department": user["department"],
        "role": user.get("role", "user"),
        "iat": now,
        "exp": now + timedelta(minutes=JWT_EXPIRE_MINUTES),
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=ALGORITHM)


def decode_token(token: str) -> dict:
    return jwt.decode(token, JWT_SECRET, algorithms=[ALGORITHM])


def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer),
) -> dict:
    """从 Authorization: Bearer <token> 解析当前用户；失败返回 401"""
    if credentials is None or not credentials.credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="未登录或缺少 Token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    try:
        payload = decode_token(credentials.credentials)
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Token 已过期")
    except jwt.PyJWTError as e:
        logger.warning("token decode failed: %s", e)
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Token 无效")
    return {
        "id": int(payload["sub"]),
        "username": payload.get("username", ""),
        "department": payload.get("department", "公共"),
        "role": payload.get("role", "user"),
    }


def require_admin(user: dict = Depends(get_current_user)) -> dict:
    """要求管理员角色，否则 403"""
    if user.get("role") != "admin":
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="需要管理员权限")
    return user
