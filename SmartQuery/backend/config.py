from dotenv import load_dotenv
import os

load_dotenv()

#千问聊天模型
QWEN_API_KEY = os.getenv("QWEN_API_KEY")
QWEN_BASE_URL = os.getenv("QWEN_BASE_URL")
QWEN_MODEL = os.getenv("QWEN_MODEL")

#嵌入模型
QWEN_EMBEDDING_MODEL = os.getenv("QWEN_EMBEDDING_MODEL")

# 视觉/OCR 模型（扫描件、图片识别）
QWEN_VL_OCR_MODEL = os.getenv("QWEN_VL_OCR_MODEL", "qwen-vl-max")

# 认证（JWT）
JWT_SECRET = os.getenv("JWT_SECRET", "dev-secret-change-me")
JWT_EXPIRE_MINUTES = int(os.getenv("JWT_EXPIRE_MINUTES", "720"))
DEFAULT_ADMIN_USER = os.getenv("DEFAULT_ADMIN_USER", "admin")
DEFAULT_ADMIN_PASSWORD = os.getenv("DEFAULT_ADMIN_PASSWORD", "admin123")
DEFAULT_ADMIN_DEPARTMENT = os.getenv("DEFAULT_ADMIN_DEPARTMENT", "公共")

MYSQL_HOST = os.getenv("MYSQL_HOST")
MYSQL_PORT = int(os.getenv("MYSQL_PORT"))
MYSQL_USER = os.getenv("MYSQL_USER")
MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD")
MYSQL_DATABASE = os.getenv("MYSQL_DATABASE")

MILVUS_HOST = os.getenv("MILVUS_HOST")
MILVUS_PORT = int(os.getenv("MILVUS_PORT"))

# HF_HOME 会被 load_dotenv() 读到环境变量，SentenceTransformer 导入时自动识别