from dotenv import load_dotenv
import os

load_dotenv()

#千问聊天模型
QWEN_API_KEY = os.getenv("QWEN_API_KEY")
QWEN_BASE_URL = os.getenv("QWEN_BASE_URL")
QWEN_MODEL = os.getenv("QWEN_MODEL")

#嵌入模型
QWEN_EMBEDDING_MODEL = os.getenv("QWEN_EMBEDDING_MODEL")



MYSQL_HOST = os.getenv("MYSQL_HOST")
MYSQL_PORT = int(os.getenv("MYSQL_PORT"))
MYSQL_USER = os.getenv("MYSQL_USER")
MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD")
MYSQL_DATABASE = os.getenv("MYSQL_DATABASE")

MILVUS_HOST = os.getenv("MILVUS_HOST")
MILVUS_PORT = int(os.getenv("MILVUS_PORT"))















# 其他没问题。HF_HOME 会被 load_dotenv() 读到环境变量，SentenceTransformer 导入时自动识别

#  如果环境文件没有配置
# MYSQL_HOST = os.getenv("MYSQL_HOST", "localhost")
# MYSQL_PORT = int(os.getenv("MYSQL_PORT", "3306"))        # 手动转 int
# MYSQL_USER = os.getenv("MYSQL_USER", "root")
# MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD", "123456")
# MYSQL_DATABASE = os.getenv("MYSQL_DATABASE", "smart_query")


# MILVUS_HOST = os.getenv("MILVUS_HOST", "localhost")
# MILVUS_PORT = int(os.getenv("MILVUS_PORT", "19530"))     # 手动转 int

# load_dotenv()
# - 读 .env 文件，把里面的变量加载到 os.environ（环境变量）
# - 不调它，os.getenv() 读不到 .env 里的值
# os.getenv("KEY")
# - 从 os.environ 里取变量值
# - 第二个参数可以设默认值：os.getenv("KEY", "默认值")
# 关系：
# load_dotenv()       # 先把 .env 读到内存
#        ↓
# os.getenv("KEY")    # 才能取到
# 不写 load_dotenv() → 只读系统环境变量，不读 .env 文件。



# from pydantic_settings import BaseSettings

# class Settings(BaseSettings):
#     qwen_api_key: str = ""
#     qwen_base_url: str = ""
#     qwen_model: str = ""

#     mysql_host: str = "localhost"
#     mysql_port: int = 3306
#     mysql_user: str = "root"
#     mysql_password: str = "123456"
#     mysql_database: str = "smart_query"

#     milvus_host: str = "localhost"
#     milvus_port: int = 19530

#     model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


# settings = Settings()
# pydantic-settings 自动把 .env 里的 MYSQL_HOST 映射到字段 mysql_host，不用写任何 os.getenv。其他模块 from config import settings 然后用 settings.mysql_host 就行了。