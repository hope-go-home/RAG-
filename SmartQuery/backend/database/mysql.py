from SmartQuery.backend.config import MYSQL_HOST,MYSQL_PORT,MYSQL_USER,MYSQL_PASSWORD,MYSQL_DATABASE
from SmartQuery.backend.logger import get_logger
from sqlalchemy import create_engine,Column,Integer,String,Text,DateTime,func
from sqlalchemy.orm import sessionmaker,declarative_base
#sessionmaker 用于生成 Session 类（或会话工厂）。Session 是 ORM 中与数据库进行交互的“工作单元”，负责管理对象的持久化操作（增删改查）、事务边界等。
from datetime import datetime

logger = get_logger(__name__)

DATABASE_URL = f"mysql+pymysql://{MYSQL_USER}:{MYSQL_PASSWORD}@{MYSQL_HOST}:{MYSQL_PORT}/{MYSQL_DATABASE}?charset=utf8mb4"
#构造一个符合 SQLAlchemy 规范的数据库连接 URL 字符串。
engine = create_engine(DATABASE_URL)
#调用 create_engine 函数，传入构造好的数据库 URL，生成一个数据库引擎对象 engine
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
#使用 sessionmaker 创建一个会话工厂，并绑定到之前创建的 engine   autocommit=False：禁用自动提交  autoflush=False：禁用自动刷新
#bind=engine：将该会话工厂与前面创建的 engine 绑定，使得通过 SessionLocal() 生成的每个 Session 都会使用该引擎连接数据库。

Base = declarative_base()

class ChatRecord(Base):
    __tablename__ = "chat_records"

    id = Column(Integer,primary_key=True,autoincrement=True)
    session_id = Column(String(64),index=True,nullable=False)
    question = Column(Text,nullable=False)
    answer = Column(Text,nullable=False)
    created_at = Column(DateTime,default=datetime.utcnow)

def init_db():
    Base.metadata.create_all(bind=engine)
    logger.info("mysql tables ready")

def save_record(session_id: str, question: str, answer: str):
    db = SessionLocal()
    record = ChatRecord(session_id=session_id, question=question, answer=answer)
    db.add(record)
    db.commit()
    db.close()


def get_history(session_id: str) -> list[dict]:
    db = SessionLocal() #调用之前创建的会话工厂 SessionLocal()，生成一个新的数据库会话对象，赋值给变量 db。该会话用于执行数据库操作（查询、提交、关闭等）。
    records = (
        db.query(ChatRecord) #从数据库的 ChatRecord 表（对应 caht_records 表）发起查询
        .filter(ChatRecord.session_id == session_id)
        .order_by(ChatRecord.created_at)  #按 created_at 列进行升序排序（默认升序）
        .all()
    )
    db.close()
    return [
        {"question": r.question, "answer": r.answer, "created_at": str(r.created_at)}
        for r in records
    ]


def list_sessions(limit: int = 50) -> list[dict]:
    """按会话聚合，返回每个会话的首问、末次时间、问答条数（用于历史会话列表）"""
    db = SessionLocal()
    try:
        rows = (
            db.query(
                ChatRecord.session_id,
                func.min(ChatRecord.id).label("first_id"),
                func.max(ChatRecord.created_at).label("last_time"),
                func.count(ChatRecord.id).label("count"),
            )
            .group_by(ChatRecord.session_id)
            .order_by(func.max(ChatRecord.created_at).desc())
            .limit(limit)
            .all()
        )
        result = []
        for r in rows:
            first_question = (
                db.query(ChatRecord.question)
                .filter(ChatRecord.id == r.first_id)
                .scalar()
            )
            result.append({
                "session_id": r.session_id,
                "first_question": first_question or "",
                "last_time": str(r.last_time),
                "count": r.count,
            })
        return result
    finally:
        db.close()


# ---------------- 文件注册表（重复入库检测）---------------- #

class UploadedFile(Base):
    """已入库文件登记表：以文件内容 SHA-256 哈希为唯一标识，防止重复入库"""
    __tablename__ = "uploaded_files"

    id = Column(Integer, primary_key=True, autoincrement=True)
    file_hash = Column(String(64), unique=True, index=True, nullable=False)  # 内容哈希
    filename = Column(String(255), nullable=False)
    partition = Column(String(16), nullable=False)          # 入库分区（pdf/docx/txt/md）
    chunk_count = Column(Integer, nullable=False, default=0)  # 入库块数
    created_at = Column(DateTime, default=datetime.utcnow)


def find_file_by_hash(file_hash: str) -> dict | None:
    """按内容哈希查文件，已入库返回记录，否则返回 None"""
    db = SessionLocal()
    try:
        row = db.query(UploadedFile).filter(UploadedFile.file_hash == file_hash).first()
        if row is None:
            return None
        return {
            "file_hash": row.file_hash,
            "filename": row.filename,
            "partition": row.partition,
            "chunk_count": row.chunk_count,
            "created_at": str(row.created_at),
        }
    finally:
        db.close()


def save_file_record(file_hash: str, filename: str, partition: str, chunk_count: int):
    """记录一条已入库文件"""
    db = SessionLocal()
    try:
        db.add(UploadedFile(file_hash=file_hash, filename=filename,
                            partition=partition, chunk_count=chunk_count))
        db.commit()
    finally:
        db.close()