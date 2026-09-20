from SmartQuery.backend.config import MYSQL_HOST,MYSQL_PORT,MYSQL_USER,MYSQL_PASSWORD,MYSQL_DATABASE
from SmartQuery.backend.logger import get_logger
from sqlalchemy import create_engine,Column,Integer,String,Text,DateTime,Boolean,func
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
    user_id = Column(Integer,index=True,nullable=True)   # 归属用户（历史会话按用户隔离）
    question = Column(Text,nullable=False)
    answer = Column(Text,nullable=False)
    created_at = Column(DateTime,default=datetime.utcnow)


def _ensure_columns():
    """轻量迁移：为已存在的表补充新增列（create_all 不会修改已存在的表）"""
    from sqlalchemy import inspect, text
    insp = inspect(engine)
    if "chat_records" in insp.get_table_names():
        cols = {c["name"] for c in insp.get_columns("chat_records")}
        if "user_id" not in cols:
            with engine.begin() as conn:
                conn.execute(text("ALTER TABLE chat_records ADD COLUMN user_id INT NULL"))
                conn.execute(text("CREATE INDEX ix_chat_records_user_id ON chat_records (user_id)"))
            logger.info("migrated: chat_records.user_id added")


def init_db():
    Base.metadata.create_all(bind=engine)
    _ensure_columns()
    logger.info("mysql tables ready")
    seed_default_admin()


def save_record(session_id: str, question: str, answer: str, user_id: int | None = None):
    db = SessionLocal()
    try:
        db.add(ChatRecord(session_id=session_id, question=question,
                          answer=answer, user_id=user_id))
        db.commit()
    finally:
        db.close()


def get_history(session_id: str, user_id: int | None = None) -> list[dict]:
    """取某会话的问答记录；user_id 不为空时只取该用户的"""
    db = SessionLocal()
    try:
        q = db.query(ChatRecord).filter(ChatRecord.session_id == session_id)
        if user_id is not None:
            q = q.filter(ChatRecord.user_id == user_id)
        records = q.order_by(ChatRecord.created_at).all()
        return [
            {"question": r.question, "answer": r.answer, "created_at": str(r.created_at)}
            for r in records
        ]
    finally:
        db.close()


def list_sessions(user_id: int | None = None, limit: int = 50) -> list[dict]:
    """按会话聚合，返回首问、末次时间、问答条数；按用户隔离"""
    db = SessionLocal()
    try:
        q = db.query(
            ChatRecord.session_id,
            func.min(ChatRecord.id).label("first_id"),
            func.max(ChatRecord.created_at).label("last_time"),
            func.count(ChatRecord.id).label("count"),
        )
        if user_id is not None:
            q = q.filter(ChatRecord.user_id == user_id)
        rows = (q.group_by(ChatRecord.session_id)
                 .order_by(func.max(ChatRecord.created_at).desc())
                 .limit(limit).all())
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


def delete_session(session_id: str, user_id: int | None = None) -> int:
    """删除会话的全部问答记录，返回删除条数；user_id 不为空时校验归属"""
    db = SessionLocal()
    try:
        q = db.query(ChatRecord).filter(ChatRecord.session_id == session_id)
        if user_id is not None:
            q = q.filter(ChatRecord.user_id == user_id)
        n = q.delete(synchronize_session=False)
        db.commit()
        return n
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
    """记录一条已入库文件（按内容哈希幂等：已存在则更新，避免唯一键冲突）"""
    db = SessionLocal()
    try:
        row = db.query(UploadedFile).filter(UploadedFile.file_hash == file_hash).first()
        if row is None:
            db.add(UploadedFile(file_hash=file_hash, filename=filename,
                                partition=partition, chunk_count=chunk_count))
        else:
            row.filename = filename
            row.partition = partition
            row.chunk_count = chunk_count
        db.commit()
    finally:
        db.close()


# ---------------- 用户与审计（身份权限）---------------- #

class User(Base):
    """系统用户：部门与角色绑定在服务端，前端不可伪造"""
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, autoincrement=True)
    username = Column(String(64), unique=True, index=True, nullable=False)
    password_hash = Column(String(255), nullable=False)
    department = Column(String(64), nullable=False, default="公共")
    role = Column(String(16), nullable=False, default="user")  # admin / user
    is_active = Column(Boolean, nullable=False, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)


class AuditLog(Base):
    """审计日志：登录、提问、上传、删除等敏感操作留痕"""
    __tablename__ = "audit_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, index=True, nullable=True)
    username = Column(String(64), index=True, nullable=True)
    action = Column(String(32), nullable=False)     # login / chat / upload / delete
    resource = Column(String(255), nullable=True)
    detail = Column(Text, nullable=True)
    ip = Column(String(64), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)


def count_users() -> int:
    db = SessionLocal()
    try:
        return db.query(func.count(User.id)).scalar() or 0
    finally:
        db.close()


def create_user(username: str, password: str, department: str = "公共",
                role: str = "user") -> int:
    """创建用户，返回用户 id"""
    from SmartQuery.backend.auth import hash_password
    db = SessionLocal()
    try:
        user = User(username=username, password_hash=hash_password(password),
                    department=department, role=role)
        db.add(user)
        db.commit()
        db.refresh(user)
        return user.id
    finally:
        db.close()


def get_user_by_username(username: str) -> dict | None:
    db = SessionLocal()
    try:
        u = db.query(User).filter(User.username == username).first()
        if u is None:
            return None
        return {
            "id": u.id, "username": u.username, "password_hash": u.password_hash,
            "department": u.department, "role": u.role, "is_active": u.is_active,
        }
    finally:
        db.close()


def list_users(limit: int = 200) -> list[dict]:
    """用户列表（不含密码哈希）"""
    db = SessionLocal()
    try:
        rows = db.query(User).order_by(User.id).limit(limit).all()
        return [{
            "id": u.id, "username": u.username, "department": u.department,
            "role": u.role, "is_active": u.is_active, "created_at": str(u.created_at),
        } for u in rows]
    finally:
        db.close()


def write_audit(user_id: int | None, username: str | None, action: str,
                resource: str = "", detail: str = "", ip: str = ""):
    db = SessionLocal()
    try:
        db.add(AuditLog(user_id=user_id, username=username, action=action,
                        resource=resource, detail=detail[:2000], ip=ip))
        db.commit()
    except Exception as e:
        logger.warning("write_audit failed: %s", e)
    finally:
        db.close()


def list_audit(limit: int = 100) -> list[dict]:
    db = SessionLocal()
    try:
        rows = (db.query(AuditLog).order_by(AuditLog.id.desc()).limit(limit).all())
        return [{
            "id": r.id, "username": r.username, "action": r.action,
            "resource": r.resource, "detail": r.detail, "ip": r.ip,
            "created_at": str(r.created_at),
        } for r in rows]
    finally:
        db.close()


def seed_default_admin():
    """无任何用户时，创建默认管理员（账号来自 config / .env）"""
    try:
        if count_users() > 0:
            return
        from SmartQuery.backend.config import (
            DEFAULT_ADMIN_USER, DEFAULT_ADMIN_PASSWORD, DEFAULT_ADMIN_DEPARTMENT,
        )
        create_user(DEFAULT_ADMIN_USER, DEFAULT_ADMIN_PASSWORD,
                    DEFAULT_ADMIN_DEPARTMENT, "admin")
        logger.info("seeded default admin user=%s dept=%s",
                    DEFAULT_ADMIN_USER, DEFAULT_ADMIN_DEPARTMENT)
    except Exception as e:
        logger.warning("seed_default_admin failed: %s", e)


# ---------------- 文档生命周期（增量更新 / 软删除）---------------- #

class Document(Base):
    """文档登记表：以 source(文件名) 为唯一键，支持增量更新与软删除"""
    __tablename__ = "documents"

    id = Column(Integer, primary_key=True, autoincrement=True)
    source = Column(String(255), unique=True, index=True, nullable=False)  # 文件名
    stored_path = Column(String(512), nullable=True)   # 落盘路径（用于重建索引）
    file_hash = Column(String(64), nullable=True)
    doc_type = Column(String(64), nullable=False, default="员工手册")
    department = Column(String(64), nullable=False, default="公共")
    chunk_count = Column(Integer, nullable=False, default=0)
    version = Column(Integer, nullable=False, default=1)
    status = Column(String(16), nullable=False, default="active")  # active / deleted
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    created_at = Column(DateTime, default=datetime.utcnow)


def _doc_to_dict(d: Document) -> dict:
    return {
        "id": d.id, "source": d.source, "stored_path": d.stored_path,
        "file_hash": d.file_hash, "doc_type": d.doc_type, "department": d.department,
        "chunk_count": d.chunk_count, "version": d.version, "status": d.status,
        "updated_at": str(d.updated_at), "created_at": str(d.created_at),
    }


def get_document_by_source(source: str) -> dict | None:
    db = SessionLocal()
    try:
        d = db.query(Document).filter(Document.source == source).first()
        return _doc_to_dict(d) if d else None
    finally:
        db.close()


def get_document_by_id(doc_id: int) -> dict | None:
    db = SessionLocal()
    try:
        d = db.query(Document).filter(Document.id == doc_id).first()
        return _doc_to_dict(d) if d else None
    finally:
        db.close()


def list_documents(limit: int = 200) -> list[dict]:
    db = SessionLocal()
    try:
        rows = (db.query(Document).filter(Document.status == "active")
                .order_by(Document.id.desc()).limit(limit).all())
        return [_doc_to_dict(d) for d in rows]
    finally:
        db.close()


def upsert_document(source: str, stored_path: str, file_hash: str, doc_type: str,
                    department: str, chunk_count: int) -> dict:
    """新建或更新文档记录；更新时 version+1、status 置回 active"""
    db = SessionLocal()
    try:
        d = db.query(Document).filter(Document.source == source).first()
        if d is None:
            d = Document(source=source, stored_path=stored_path, file_hash=file_hash,
                         doc_type=doc_type, department=department,
                         chunk_count=chunk_count, version=1, status="active")
            db.add(d)
        else:
            d.stored_path = stored_path
            d.file_hash = file_hash
            d.doc_type = doc_type
            d.department = department
            d.chunk_count = chunk_count
            d.version = (d.version or 1) + 1
            d.status = "active"
        db.commit()
        db.refresh(d)
        return _doc_to_dict(d)
    finally:
        db.close()


def soft_delete_document(doc_id: int) -> dict | None:
    """软删除：status 置 deleted，返回被删文档信息"""
    db = SessionLocal()
    try:
        d = db.query(Document).filter(Document.id == doc_id).first()
        if d is None:
            return None
        info = _doc_to_dict(d)
        d.status = "deleted"
        db.commit()
        return info
    finally:
        db.close()