"""
创建系统用户
============
用法：
  python scripts/create_user.py --username analyst --password pass123 --role user
  python scripts/create_user.py --username admin2 --password admin123 --role admin
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from SmartQuery.backend.database.mysql import init_db, create_user, get_user_by_username


def main() -> None:
    ap = argparse.ArgumentParser(description="创建系统用户")
    ap.add_argument("--username", required=True)
    ap.add_argument("--password", required=True)
    ap.add_argument("--role", default="user", choices=["admin", "user"])
    args = ap.parse_args()

    init_db()
    if get_user_by_username(args.username):
        print(f"用户已存在：{args.username}")
        return
    uid = create_user(args.username, args.password, args.role)
    print(f"创建成功：id={uid} username={args.username} role={args.role}")


if __name__ == "__main__":
    main()
