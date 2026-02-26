# reset_db.py
import argparse
import sqlite3
from pathlib import Path

DEFAULT_DB = Path("data") / "amazon_data.db"
TABLES_ORDERED = ["reviews", "products", "search_results", "scrape_tasks"]  # 先删子表再删父表更安全


def table_exists(conn: sqlite3.Connection, table: str) -> bool:
    cur = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=? LIMIT 1", (table,)
    )
    return cur.fetchone() is not None


def count_rows(conn: sqlite3.Connection, table: str) -> int:
    cur = conn.execute(f"SELECT COUNT(1) FROM {table}")
    return int(cur.fetchone()[0])


def reset_db(db_path: Path, yes: bool, reset_autoinc: bool, vacuum: bool) -> None:
    if not db_path.exists():
        raise FileNotFoundError(f"数据库文件不存在：{db_path.resolve()}")

    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute("PRAGMA foreign_keys = ON;")

        existing = [t for t in TABLES_ORDERED if table_exists(conn, t)]
        if not existing:
            print("未找到目标表（reviews/products/search_results/scrape_tasks），无需清空。")
            return

        print(f"数据库文件：{db_path.resolve()}")
        print("即将清空以下表的数据（保留表结构）：")
        for t in existing:
            print(f"  - {t}: {count_rows(conn, t)} 行")

        if not yes:
            confirm = input("\n确认清空？输入 YES 继续：").strip()
            if confirm != "YES":
                print("已取消。")
                return

        with conn:
            for t in existing:
                conn.execute(f"DELETE FROM {t};")

            # 重置自增（仅当表用 AUTOINCREMENT 且 sqlite_sequence 存在时）
            if reset_autoinc and table_exists(conn, "sqlite_sequence"):
                placeholders = ",".join("?" for _ in existing)
                conn.execute(
                    f"DELETE FROM sqlite_sequence WHERE name IN ({placeholders});",
                    existing,
                )

        if vacuum:
            conn.execute("VACUUM;")

        print("\n✅ 清空完成。当前行数：")
        for t in existing:
            print(f"  - {t}: {count_rows(conn, t)} 行")

    finally:
        conn.close()


def main():
    parser = argparse.ArgumentParser(description="清空 Amazon 爬虫 SQLite 数据（保留表结构）")
    parser.add_argument("--db", type=str, default=str(DEFAULT_DB), help="数据库路径（默认: data/amazon_data.db）")
    parser.add_argument("--yes", action="store_true", help="跳过确认，直接清空")
    parser.add_argument("--reset-autoinc", action="store_true", help="重置自增（sqlite_sequence）")
    parser.add_argument("--vacuum", action="store_true", help="清空后执行 VACUUM 回收空间")

    args = parser.parse_args()
    reset_db(Path(args.db), yes=args.yes, reset_autoinc=args.reset_autoinc, vacuum=args.vacuum)


if __name__ == "__main__":
    main()
