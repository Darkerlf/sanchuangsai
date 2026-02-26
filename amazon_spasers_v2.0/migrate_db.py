"""
数据库迁移 - 添加销量字段
"""

import sqlite3
from pathlib import Path


def migrate():
    """添加销量字段到已有数据库"""
    db_path = Path(__file__).parent / 'data' / 'amazon_data.db'

    if not db_path.exists():
        print("数据库不存在，无需迁移")
        return

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # 检查字段是否已存在
    cursor.execute("PRAGMA table_info(products)")
    columns = [col[1] for col in cursor.fetchall()]

    if 'bought_count' not in columns:
        print("添加 bought_count 字段...")
        cursor.execute("ALTER TABLE products ADD COLUMN bought_count TEXT DEFAULT ''")

    if 'bought_count_number' not in columns:
        print("添加 bought_count_number 字段...")
        cursor.execute("ALTER TABLE products ADD COLUMN bought_count_number INTEGER DEFAULT 0")

    conn.commit()
    conn.close()

    print("✅ 数据库迁移完成!")


if __name__ == '__main__':
    migrate()
