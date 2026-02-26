"""
数据库连接管理 - SQLite 数据库
"""

import sqlite3
import logging
from pathlib import Path
from contextlib import contextmanager
from typing import Optional, List, Dict, Any

from config import config

logger = logging.getLogger(__name__)


class Database:
    """SQLite 数据库管理"""

    def __init__(self, db_path: str = None):
        self.db_path = db_path or config.DATABASE_PATH
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_tables()

    @contextmanager
    def get_connection(self):
        """获取数据库连接（上下文管理器）"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception as e:
            conn.rollback()
            logger.error(f"数据库操作失败: {e}")
            raise
        finally:
            conn.close()

    def _init_tables(self):
        """初始化数据库表"""
        with self.get_connection() as conn:
            cursor = conn.cursor()

            # 搜索结果表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS search_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    asin TEXT NOT NULL,
                    brand TEXT,
                    search_keyword TEXT,
                    search_rank INTEGER,
                    title TEXT,
                    price TEXT,
                    original_price TEXT,
                    rating REAL,
                    rating_count INTEGER,
                    is_sponsored INTEGER DEFAULT 0,
                    is_amazon_choice INTEGER DEFAULT 0,
                    is_best_seller INTEGER DEFAULT 0,
                    url TEXT,
                    image_url TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(asin, search_keyword, brand)
                )
            ''')

            # 商品详情表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS products (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    asin TEXT NOT NULL,
                    brand TEXT,
                    title TEXT,
                    price TEXT,
                    original_price TEXT,
                    rating REAL,
                    rating_count INTEGER,
                    rating_5_star INTEGER,
                    rating_4_star INTEGER,
                    rating_3_star INTEGER,
                    rating_2_star INTEGER,
                    rating_1_star INTEGER,
                    bought_count TEXT,           -- ⭐ 新增：销量原始文本
                    bought_count_number INTEGER DEFAULT 0,  -- ⭐ 新增：销量数字
                    bsr_rank INTEGER,
                    bsr_category TEXT,
                    sub_category_ranks TEXT,
                    bullet_points TEXT,
                    description TEXT,
                    images TEXT,
                    variants TEXT,
                    seller TEXT,
                    is_fba INTEGER DEFAULT 0,
                    has_aplus INTEGER DEFAULT 0,
                    first_available TEXT,
                    availability TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(asin, brand)
                )
            ''')

            # 评论表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS reviews (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    asin TEXT NOT NULL,
                    brand TEXT,
                    review_id TEXT NOT NULL,
                    title TEXT,
                    content TEXT,
                    rating INTEGER,
                    date TEXT,
                    verified_purchase INTEGER DEFAULT 0,
                    helpful_votes INTEGER DEFAULT 0,
                    author TEXT,
                    variant TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(asin, review_id, brand)
                )
            ''')

            # 抓取任务表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS scrape_tasks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    task_type TEXT NOT NULL,
                    target TEXT NOT NULL,
                    brand TEXT,
                    status TEXT DEFAULT 'pending',
                    retry_count INTEGER DEFAULT 0,
                    error_message TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')

            # 创建索引
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_search_asin ON search_results(asin)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_search_keyword ON search_results(search_keyword)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_products_asin ON products(asin)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_products_brand ON products(brand)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_reviews_asin ON reviews(asin)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_reviews_rating ON reviews(rating)')

            logger.info("数据库表初始化完成")

    def execute(self, sql: str, params: tuple = ()) -> List[Dict]:
        """执行SQL查询"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(sql, params)
            rows = cursor.fetchall()
            return [dict(row) for row in rows]

    def execute_many(self, sql: str, params_list: List[tuple]):
        """批量执行SQL"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.executemany(sql, params_list)
            return cursor.rowcount

    def get_table_count(self, table: str) -> int:
        """获取表记录数"""
        result = self.execute(f"SELECT COUNT(*) as count FROM {table}")
        return result[0]['count'] if result else 0

    def get_stats(self) -> Dict[str, Any]:
        """获取数据库统计"""
        return {
            'search_results': self.get_table_count('search_results'),
            'products': self.get_table_count('products'),
            'reviews': self.get_table_count('reviews'),
            'scrape_tasks': self.get_table_count('scrape_tasks'),
        }


