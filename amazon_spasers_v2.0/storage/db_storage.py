"""
数据库存储 - CRUD 操作
"""

import json
import logging
from typing import List, Optional, Dict, Any
from datetime import datetime

from .database import Database

logger = logging.getLogger(__name__)


class DatabaseStorage:
    """数据库存储操作"""

    def __init__(self, db: Database = None):
        self.db = db or Database()

    # ==================== 搜索结果 ====================

    def save_search_results(self, results: List[Any], brand: str = "") -> int:
        """保存搜索结果"""
        if not results:
            return 0

        sql = '''
            INSERT OR REPLACE INTO search_results 
            (asin, brand, search_keyword, search_rank, title, price, original_price,
             rating, rating_count, is_sponsored, is_amazon_choice, is_best_seller,
             url, image_url)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        '''

        params_list = []
        for r in results:
            data = r.to_dict() if hasattr(r, 'to_dict') else r
            params_list.append((
                data.get('asin'),
                brand or data.get('brand', ''),
                data.get('search_keyword'),
                data.get('search_rank'),
                data.get('title'),
                data.get('price'),
                data.get('original_price'),
                data.get('rating'),
                data.get('rating_count'),
                1 if data.get('is_sponsored') else 0,
                1 if data.get('is_amazon_choice') else 0,
                1 if data.get('is_best_seller') else 0,
                data.get('url'),
                data.get('image_url'),
            ))

        count = self.db.execute_many(sql, params_list)
        logger.info(f"保存搜索结果: {count} 条 (品牌: {brand})")
        return count

    def get_search_results(self, keyword: str = None, limit: int = None) -> List[Dict]:
        """获取搜索结果"""
        sql = "SELECT * FROM search_results"
        params = []

        if keyword:
            sql += " WHERE search_keyword = ?"
            params.append(keyword)

        sql += " ORDER BY search_rank ASC"

        if limit:
            sql += f" LIMIT {limit}"

        return self.db.execute(sql, tuple(params))

    def get_search_asins(self) -> List[str]:
        """获取所有搜索结果的ASIN"""
        results = self.db.execute("SELECT DISTINCT asin FROM search_results")
        return [r['asin'] for r in results]

    # ==================== 商品详情 ====================

    def save_products(self, products: List[Any], brand: str = "") -> int:
        """保存商品详情"""
        if not products:
            return 0

        sql = '''
            INSERT OR REPLACE INTO products 
            (asin, brand, title, price, original_price, rating, rating_count,
             rating_5_star, rating_4_star, rating_3_star, rating_2_star, rating_1_star,
             bought_count, bought_count_number, bsr_rank, bsr_category, sub_category_ranks, bullet_points, description,
             images, variants, seller, is_fba, has_aplus, first_available, availability,
             updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        '''

        params_list = []
        for p in products:
            data = p.to_dict() if hasattr(p, 'to_dict') else p
            rating_dist = data.get('rating_distribution', {})
            product_brand = brand or data.get('brand', '')

            params_list.append((
                data.get('asin'),
                product_brand,
                data.get('title'),
                data.get('price'),
                data.get('original_price'),
                data.get('rating'),
                data.get('rating_count'),
                rating_dist.get('5_star', 0),
                rating_dist.get('4_star', 0),
                rating_dist.get('3_star', 0),
                rating_dist.get('2_star', 0),
                rating_dist.get('1_star', 0),
                data.get('bought_count', ''),
                data.get('bought_count_number', 0),
                data.get('bsr_rank'),
                data.get('bsr_category'),
                json.dumps(data.get('sub_category_ranks', []), ensure_ascii=False),
                json.dumps(data.get('bullet_points', []), ensure_ascii=False),
                data.get('description'),
                json.dumps(data.get('images', []), ensure_ascii=False),
                json.dumps(data.get('variants', []), ensure_ascii=False),
                data.get('seller'),
                1 if data.get('is_fba') else 0,
                1 if data.get('has_aplus') else 0,
                data.get('first_available'),
                data.get('availability'),
                datetime.now().isoformat(),
            ))

        count = self.db.execute_many(sql, params_list)
        logger.info(f"保存商品详情: {count} 条 (品牌: {product_brand})")
        return count

    def get_products(self, asins: List[str] = None, brand: str = None, limit: int = None) -> List[Dict]:
        """获取商品详情"""
        sql = "SELECT * FROM products WHERE 1=1"
        params = []

        if asins:
            placeholders = ','.join(['?' for _ in asins])
            sql += f" AND asin IN ({placeholders})"
            params.extend(asins)

        if brand:
            sql += " AND brand = ?"
            params.append(brand)

        sql += " ORDER BY rating_count DESC"

        if limit:
            sql += f" LIMIT {limit}"

        results = self.db.execute(sql, tuple(params))

        # 解析 JSON 字段
        for r in results:
            for field in ['sub_category_ranks', 'bullet_points', 'images', 'variants']:
                if r.get(field):
                    try:
                        r[field] = json.loads(r[field])
                    except json.JSONDecodeError:
                        r[field] = []

        return results

    def get_product(self, asin: str) -> Optional[Dict]:
        """获取单个商品"""
        products = self.get_products([asin])
        return products[0] if products else None

    def get_product_asins(self) -> List[str]:
        """获取所有已抓取商品的ASIN"""
        results = self.db.execute("SELECT asin FROM products")
        return [r['asin'] for r in results]

    def get_products_for_review(self, top_n: int = 50) -> List[str]:
        """获取评论数最多的商品ASIN用于抓取评论"""
        results = self.db.execute(
            "SELECT asin FROM products WHERE rating_count > 0 ORDER BY rating_count DESC LIMIT ?",
            (top_n,)
        )
        return [r['asin'] for r in results]

    # ==================== 评论 ====================

    def save_reviews(self, reviews: List[Any], brand: str = "") -> int:
        """保存评论"""
        if not reviews:
            return 0

        sql = '''
            INSERT OR REPLACE INTO reviews 
            (asin, brand, review_id, title, content, rating, date, 
             verified_purchase, helpful_votes, author, variant)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        '''

        params_list = []
        for r in reviews:
            data = r.to_dict() if hasattr(r, 'to_dict') else r
            params_list.append((
                data.get('asin'),
                brand or data.get('brand', ''),
                data.get('review_id'),
                data.get('title'),
                data.get('content'),
                data.get('rating'),
                data.get('date'),
                1 if data.get('verified_purchase') else 0,
                data.get('helpful_votes', 0),
                data.get('author'),
                data.get('variant'),
            ))

        count = self.db.execute_many(sql, params_list)
        logger.info(f"保存评论: {count} 条 (品牌: {brand})")
        return count

    def get_reviews(self, asin: str = None, brand: str = None, rating: int = None,
                    verified_only: bool = False, limit: int = None) -> List[Dict]:
        """获取评论"""
        sql = "SELECT * FROM reviews WHERE 1=1"
        params = []

        if asin:
            sql += " AND asin = ?"
            params.append(asin)

        if brand:
            sql += " AND brand = ?"
            params.append(brand)

        if rating:
            sql += " AND rating = ?"
            params.append(rating)

        if verified_only:
            sql += " AND verified_purchase = 1"

        sql += " ORDER BY helpful_votes DESC, date DESC"

        if limit:
            sql += f" LIMIT {limit}"

        return self.db.execute(sql, tuple(params))

    def get_review_stats(self, asin: str = None) -> Dict:
        """获取评论统计"""
        where = f"WHERE asin = '{asin}'" if asin else ""

        stats = self.db.execute(f'''
            SELECT 
                COUNT(*) as total,
                AVG(rating) as avg_rating,
                SUM(CASE WHEN verified_purchase = 1 THEN 1 ELSE 0 END) as verified_count,
                SUM(CASE WHEN rating >= 4 THEN 1 ELSE 0 END) as positive_count,
                SUM(CASE WHEN rating <= 2 THEN 1 ELSE 0 END) as negative_count
            FROM reviews {where}
        ''')

        return stats[0] if stats else {}

    # ==================== 任务管理 ====================

    def add_task(self, task_type: str, target: str) -> int:
        """添加抓取任务"""
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT OR IGNORE INTO scrape_tasks (task_type, target) VALUES (?, ?)",
                (task_type, target)
            )
            return cursor.lastrowid

    def get_pending_tasks(self, task_type: str, limit: int = 100) -> List[Dict]:
        """获取待处理任务"""
        return self.db.execute(
            "SELECT * FROM scrape_tasks WHERE task_type = ? AND status = 'pending' LIMIT ?",
            (task_type, limit)
        )

    def update_task_status(self, task_id: int, status: str, error: str = None):
        """更新任务状态"""
        self.db.execute(
            "UPDATE scrape_tasks SET status = ?, error_message = ?, updated_at = ? WHERE id = ?",
            (status, error, datetime.now().isoformat(), task_id)
        )

    # ==================== 数据分析 ====================

    def get_brand_stats(self) -> List[Dict]:
        """获取品牌统计"""
        return self.db.execute('''
            SELECT 
                brand,
                COUNT(*) as product_count,
                AVG(rating) as avg_rating,
                SUM(rating_count) as total_reviews,
                AVG(bsr_rank) as avg_bsr
            FROM products 
            WHERE brand IS NOT NULL AND brand != ''
            GROUP BY brand
            ORDER BY product_count DESC
            LIMIT 50
        ''')

    def get_price_distribution(self) -> List[Dict]:
        """获取价格分布"""
        return self.db.execute('''
            SELECT 
                CASE 
                    WHEN CAST(REPLACE(REPLACE(price, '$', ''), ',', '') AS REAL) < 20 THEN '0-20'
                    WHEN CAST(REPLACE(REPLACE(price, '$', ''), ',', '') AS REAL) < 50 THEN '20-50'
                    WHEN CAST(REPLACE(REPLACE(price, '$', ''), ',', '') AS REAL) < 100 THEN '50-100'
                    WHEN CAST(REPLACE(REPLACE(price, '$', ''), ',', '') AS REAL) < 200 THEN '100-200'
                    ELSE '200+'
                END as price_range,
                COUNT(*) as count
            FROM products 
            WHERE price IS NOT NULL AND price != ''
            GROUP BY price_range
            ORDER BY 
                CASE price_range
                    WHEN '0-20' THEN 1
                    WHEN '20-50' THEN 2
                    WHEN '50-100' THEN 3
                    WHEN '100-200' THEN 4
                    ELSE 5
                END
        ''')

    def get_merged_data(self, brand: str = None) -> List[Dict]:
        """获取搜索结果与商品详情合并数据"""
        sql = '''
            SELECT 
                s.asin,
                s.search_keyword,
                s.search_rank,
                s.is_sponsored,
                s.is_amazon_choice,
                s.is_best_seller,
                p.title,
                p.brand,
                p.price,
                p.rating,
                p.rating_count,
                p.bsr_rank,
                p.bsr_category,
                p.is_fba,
                p.has_aplus,
                p.seller,
                p.first_available,
                (SELECT COUNT(*) FROM reviews r WHERE r.asin = s.asin AND (r.brand = ? OR ? = '')) as review_count
            FROM search_results s
            LEFT JOIN products p ON s.asin = p.asin
            WHERE (? = '' OR s.brand = ?)
            ORDER BY s.search_rank
        '''
        return self.db.execute(sql, (brand or '', brand or '', brand or '', brand or ''))

    def get_products_by_brand(self, brand: str, limit: int = None) -> List[Dict]:
        """按品牌获取商品"""
        return self.get_products(brand=brand, limit=limit)

    def get_reviews_by_brand(self, brand: str, limit: int = None) -> List[Dict]:
        """按品牌获取评论"""
        return self.get_reviews(brand=brand, limit=limit)

    def get_brand_progress(self, brand: str) -> Dict:
        """获取品牌爬取进度"""
        search_count = self.db.execute(
            "SELECT COUNT(*) as count FROM search_results WHERE brand = ?", (brand,)
        )[0]['count']

        product_count = self.db.execute(
            "SELECT COUNT(*) as count FROM products WHERE brand = ?", (brand,)
        )[0]['count']

        review_count = self.db.execute(
            "SELECT COUNT(*) as count FROM reviews WHERE brand = ?", (brand,)
        )[0]['count']

        return {
            'brand': brand,
            'search_results': search_count,
            'products': product_count,
            'reviews': review_count,
        }
