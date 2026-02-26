"""
Amazon 商品研究工具 - 主程序
"""
import sys
import logging
import argparse
from datetime import datetime
from pathlib import Path
import random

from config import config
from browser import BrowserManager
from scrapers import SearchScraper, ProductScraper, ReviewScraper
from storage import Database, DatabaseStorage, JsonStorage, ExcelExporter


# ==================== 日志配置 ====================
def setup_logging():
    """配置日志"""
    log_dir = Path(config.LOG_DIR)
    log_dir.mkdir(parents=True, exist_ok=True)

    log_file = log_dir / f"scraper_{datetime.now().strftime('%Y%m%d')}.log"

    logging.basicConfig(
        level=getattr(logging, config.LOG_LEVEL),
        format=config.LOG_FORMAT,
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )

    # 降低第三方库日志级别
    logging.getLogger('selenium').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)


logger = logging.getLogger(__name__)


# ==================== 主运行类 ====================
class AmazonScraperApp:
    """Amazon 爬虫应用"""

    def __init__(self, headless: bool = False):
        self.browser = BrowserManager(headless=headless)
        self.db = Database()
        self.db_storage = DatabaseStorage(self.db)
        self.json_storage = JsonStorage()
        self.exporter = ExcelExporter()

        self.search_scraper = None
        self.product_scraper = None
        self.review_scraper = None

    def start(self):
        """启动浏览器"""
        self.browser.create_driver()
        self.search_scraper = SearchScraper(self.browser)
        self.product_scraper = ProductScraper(self.browser)
        self.review_scraper = ReviewScraper(self.browser)
        
        # 检查登录状态
        if not self.browser.check_login_status():
            logger.warning("检测到未登录，请手动登录后再继续")
            self.browser.manual_login_guide()
            
            # 再次检查登录状态
            if not self.browser.check_login_status():
                logger.error("登录失败，程序退出")
                raise RuntimeError("需要登录 Amazon 才能继续使用爬虫")
        
        logger.info("爬虫已启动")

    def stop(self):
        """停止浏览器"""
        self.browser.close()
        logger.info("爬虫已停止")

    def run_search(self, keywords: list = None, brand: str = None):
        """运行搜索任务"""
        keywords = keywords or config.SEARCH_KEYWORDS
        brand = brand or config.CURRENT_BRAND

        logger.info(f"开始搜索: {keywords}")
        if brand:
            logger.info(f"目标品牌: {brand}")

        results = self.search_scraper.scrape_keywords(keywords, brand=brand or "")

        # 保存到数据库
        self.db_storage.save_search_results(results, brand=brand or "")

        # 保存到 JSON
        self.json_storage.save([r.to_dict() for r in results], 'search_results.json')

        logger.info(f"搜索完成，共 {len(results)} 个商品")
        return results

    def run_products(self, asins: list = None, brand: str = None):
        """运行商品详情抓取任务"""
        brand = brand or config.CURRENT_BRAND

        if asins is None:
            # 从数据库获取待抓取的 ASIN
            scraped_asins = set(self.db_storage.get_product_asins())
            search_asins = set(self.db_storage.get_search_asins())
            asins = list(search_asins - scraped_asins)

        if not asins:
            logger.info("没有需要抓取的商品")
            return []

        logger.info(f"开始抓取商品详情: {len(asins)} 个")
        if brand:
            logger.info(f"目标品牌: {brand}")

        def save_callback(products):
            self.db_storage.save_products(products, brand=brand or "")

        products = self.product_scraper.scrape_batch(asins, save_callback=save_callback, brand=brand or "")

        # 保存到 JSON
        self.json_storage.save([p.to_dict() for p in products], 'products.json')

        logger.info(f"商品抓取完成，共 {len(products)} 个")
        return products

    def run_reviews(self, asins: list = None, top_n: int = None, brand: str = None):
        """
        运行评论抓取任务（支持断点续传 + 周期性重启浏览器）
        """
        top_n = top_n or config.REVIEWS_TOP_N
        brand = brand or config.CURRENT_BRAND


        # 1. 获取原本计划要抓取的所有 ASIN
        if asins is None:
            # 获取评论数最多的商品
            asins = self.db_storage.get_products_for_review(top_n)

        if not asins:
            logger.info("没有需要抓取评论的商品")
            return []

        # ========================================================
        # 🚀 新增：断点续传过滤逻辑 (自动跳过已抓取的商品)
        # ========================================================
        try:
            # 从数据库的 reviews 表中查找所有已经存在的 ASIN
            # 注意：这里假设只要 reviews 表里有这个 ASIN 的记录，就算爬过了。
            # 如果你想更严谨（比如检查评论数是否够多），逻辑会更复杂。
            existing_rows = self.db.execute("""
                            SELECT asin 
                            FROM reviews 
                            GROUP BY asin 
                            HAVING COUNT(*) > 100  -- 阈值：至少有5条评论才算已爬过
                        """)
            scraped_asins = {row['asin'] for row in existing_rows}

            original_count = len(asins)
            # 过滤
            asins = [a for a in asins if a not in scraped_asins]

            skipped_count = original_count - len(asins)
            if skipped_count > 0:
                logger.info("=" * 50)
                logger.info(f"⏭️  断点续传启动: 发现 {skipped_count} 个商品已抓取，将自动跳过。")
                logger.info(f"📋  剩余任务: {len(asins)} 个商品")
                logger.info("=" * 50)

            if not asins:
                logger.info("🎉 所有目标商品的评论都已存在数据库中，无需抓取！")
                return []

        except Exception as e:
            logger.warning(f"断点续传检查失败（可能是首次运行），将全部抓取: {e}")
        # ========================================================

        logger.info(f"开始抓取评论: {len(asins)} 个商品")
        if brand:
            logger.info(f"目标品牌: {brand}")

        all_reviews = []

        # === 分批次处理 (保持之前的防封号逻辑) ===
        BATCH_SIZE = 5

        for i in range(0, len(asins), BATCH_SIZE):
            batch_asins = asins[i: i + BATCH_SIZE]
            current_batch_num = (i // BATCH_SIZE) + 1
            total_batches = (len(asins) + BATCH_SIZE - 1) // BATCH_SIZE

            logger.info(f"\n🔄 正在执行第 {current_batch_num}/{total_batches} 批次 (本批 {len(batch_asins)} 个商品)...")

            # 1. 确保浏览器是新鲜开启的
            if self.browser.driver is None:
                logger.info("启动新浏览器实例...")
                self.browser.create_driver()
                self.review_scraper = ReviewScraper(self.browser)

            # 2. 执行本批次的抓取
            def save_callback(reviews):
                self.db_storage.save_reviews(reviews, brand=brand or "")

            try:
                batch_reviews = self.review_scraper.scrape_batch(
                    batch_asins,
                    save_callback=save_callback,
                    brand=brand or ""
                )
                all_reviews.extend(batch_reviews)

            except Exception as e:
                logger.error(f"批次执行异常: {e}")

            # 3. 本批次结束，关闭浏览器
            logger.info("♻️ 本批次完成，关闭浏览器以规避检测...")
            self.browser.close()

            # 4. 批次间长休息
            if i + BATCH_SIZE < len(asins):
                sleep_time = random.uniform(15, 30)
                logger.info(f"☕ 休息 {sleep_time:.1f} 秒后继续...")
                import time
                time.sleep(sleep_time)

        # 保存总结果到 JSON (注意：这里只保存本次新抓取的，旧的在数据库里)
        # 如果你想把所有评论（含旧的）都导出，建议运行 python main.py export
        if all_reviews:
            self.json_storage.save([r.to_dict() for r in all_reviews], 'reviews_new.json')

        logger.info(f"本次任务完成，共抓取 {len(all_reviews)} 条新评论")
        return all_reviews

    def run_all(self, keywords: list = None):
        """运行完整流程"""
        logger.info("=" * 50)
        logger.info("开始完整抓取流程")
        logger.info("=" * 50)

        # 1. 搜索
        logger.info("\n📍 阶段 1/3: 搜索商品")
        self.run_search(keywords)

        # 2. 商品详情
        logger.info("\n📍 阶段 2/3: 抓取商品详情")
        self.run_products()

        # 3. 评论
        logger.info("\n📍 阶段 3/3: 抓取评论")
        self.run_reviews()

        # 4. 导出
        logger.info("\n📍 导出数据...")
        self.export_all()

        # 5. 统计
        self.print_stats()

        logger.info("\n✅ 完整流程执行完成!")

    def run_brand_stage(self, brand: str, stage: str, keywords: list = None):
        """
        按品牌和阶段爬取

        Args:
            brand: 品牌名称
            stage: 爬取阶段 (search/products/reviews)
            keywords: 搜索关键词（可选）
        """
        logger.info("=" * 50)
        logger.info(f"开始爬取品牌: {brand}, 阶段: {stage}")
        logger.info("=" * 50)

        config.CURRENT_BRAND = brand

        if stage == "search":
            logger.info("\n📍 阶段: 搜索商品")
            self.run_search(keywords, brand=brand)

        elif stage == "products":
            logger.info("\n📍 阶段: 抓取商品详情")
            self.run_products(brand=brand)

        elif stage == "reviews":
            logger.info("\n📍 阶段: 抓取评论")
            self.run_reviews(brand=brand)

        # 导出该品牌数据
        logger.info("\n📍 导出品牌数据...")
        self.export_brand(brand)

        # 显示统计
        self.print_brand_stats(brand)

        logger.info(f"\n✅ 品牌 {brand} 的 {stage} 阶段执行完成!")

    def export_all(self):
        """导出所有数据"""
        # 获取数据
        search_results = self.db_storage.get_search_results()
        products = self.db_storage.get_products()
        reviews = self.db_storage.get_reviews()
        merged = self.db_storage.get_merged_data()

        # 导出到 Excel（多 sheet）
        self.exporter.export_to_excel({
            '搜索结果': search_results,
            '商品详情': products,
            '评论': reviews,
            '综合数据': merged,
        }, 'amazon_data.xlsx')

        # 导出到 CSV
        self.exporter.export_to_csv(merged, 'merged_data.csv')

        # 生成报告
        self.exporter.generate_report(self.db_storage)

        logger.info("数据导出完成")

    def export_brand(self, brand: str):
        """导出指定品牌的数据"""
        self.exporter.export_brand_data(self.db_storage, brand)

    def print_stats(self):
        """打印统计信息"""
        stats = self.db.get_stats()

        print("\n" + "=" * 50)
        print("📊 数据统计")
        print("=" * 50)
        print(f"  搜索结果: {stats['search_results']} 条")
        print(f"  商品详情: {stats['products']} 个")
        print(f"  评论数量: {stats['reviews']} 条")
        print("=" * 50)

        # 品牌过滤状态
        if config.FILTER_BY_BRAND:
            print(f"\n🎯 品牌过滤: 已启用")
            print(f"  目标品牌: {', '.join(config.TARGET_BRANDS)}")
            print(f"  匹配模式: {config.BRAND_MATCH_MODE}")
        else:
            print(f"\n🎯 品牌过滤: 已禁用")

        # 品牌统计
        brand_stats = self.db_storage.get_brand_stats()
        if brand_stats:
            print("\n📈 品牌分布 (Top 10):")
            for i, brand in enumerate(brand_stats[:10], 1):
                print(f"  {i}. {brand['brand']}: {brand['product_count']} 个商品, "
                      f"评分 {brand['avg_rating']:.1f}, "
                      f"评论 {brand['total_reviews']} 条")

        # 评论统计
        review_stats = self.db_storage.get_review_stats()
        if review_stats and review_stats.get('total'):
            print(f"\n💬 评论统计:")
            print(f"  总评论数: {review_stats['total']}")
            print(f"  平均评分: {review_stats['avg_rating']:.2f}")
            print(
                f"  验证购买: {review_stats['verified_count']} ({review_stats['verified_count'] / review_stats['total'] * 100:.1f}%)")
            print(
                f"  好评(4-5星): {review_stats['positive_count']} ({review_stats['positive_count'] / review_stats['total'] * 100:.1f}%)")
            print(
                f"  差评(1-2星): {review_stats['negative_count']} ({review_stats['negative_count'] / review_stats['total'] * 100:.1f}%)")

    def print_brand_stats(self, brand: str):
        """打印指定品牌的统计信息"""
        progress = self.db_storage.get_brand_progress(brand)

        print("\n" + "=" * 50)
        print(f"📊 品牌 {brand} 数据统计")
        print("=" * 50)
        print(f"  搜索结果: {progress['search_results']} 条")
        print(f"  商品详情: {progress['products']} 个")
        print(f"  评论数量: {progress['reviews']} 条")
        print("=" * 50)

        # 该品牌商品统计
        brand_products = self.db_storage.get_products_by_brand(brand)
        if brand_products:
            avg_rating = sum(p.get('rating', 0) for p in brand_products if p.get('rating')) / len([p for p in brand_products if p.get('rating')])
            total_reviews = sum(p.get('rating_count', 0) for p in brand_products)
            print(f"\n📈 品牌统计:")
            print(f"  平均评分: {avg_rating:.2f}")
            print(f"  总评论数: {total_reviews}")

        # 该品牌评论统计
        brand_reviews = self.db_storage.get_reviews_by_brand(brand)
        if brand_reviews:
            verified_count = sum(1 for r in brand_reviews if r.get('verified_purchase'))
            positive_count = sum(1 for r in brand_reviews if r.get('rating', 0) >= 4)
            negative_count = sum(1 for r in brand_reviews if r.get('rating', 0) <= 2)
            total = len(brand_reviews)

            print(f"\n💬 评论统计:")
            print(f"  总评论数: {total}")
            print(f"  验证购买: {verified_count} ({verified_count / total * 100:.1f}%)")
            print(f"  好评(4-5星): {positive_count} ({positive_count / total * 100:.1f}%)")
            print(f"  差评(1-2星): {negative_count} ({negative_count / total * 100:.1f}%)")


# ==================== 命令行入口 ====================
def main():
    """主入口函数"""
    setup_logging()

    parser = argparse.ArgumentParser(description='Amazon 商品研究工具')
    parser.add_argument('command', nargs='?', default='all',
                        choices=['all', 'search', 'products', 'reviews', 'export', 'stats', 'brand'],
                        help='执行的命令')
    parser.add_argument('-k', '--keywords', nargs='+', help='搜索关键词')
    parser.add_argument('-a', '--asins', nargs='+', help='商品 ASIN')
    parser.add_argument('--headless', action='store_true', default=False, help='无头模式')
    parser.add_argument('--no-headless', action='store_false', dest='headless', help='有界面模式')
    parser.add_argument('-n', '--top-n', type=int, default=50, help='评论抓取商品数')
    parser.add_argument('--brands', nargs='+', help='目标品牌列表')
    parser.add_argument('--no-brand-filter', action='store_true', help='禁用品牌过滤')
    parser.add_argument('--brand', type=str, help='指定品牌（用于分阶段爬取）')
    parser.add_argument('--stage', type=str, choices=['search', 'products', 'reviews'],
                        help='指定爬取阶段（需要配合 --brand 使用）')
    parser.add_argument('--export-brand', type=str, help='导出指定品牌的数据')

    args = parser.parse_args()

    # 动态更新品牌配置
    if args.brands:
        config.TARGET_BRANDS = args.brands
        config.FILTER_BY_BRAND = True
        logger.info(f"设置目标品牌: {config.TARGET_BRANDS}")

    if args.no_brand_filter:
        config.FILTER_BY_BRAND = False
        logger.info("品牌过滤已禁用")

    app = AmazonScraperApp(headless=args.headless)

    try:
        if args.command in ['all', 'search', 'products', 'reviews', 'brand']:
            app.start()

        if args.command == 'all':
            app.run_all(args.keywords)

        elif args.command == 'search':
            app.run_search(args.keywords, brand=args.brand)
            app.print_stats()

        elif args.command == 'products':
            app.run_products(args.asins, brand=args.brand)
            app.print_stats()

        elif args.command == 'reviews':
            app.run_reviews(args.asins, args.top_n, brand=args.brand)
            app.print_stats()

        elif args.command == 'export':
            if args.export_brand:
                app.export_brand(args.export_brand)
            else:
                app.export_all()
            app.print_stats()

        elif args.command == 'stats':
            if args.brand:
                app.print_brand_stats(args.brand)
            else:
                app.print_stats()

        elif args.command == 'brand':
            if not args.brand or not args.stage:
                logger.error("使用 'brand' 命令需要同时指定 --brand 和 --stage 参数")
                logger.info("示例: python main.py brand --brand imarku --stage search")
                return

            app.run_brand_stage(args.brand, args.stage, args.keywords)

    except KeyboardInterrupt:
        logger.info("\n用户中断")
    except Exception as e:
        logger.error(f"运行出错: {e}", exc_info=True)
    finally:
        app.stop()


if __name__ == '__main__':
    main()
