"""
从 CSV 文件读取 ASIN 并抓取评论
"""
import pandas as pd
import logging
from main import AmazonScraperApp, setup_logging


def run_reviews_from_csv(csv_file='merged_data.csv'):
    # 1. 配置日志
    setup_logging()
    logger = logging.getLogger(__name__)

    print(f"正在从 {csv_file} 读取 ASIN...")

    try:
        # 2. 读取 CSV 数据
        df = pd.read_csv('./data/merged_data.csv')

        if 'asin' not in df.columns:
            print("❌ 错误: CSV 文件中未找到 'asin' 列")
            return

        # 3. 筛选需要抓取的商品
        # 如果有 rating_count 列，只抓取有评分的商品，节省时间
        if 'rating_count' in df.columns:
            # 过滤掉 rating_count 为空或为 0 的商品
            valid_products = df[df['rating_count'] > 0]
            asins = valid_products['asin'].unique().tolist()
            print(f"📝 发现 {len(df)} 个商品，其中 {len(asins)} 个包含评分，将对这些商品抓取评论。")
        else:
            asins = df['asin'].unique().tolist()
            print(f"📝 将抓取全部 {len(asins)} 个商品的评论。")

        if not asins:
            print("⚠️ 没有找到需要抓取评论的 ASIN。")
            return

        # 4. 启动爬虫
        # headless=True 为无头模式（不显示浏览器），如果需要观察过程请改为 False
        app = AmazonScraperApp(headless=False)

        try:
            print("\n🚀 开始启动浏览器...")
            app.start()

            print(f"📊 开始抓取评论列表...")
            # 调用 run_reviews 并传入我们从 CSV 读取的 asin 列表
            app.run_reviews(asins=asins)

            print("\n✅ 评论抓取完成！正在导出数据...")
            app.export_all()
            print(f"📂 数据已更新并导出到 data/ 目录")

        except Exception as e:
            logger.error(f"运行过程中出错: {e}", exc_info=True)
        finally:
            app.stop()
            print("🛑 程序已结束")

    except FileNotFoundError:
        print(f"❌ 找不到文件: {csv_file}")
    except Exception as e:
        print(f"❌ 发生未知错误: {e}")


if __name__ == "__main__":
    run_reviews_from_csv()