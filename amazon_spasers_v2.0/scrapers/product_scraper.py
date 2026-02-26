"""
商品爬虫 - 抓取 Amazon 商品详情页面
"""

import time
import random
import logging
from typing import List, Optional

from browser import BrowserManager
from parsers import ProductParser
from parsers.product_parser import ProductDetail
from config import config

logger = logging.getLogger(__name__)


class ProductScraper:
    """商品详情爬虫"""

    def __init__(self, browser: BrowserManager):
        self.browser = browser
        self.parser = ProductParser()

    def scrape(self, asin: str, retries: int = None, brand: str = "") -> Optional[ProductDetail]:
        """
        抓取商品详情

        Args:
            asin: 商品ASIN
            retries: 重试次数
            brand: 品牌（用于过滤）

        Returns:
            商品详情对象
        """
        retries = retries or config.MAX_RETRIES

        for attempt in range(retries):
            try:
                url = f"https://www.amazon.com/dp/{asin}"
                logger.info(f"抓取商品: {asin} (尝试 {attempt + 1}/{retries})")

                driver = self.browser.get_driver()
                driver.get(url)
                self.browser.wait_for_page_load()

                # 随机延迟
                time.sleep(random.uniform(*config.PAGE_LOAD_DELAY))

                # 检查验证码
                if self.browser.check_captcha():
                    logger.warning("请手动处理验证码后按回车继续...")
                    input()
                    continue

                # 滚动页面
                self.browser.scroll_page()

                # 解析
                html = driver.page_source
                product = self.parser.parse(html, asin=asin)

                if product.title:
                    # 品牌过滤
                    if config.FILTER_BY_BRAND and brand:
                        config.TARGET_BRANDS = [brand]
                        if not self._is_target_brand(product.brand):
                            logger.info(f"过滤非目标品牌商品: {asin} - 品牌: {product.brand}")
                            return None

                    logger.info(f"成功: {asin} - {product.title[:50]}...")
                    return product
                else:
                    logger.warning(f"解析失败，无标题: {asin}")

            except Exception as e:
                logger.error(f"抓取商品 {asin} 失败: {e}")

            time.sleep(random.uniform(*config.RETRY_DELAY))

        logger.error(f"放弃抓取: {asin}")
        return None

    def _is_target_brand(self, brand: str) -> bool:
        """检查是否为目标品牌"""
        if not config.FILTER_BY_BRAND:
            return True

        if not brand:
            return False

        brand_lower = brand.lower().strip()
        target_brands_lower = [b.lower().strip() for b in config.TARGET_BRANDS]

        if config.BRAND_MATCH_MODE == "exact":
            return brand_lower in target_brands_lower
        else:
            return any(target in brand_lower for target in target_brands_lower)

    def scrape_batch(self, asins: List[str], save_callback=None, brand: str = "") -> List[ProductDetail]:
        """
        批量抓取商品详情

        Args:
            asins: ASIN列表
            save_callback: 保存回调函数
            brand: 品牌（用于过滤）

        Returns:
            商品详情列表
        """
        products = []
        failed = []

        total = len(asins)
        for idx, asin in enumerate(asins, 1):
            logger.info(f"进度: {idx}/{total}")

            product = self.scrape(asin, brand=brand)
            if product:
                products.append(product)

                # 定期保存
                if save_callback and idx % 10 == 0:
                    save_callback(products)
            else:
                failed.append(asin)

            time.sleep(random.uniform(*config.REQUEST_DELAY))

        # 最终保存
        if save_callback:
            save_callback(products)

        logger.info(f"批量抓取完成: 成功 {len(products)}, 失败 {len(failed)}")
        if failed:
            logger.warning(f"失败的ASIN: {failed}")

        return products
