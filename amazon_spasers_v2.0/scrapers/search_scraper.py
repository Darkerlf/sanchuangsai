"""
搜索爬虫 - 抓取 Amazon 搜索结果页面
"""

import time
import random
import logging
from typing import List, Optional

from browser import BrowserManager
from parsers import SearchParser
from parsers.search_parser import SearchResult
from config import config

logger = logging.getLogger(__name__)


class SearchScraper:
    """搜索爬虫"""

    def __init__(self, browser: BrowserManager):
        self.browser = browser
        self.parser = SearchParser()

    def scrape(self, keyword: str, max_pages: int = None, brand: str = "") -> List[SearchResult]:
        """
        抓取搜索结果

        Args:
            keyword: 搜索关键词
            max_pages: 最大页数
            brand: 品牌（用于过滤）

        Returns:
            搜索结果列表
        """
        max_pages = max_pages or config.SEARCH_MAX_PAGES
        all_results = []
        seen_asins = set()
        filtered_count = 0

        logger.info(f"开始搜索: {keyword}")
        if config.FILTER_BY_BRAND and brand:
            logger.info(f"品牌过滤已启用，目标品牌: {brand} (匹配模式: {config.BRAND_MATCH_MODE})")
            config.TARGET_BRANDS = [brand]

        for page in range(1, max_pages + 1):
            try:
                url = f"https://www.amazon.com/s?k={keyword.replace(' ', '+')}&page={page}"
                logger.info(f"抓取搜索页 {page}/{max_pages}")

                driver = self.browser.get_driver()
                driver.get(url)
                self.browser.wait_for_page_load()

                # 随机延迟
                time.sleep(random.uniform(*config.PAGE_LOAD_DELAY))

                # 检查验证码
                if self.browser.check_captcha():
                    logger.warning("请手动处理验证码后按回车继续...")
                    input()
                    driver.get(url)
                    self.browser.wait_for_page_load()

                # 滚动页面
                self.browser.scroll_page()

                # 解析结果
                html = driver.page_source
                results = self.parser.parse(html, keyword=keyword, page=page)

                # 品牌过滤
                if config.FILTER_BY_BRAND:
                    filtered_results = []
                    for r in results:
                        if self._is_target_brand(r.brand):
                            filtered_results.append(r)
                        else:
                            filtered_count += 1
                            logger.debug(f"过滤非目标品牌商品: {r.asin} - 品牌: {r.brand}")
                    results = filtered_results

                # 去重
                new_results = []
                for r in results:
                    if r.asin not in seen_asins:
                        seen_asins.add(r.asin)
                        new_results.append(r)

                all_results.extend(new_results)
                logger.info(f"页面 {page}: 找到 {len(new_results)} 个新商品")

                # 检查是否有下一页
                from bs4 import BeautifulSoup
                soup = BeautifulSoup(html, 'html.parser')
                next_button = soup.select_one('a.s-pagination-next:not(.s-pagination-disabled)')
                if not next_button:
                    logger.info("没有更多页面")
                    break

                # 页间延迟
                time.sleep(random.uniform(*config.REQUEST_DELAY))

            except Exception as e:
                logger.error(f"搜索页 {page} 抓取失败: {e}")
                continue

        logger.info(f"搜索完成: {keyword}, 共 {len(all_results)} 个商品")
        if config.FILTER_BY_BRAND and filtered_count > 0:
            logger.info(f"品牌过滤: 已过滤 {filtered_count} 个非目标品牌商品")
        return all_results

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

    def scrape_keywords(self, keywords: List[str] = None, brand: str = "") -> List[SearchResult]:
        """
        抓取多个关键词

        Args:
            keywords: 关键词列表
            brand: 品牌（用于过滤）

        Returns:
            去重后的搜索结果列表
        """
        keywords = keywords or config.SEARCH_KEYWORDS
        all_results = []
        seen_asins = set()

        for keyword in keywords:
            results = self.scrape(keyword, brand=brand)

            for r in results:
                if r.asin not in seen_asins:
                    seen_asins.add(r.asin)
                    all_results.append(r)

            time.sleep(random.uniform(*config.REQUEST_DELAY))

        logger.info(f"所有关键词搜索完成，共 {len(all_results)} 个唯一商品")
        return all_results
