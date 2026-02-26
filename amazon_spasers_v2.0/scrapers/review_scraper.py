"""
评论爬虫 - 抓取 Amazon 评论页面 (终极修正版)
"""

import time
import random
import logging
from typing import List

from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException

from browser import BrowserManager
from parsers import ReviewParser
from parsers.review_parser import Review
from config import config

logger = logging.getLogger(__name__)


class ReviewScraper:
    """评论爬虫"""

    def __init__(self, browser: BrowserManager):
        self.browser = browser
        self.parser = ReviewParser()

    def scrape(self, asin: str, max_pages: int = None, brand: str = "") -> List[Review]:
        """
        抓取商品评论
        特性：
        1. 模拟真人点击翻页 (解决重定向回首页问题)
        2. 狗狗页(Dog Page)检测与自动等待 (解决反爬拦截)
        """
        max_pages = max_pages or config.REVIEWS_MAX_PAGES
        all_reviews = []
        seen_ids = set()

        logger.info(f"开始抓取评论: {asin} (品牌: {brand})")

        # 1. 初始访问第 1 页
        # 使用显式参数有助于建立正确的 Session
        url = f"https://www.amazon.com/product-reviews/{asin}/ref=cm_cr_dp_d_show_all_btm?ie=UTF8&reviewerType=all_reviews&sortBy=recent"

        driver = self.browser.get_driver()
        try:
            driver.get(url)
            self.browser.wait_for_page_load()
        except Exception as e:
            logger.error(f"初始页面加载失败: {e}")
            return []

        for page in range(1, max_pages + 1):
            try:
                logger.info(f"正在处理第 {page} 页...")

                # ==========================================
                # 🛑 核心防护：检测亚马逊“狗狗页” (反爬拦截)
                # ==========================================
                title = driver.title.lower()
                if "sorry" in title or "server busy" in title or "robot check" in title:
                    logger.warning(f"🐶 汪汪！检测到 Amazon 狗狗页 (反爬拦截) - ASIN: {asin}")
                    logger.info("⏳ 触发熔断保护：暂停 60 秒等待解封...")
                    time.sleep(60)

                    try:
                        logger.info("🔄 尝试刷新页面...")
                        driver.refresh()
                        self.browser.wait_for_page_load()

                        # 再次检查是否解除
                        if "sorry" in driver.title.lower():
                            logger.error("❌ 刷新无效，仍然被拦截。放弃当前商品剩余页面。")
                            break
                    except Exception:
                        break
                # ==========================================

                # --- 随机行为模拟 ---
                time.sleep(random.uniform(2.5, 5.0))  # 稍微调大等待时间
                self.browser.scroll_page()

                # --- 验证码检查 ---
                if self.browser.check_captcha():
                    logger.warning("遇到验证码，请手动处理...")
                    # 可以在这里加 input() 阻塞，或者直接跳过
                    time.sleep(5)
                    if self.browser.check_captcha():
                         logger.error("验证码未通过，跳过")
                         break

                # --- 解析当前页 ---
                html = driver.page_source
                reviews = self.parser.parse(html, asin=asin)

                # 标记品牌
                for review in reviews:
                    review.brand = brand

                # --- 去重逻辑 ---
                new_reviews = []
                for r in reviews:
                    if r.review_id not in seen_ids:
                        seen_ids.add(r.review_id)
                        new_reviews.append(r)

                # --- ⚠️ 防止无限重定向回第 1 页的保护机制 ---
                # 如果不是第 1 页，却找到了评论，但全是旧的，说明亚马逊把我们踢回了第 1 页
                if page > 1 and len(reviews) > 0 and len(new_reviews) == 0:
                    logger.warning(f"⚠️ 第 {page} 页检测到重复内容（Amazon 重定向回首页），停止抓取。")
                    break

                # 如果页面本身就没有评论（解析出0条），且不是第1页，说明到底了
                if len(reviews) == 0 and page > 1:
                    logger.info("没有读取到评论，可能已到达末页。")
                    break

                logger.info(f"✅ 第 {page} 页: 成功提取 {len(new_reviews)} 条新评论")
                all_reviews.extend(new_reviews)

                # --- 翻页逻辑 (使用 JavaScript 点击) ---
                if page < max_pages:
                    try:
                        # 1. 寻找“下一页”按钮 (精准定位 li.a-last > a)
                        next_btn = WebDriverWait(driver, 5).until(
                            EC.presence_of_element_located((By.CSS_SELECTOR, "li.a-last a"))
                        )

                        # 2. 获取 URL 仅用于日志，不用于跳转
                        next_url = next_btn.get_attribute("href")
                        logger.debug(f"准备跳转下一页... Target: {next_url}")

                        # 3. 【关键】使用 JS 点击，保留 Referer，模拟真实用户
                        driver.execute_script("arguments[0].click();", next_btn)

                        # 4. 【关键】等待 URL 发生变化，确保翻页成功
                        # 我们等待 URL 中出现 pageNumber={page+1}
                        try:
                            WebDriverWait(driver, 10).until(
                                lambda d: f"pageNumber={page + 1}" in d.current_url
                                or (next_url and next_url in d.current_url)
                            )
                        except TimeoutException:
                            logger.warning(f"⏳ 第 {page} 页点击后 URL 未及时变化，可能加载较慢或已被重定向")

                        self.browser.wait_for_page_load()

                    except (NoSuchElementException, TimeoutException):
                        logger.info("🚫 没有“下一页”按钮了，抓取结束。")
                        break
                    except Exception as e:
                        logger.error(f"❌ 翻页操作失败: {e}")
                        break

            except Exception as e:
                logger.error(f"第 {page} 页发生未知错误: {e}")
                break

        logger.info(f"🎉 评论抓取完成: {asin}, 共 {len(all_reviews)} 条")
        return all_reviews

    def scrape_batch(self, asins: List[str], save_callback=None, brand: str = "") -> List[Review]:
        """
        批量抓取评论
        """
        all_reviews = []

        total = len(asins)
        for idx, asin in enumerate(asins, 1):
            logger.info(f"👉 正在处理第 {idx}/{total} 个商品: {asin}")

            reviews = self.scrape(asin, brand=brand)
            all_reviews.extend(reviews)

            # 定期保存
            if save_callback:
                save_callback(reviews)  # 每次爬完一个商品就保存一次，更安全

            # 商品之间的大延迟，防止封号
            if idx < total:
                sleep_time = random.uniform(5.0, 10.0)
                logger.info(f"💤 商品间休息 {sleep_time:.1f} 秒...")
                time.sleep(sleep_time)

        logger.info(f"批量评论抓取完成，共 {len(all_reviews)} 条")
        return all_reviews