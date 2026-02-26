"""
评论解析器 - 解析 Amazon 评论页面
"""

import re
import hashlib
import logging
from typing import List, Optional
from dataclasses import dataclass, asdict
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)


@dataclass
class Review:
    """评论数据模型"""
    asin: str
    review_id: str
    brand: str = ""
    title: str = ""
    content: str = ""
    rating: int = 0
    date: str = ""
    verified_purchase: bool = False
    helpful_votes: int = 0
    author: str = ""
    variant: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


class ReviewParser:
    """评论解析器"""

    def parse(self, html: str, asin: str = "") -> List[Review]:
        """
        解析评论页面

        Args:
            html: 页面HTML内容
            asin: 商品ASIN

        Returns:
            评论列表
        """
        soup = BeautifulSoup(html, 'html.parser')
        reviews = []

        review_selectors = [
            'div[data-hook="review"]',
            'div.review',
            'div[id^="customer_review-"]',
        ]

        items = []
        for selector in review_selectors:
            items = soup.select(selector)
            if items:
                break

        for item in items:
            try:
                review = Review(
                    asin=asin,
                    review_id=self._extract_review_id(item),
                    title=self._extract_title(item),
                    content=self._extract_content(item),
                    rating=self._extract_rating(item),
                    date=self._extract_date(item),
                    verified_purchase=self._is_verified_purchase(item),
                    helpful_votes=self._extract_helpful_votes(item),
                    author=self._extract_author(item),
                    variant=self._extract_variant(item),
                )

                if review.review_id:
                    reviews.append(review)

            except Exception as e:
                logger.debug(f"解析评论失败: {e}")
                continue

        logger.info(f"解析评论页，找到 {len(reviews)} 条评论")
        return reviews

    def _extract_review_id(self, item: BeautifulSoup) -> str:
        """提取评论ID"""
        review_id = item.get('id', '')
        if review_id.startswith('customer_review-'):
            return review_id.replace('customer_review-', '')
        if review_id:
            return review_id
        # 生成唯一ID
        return hashlib.md5(item.get_text()[:100].encode()).hexdigest()[:12]

    def _extract_title(self, item: BeautifulSoup) -> str:
        """提取评论标题"""
        selectors = [
            'a[data-hook="review-title"] span:not(.a-size-base)',
            '[data-hook="review-title"]',
            '.review-title',
        ]

        for selector in selectors:
            element = item.select_one(selector)
            if element:
                text = element.get_text(strip=True)
                if text and len(text) > 2:
                    return text
        return ""

    def _extract_content(self, item: BeautifulSoup) -> str:
        """提取评论内容"""
        selectors = [
            'span[data-hook="review-body"] span',
            '[data-hook="review-body"]',
            '.review-text-content span',
        ]

        for selector in selectors:
            element = item.select_one(selector)
            if element:
                text = element.get_text(strip=True)
                if text:
                    return text
        return ""

    def _extract_rating(self, item: BeautifulSoup) -> int:
        """提取评论评分"""
        selectors = [
            'i[data-hook="review-star-rating"] span.a-icon-alt',
            'i.review-rating span.a-icon-alt',
            'span.a-icon-alt',
        ]

        for selector in selectors:
            element = item.select_one(selector)
            if element:
                text = element.get_text(strip=True)
                match = re.search(r'([\d.]+)', text)
                if match:
                    try:
                        return int(float(match.group(1)))
                    except ValueError:
                        pass
        return 0

    def _extract_date(self, item: BeautifulSoup) -> str:
        """提取评论日期"""
        element = item.select_one('span[data-hook="review-date"]')
        if element:
            text = element.get_text(strip=True)
            match = re.search(r'on\s+(.+)$', text)
            if match:
                return match.group(1)
            return text
        return ""

    def _is_verified_purchase(self, item: BeautifulSoup) -> bool:
        """检测是否验证购买"""
        return item.select_one('span[data-hook="avp-badge"]') is not None

    def _extract_helpful_votes(self, item: BeautifulSoup) -> int:
        """提取有用投票数"""
        helpful = item.select_one('span[data-hook="helpful-vote-statement"]')
        if helpful:
            text = helpful.get_text(strip=True)
            match = re.search(r'(\d+)\s+people?', text)
            if match:
                return int(match.group(1))
            if 'one person' in text.lower():
                return 1
        return 0

    def _extract_author(self, item: BeautifulSoup) -> str:
        """提取作者"""
        element = item.select_one('span.a-profile-name')
        if element:
            return element.get_text(strip=True)
        return ""

    def _extract_variant(self, item: BeautifulSoup) -> str:
        """提取变体信息"""
        variant = item.select_one('a[data-hook="format-strip"]')
        if variant:
            return variant.get_text(strip=True)
        return ""
