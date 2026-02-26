"""
搜索结果解析器 - 解析 Amazon 搜索结果页面
"""

import re
import logging
from typing import List, Optional
from dataclasses import dataclass, field, asdict
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """搜索结果数据模型"""
    asin: str
    search_keyword: str
    search_rank: int
    title: str = ""
    brand: str = ""
    price: str = ""
    original_price: str = ""
    rating: Optional[float] = None
    rating_count: Optional[int] = None
    is_sponsored: bool = False
    is_amazon_choice: bool = False
    is_best_seller: bool = False
    url: str = ""
    image_url: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


class SearchParser:
    """搜索结果解析器"""

    def parse(self, html: str, keyword: str = "", page: int = 1) -> List[SearchResult]:
        """
        解析搜索结果页面

        Args:
            html: 页面HTML内容
            keyword: 搜索关键词
            page: 当前页码

        Returns:
            搜索结果列表
        """
        soup = BeautifulSoup(html, 'html.parser')
        results = []

        # 搜索结果容器选择器
        container_selectors = [
            'div[data-component-type="s-search-result"]',
            'div.s-result-item[data-asin]',
        ]

        items = []
        for selector in container_selectors:
            items = soup.select(selector)
            if items:
                break

        for idx, item in enumerate(items):
            try:
                asin = item.get('data-asin', '')
                if not asin or len(asin) != 10:
                    continue

                result = SearchResult(
                    asin=asin,
                    search_keyword=keyword,
                    search_rank=(page - 1) * 48 + idx + 1,
                    title=self._extract_title(item),
                    brand=self._extract_brand_v2(item),
                    price=self._extract_price(item),
                    original_price=self._extract_original_price(item),
                    rating=self._extract_rating(item),
                    rating_count=self._extract_rating_count(item),
                    is_sponsored=self._is_sponsored(item),
                    is_amazon_choice=self._has_badge(item, "Amazon's Choice"),
                    is_best_seller=self._has_badge(item, "Best Seller"),
                    url=f"https://www.amazon.com/dp/{asin}",
                    image_url=self._extract_image(item),
                )
                results.append(result)

            except Exception as e:
                logger.debug(f"解析搜索项失败: {e}")
                continue

        logger.info(f"解析搜索页 {page}，找到 {len(results)} 个商品")
        return results

    def _extract_title(self, item: BeautifulSoup) -> str:
        """提取完整标题"""
        # 方法1: 多个选择器尝试
        title_selectors = [
            'h2 a.a-link-normal span.a-text-normal',
            'h2 a span.a-text-normal',
            'h2.a-size-mini a span',
            'h2 a.a-link-normal',
            'span[data-component-type="s-product-title"] span',
        ]

        for selector in title_selectors:
            element = item.select_one(selector)
            if element:
                text = element.get_text(strip=True)
                if len(text) > 15:
                    return text

        # 方法2: 拼接 h2 下所有 span
        h2 = item.select_one('h2')
        if h2:
            spans = h2.select('span')
            if spans:
                full_title = ' '.join(span.get_text(strip=True) for span in spans if span.get_text(strip=True))
                if len(full_title) > 15:
                    return full_title
            return h2.get_text(strip=True)

        # 方法3: 从 aria-label 获取
        link = item.select_one('a.a-link-normal[href*="/dp/"]')
        if link and link.get('aria-label'):
            return link.get('aria-label')

        return ""

    def _extract_price(self, item: BeautifulSoup) -> str:
        """提取当前价格"""
        price_selectors = [
            'span.a-price span.a-offscreen',
            'span.a-price-whole',
            'span.a-color-price',
        ]

        for selector in price_selectors:
            element = item.select_one(selector)
            if element:
                price_text = element.get_text(strip=True)
                if '$' in price_text:
                    return price_text

        return ""

    def _extract_original_price(self, item: BeautifulSoup) -> str:
        """提取原价（划线价）"""
        selectors = [
            'span.a-price.a-text-price span.a-offscreen',
            'span[data-a-strike="true"] span.a-offscreen',
        ]

        for selector in selectors:
            element = item.select_one(selector)
            if element:
                return element.get_text(strip=True)

        return ""

    def _extract_rating(self, item: BeautifulSoup) -> Optional[float]:
        """提取评分"""
        rating_selectors = [
            'i.a-icon-star-small span.a-icon-alt',
            'i.a-icon-star span.a-icon-alt',
            'span.a-icon-alt',
        ]

        for selector in rating_selectors:
            element = item.select_one(selector)
            if element:
                text = element.get_text(strip=True)
                match = re.search(r'([\d.]+)\s*out of', text)
                if match:
                    try:
                        return float(match.group(1))
                    except ValueError:
                        pass

        # 从 aria-label 获取
        star_icon = item.select_one('i[class*="a-icon-star"]')
        if star_icon:
            aria = star_icon.get('aria-label', '')
            match = re.search(r'([\d.]+)\s*out of', aria)
            if match:
                try:
                    return float(match.group(1))
                except ValueError:
                    pass

        return None

    def _extract_rating_count(self, item: BeautifulSoup) -> Optional[int]:
        """提取评论数"""
        count_selectors = [
            'span.a-size-base.s-underline-text',
            'a[href*="#customerReviews"] span.a-size-base',
            'span[aria-label*="ratings"]',
        ]

        for selector in count_selectors:
            elements = item.select(selector)
            for element in elements:
                text = element.get_text(strip=True)
                clean_text = text.replace(',', '').replace('(', '').replace(')', '')
                if clean_text.isdigit():
                    return int(clean_text)

        # 从 aria-label 获取
        review_link = item.select_one('a[href*="#customerReviews"]')
        if review_link:
            aria = review_link.get('aria-label', '')
            match = re.search(r'([\d,]+)\s*rating', aria)
            if match:
                try:
                    return int(match.group(1).replace(',', ''))
                except ValueError:
                    pass

        return None

    def _is_sponsored(self, item: BeautifulSoup) -> bool:
        """检测是否为广告"""
        sponsored_selectors = [
            'span.s-label-popover-default',
            'span[data-component-type="s-ad-badge"]',
            'div.s-ad-badge',
            'span.puis-label-popover-default',
        ]

        for selector in sponsored_selectors:
            if item.select_one(selector):
                return True

        # 检查文本
        for span in item.select('span'):
            if span.get_text(strip=True).lower() == 'sponsored':
                return True

        return False

    def _has_badge(self, item: BeautifulSoup, badge_text: str) -> bool:
        """检测是否有特定徽章"""
        badges = item.select('span.a-badge-text, span.a-badge-label')
        for badge in badges:
            if badge_text.lower() in badge.get_text(strip=True).lower():
                return True
        return False

    def _extract_image(self, item: BeautifulSoup) -> str:
        """提取商品图片"""
        img = item.select_one('img.s-image')
        if img:
            return img.get('src', '')
        return ""

    def _extract_brand(self, item: BeautifulSoup) -> str:
        """提取品牌"""
        brand = ""

        brand_selectors = [
            'h5.s-line-clamp-1 a',
            'span.a-size-base-plus.a-color-base',
            'div.a-row.a-size-base.a-color-base',
            'a.a-size-base.a-link-normal',
        ]

        for selector in brand_selectors:
            element = item.select_one(selector)
            if element:
                text = element.get_text(strip=True)
                if text and len(text) < 50:
                    brand = text
                    break

        brand = re.sub(r'^(Visit the|Brand:?)\s*', '', brand, flags=re.I)
        brand = re.sub(r'\s*Store$', '', brand, flags=re.I)

        return brand.strip()

    def _is_valid_brand_text(self, text: str) -> bool:
        """验证是否为有效的品牌文本"""
        if not text:
            return False

        text_lower = text.lower().strip()

        skip_keywords = [
            'price', 'product page', 'free shipping', 'delivery', 'prime',
            'subscribe', 'save', 'deal', 'offer', 'coupon', 'promotion',
            'best seller', 'amazon\'s choice', 'sponsored', 'ad'
        ]

        if any(keyword in text_lower for keyword in skip_keywords):
            return False

        if '$' in text or '€' in text or '£' in text:
            return False

        if re.search(r'\d+\.\d{2}', text):
            return False

        if len(text) > 60:
            return False

        return True

    def _extract_brand_v2(self, item: BeautifulSoup) -> str:
        """提取品牌（改进版）"""
        brand = ""

        brand_selectors = [
            'h5.s-line-clamp-1 a[href*="/s?k="]',
            'h5.s-line-clamp-1 a',
            'a.a-size-base-plus.a-link-normal[href*="/s?k="]',
            'span.a-size-base-plus.a-color-base',
            'div.a-row.a-size-base.a-color-base a',
            'a.a-size-base.a-link-normal',
        ]

        for selector in brand_selectors:
            element = item.select_one(selector)
            if element:
                text = element.get_text(strip=True)
                if self._is_valid_brand_text(text):
                    brand = text
                    logger.debug(f"品牌提取成功 (选择器: {selector}): {brand}")
                    break
                else:
                    logger.debug(f"跳过无效品牌文本 (选择器: {selector}): {text}")

        brand = re.sub(r'^(Visit the|Brand:?)\s*', '', brand, flags=re.I)
        brand = re.sub(r'\s*Store$', '', brand, flags=re.I)

        return brand.strip()
