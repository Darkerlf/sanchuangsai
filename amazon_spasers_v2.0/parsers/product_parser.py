"""
商品详情解析器 - 解析 Amazon 商品详情页面
"""

import re
import json
import logging
from typing import Optional, List, Dict, Tuple
from dataclasses import dataclass, field, asdict
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)


@dataclass
class ProductDetail:
    """商品详情数据模型"""
    asin: str
    title: str = ""
    brand: str = ""
    price: str = ""
    original_price: str = ""
    rating: Optional[float] = None
    rating_count: Optional[int] = None
    rating_distribution: Dict[str, int] = field(default_factory=dict)

    # ⭐ 新增：销量字段
    bought_count: str = ""              # 原始文本
    bought_count_number: int = 0        # 解析后的数字

    bsr_rank: Optional[int] = None
    bsr_category: str = ""
    sub_category_ranks: List[Dict] = field(default_factory=list)
    bullet_points: List[str] = field(default_factory=list)
    description: str = ""
    images: List[str] = field(default_factory=list)
    variants: List[Dict] = field(default_factory=list)
    seller: str = ""
    is_fba: bool = False
    has_aplus: bool = False
    first_available: str = ""
    availability: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


class ProductParser:
    """商品详情解析器"""

    def parse(self, html: str, asin: str = "") -> ProductDetail:
        """解析商品详情页面"""
        soup = BeautifulSoup(html, 'html.parser')

        product = ProductDetail(asin=asin)

        try:
            product.title = self._extract_title(soup)
            product.brand = self._extract_brand(soup)
            product.price = self._extract_price(soup)
            product.original_price = self._extract_original_price(soup)
            product.rating = self._extract_rating(soup)
            product.rating_count = self._extract_rating_count(soup)
            product.rating_distribution = self._extract_rating_distribution(soup)

            # ⭐ 新增：提取销量
            product.bought_count, product.bought_count_number = self._extract_bought_count(soup)

            product.bsr_rank, product.bsr_category = self._extract_bsr(soup)
            product.sub_category_ranks = self._extract_sub_category_ranks(soup)
            product.bullet_points = self._extract_bullet_points(soup)
            product.description = self._extract_description(soup)
            product.images = self._extract_images(soup)
            product.variants = self._extract_variants(soup)
            product.seller = self._extract_seller(soup)
            product.is_fba = self._is_fba(soup)
            product.has_aplus = self._has_aplus(soup)
            product.first_available = self._extract_first_available(soup)
            product.availability = self._extract_availability(soup)

            logger.debug(f"成功解析商品: {asin} - {product.title[:50] if product.title else 'No Title'}...")

            # ⭐ 记录销量
            if product.bought_count:
                logger.info(f"商品 {asin} 销量: {product.bought_count}")

        except Exception as e:
            logger.error(f"解析商品 {asin} 时出错: {e}")

        return product

    def _safe_select_one(self, soup: BeautifulSoup, selectors: List[str], attr: str = None) -> str:
        """安全选择器"""
        for selector in selectors:
            try:
                element = soup.select_one(selector)
                if element:
                    if attr:
                        return element.get(attr, "").strip()
                    return element.get_text(strip=True)
            except Exception:
                continue
        return ""

    def _extract_title(self, soup: BeautifulSoup) -> str:
        """提取标题"""
        selectors = ['#productTitle', 'h1#title span', 'h1.a-size-large']
        return self._safe_select_one(soup, selectors)

    def _extract_brand(self, soup: BeautifulSoup) -> str:
        """提取品牌"""
        selectors = ['#bylineInfo', 'a#bylineInfo', '.po-brand .po-break-word']
        brand = self._safe_select_one(soup, selectors)
        brand = re.sub(r'^(Visit the|Brand:?)\s*', '', brand, flags=re.I)
        brand = re.sub(r'\s*Store$', '', brand, flags=re.I)
        return brand.strip()

    def _extract_price(self, soup: BeautifulSoup) -> str:
        """提取价格"""
        selectors = [
            'span.a-price.aok-align-center span.a-offscreen',
            '#corePrice_feature_div span.a-offscreen',
            '#priceblock_ourprice',
            '#priceblock_dealprice',
            'span.a-price span.a-offscreen',
        ]
        return self._safe_select_one(soup, selectors)

    def _extract_original_price(self, soup: BeautifulSoup) -> str:
        """提取原价"""
        selectors = [
            'span.a-price.a-text-price span.a-offscreen',
            '.a-text-strike',
            'span[data-a-strike="true"] span.a-offscreen',
        ]
        return self._safe_select_one(soup, selectors)

    def _extract_rating(self, soup: BeautifulSoup) -> Optional[float]:
        """提取评分"""
        selectors = ['#acrPopover', '#averageCustomerReviews span.a-icon-alt']

        for selector in selectors:
            element = soup.select_one(selector)
            if element:
                text = element.get('title', '') or element.get_text(strip=True)
                match = re.search(r'([\d.]+)\s*out of', text)
                if match:
                    try:
                        return float(match.group(1))
                    except ValueError:
                        continue
        return None

    def _extract_rating_count(self, soup: BeautifulSoup) -> Optional[int]:
        """提取评论数"""
        selectors = ['#acrCustomerReviewText']

        for selector in selectors:
            element = soup.select_one(selector)
            if element:
                text = element.get_text(strip=True)
                match = re.search(r'([\d,]+)', text)
                if match:
                    try:
                        return int(match.group(1).replace(',', ''))
                    except ValueError:
                        continue
        return None

    def _extract_rating_distribution(self, soup: BeautifulSoup) -> Dict[str, int]:
        """提取评分分布"""
        distribution = {}

        rows = soup.select('#histogramTable tr, #cm_cr_dp_d_rating_histogram tr')
        for row in rows:
            try:
                star_cell = row.select_one('td.a-text-right a, td:first-child')
                if star_cell:
                    star_text = star_cell.get_text(strip=True)
                    star_match = re.search(r'(\d)', star_text)
                    if star_match:
                        star = f"{star_match.group(1)}_star"
                        pct_cell = row.select_one('td.a-text-right ~ td a, td:nth-child(3) a')
                        if pct_cell:
                            pct_text = pct_cell.get_text(strip=True)
                            pct_match = re.search(r'(\d+)%', pct_text)
                            if pct_match:
                                distribution[star] = int(pct_match.group(1))
            except Exception:
                continue

        if not distribution:
            histogram_rows = soup.select('table#histogramTable tr a, #reviewsMedley a[href*="filterByStar"]')
            for row in histogram_rows:
                aria = row.get('aria-label', '') or row.get('title', '')
                match = re.search(r'(\d)\s*star.*?(\d+)%', aria, re.I)
                if match:
                    star = f"{match.group(1)}_star"
                    distribution[star] = int(match.group(2))

        if not distribution:
            bars = soup.select('.a-histogram-row, div[class*="histogram"] div[class*="row"]')
            for bar in bars:
                try:
                    label = bar.select_one('a, span')
                    if label:
                        text = label.get_text(strip=True)
                        star_match = re.search(r'(\d)\s*star', text, re.I)
                        pct_match = re.search(r'(\d+)%', text)
                        if star_match and pct_match:
                            star = f"{star_match.group(1)}_star"
                            distribution[star] = int(pct_match.group(1))
                except Exception:
                    continue

        return distribution

    # ⭐ 新增：销量提取方法
    def _extract_bought_count(self, soup: BeautifulSoup) -> Tuple[str, int]:
        """
        提取销量数据

        Returns:
            Tuple[str, int]: (原始文本, 解析后的数字)
        """
        bought_text = ""
        bought_number = 0

        # 多种选择器
        selectors = [
            '#social-proofing-faceout-title-tk_bought .a-text-bold',
            '#social-proofing-faceout-title-tk_bought span',
            '#socialProofingAsinFaceout_feature_div .a-text-bold',
            '#socialProofingAsinFaceout_feature_div span.a-text-bold',
            'span[data-csa-c-content-id*="social-proofing"]',
            '#social-proofing-faceout-title-tk_bought',
            '.social-proofing-faceout-title-container',
        ]

        for selector in selectors:
            try:
                element = soup.select_one(selector)
                if element:
                    text = element.get_text(strip=True)
                    if 'bought' in text.lower():
                        bought_text = text
                        break
            except Exception:
                continue

        # 如果选择器没找到，搜索整个页面
        if not bought_text:
            spans = soup.find_all('span')
            for span in spans:
                text = span.get_text(strip=True)
                if 'bought in past month' in text.lower():
                    bought_text = text
                    break

            if not bought_text:
                html_text = str(soup)
                match = re.search(r'>(\d+[KkMm]?\+?\s*bought\s+in\s+past\s+month)<', html_text, re.I)
                if match:
                    bought_text = match.group(1)

        # 解析数字
        if bought_text:
            bought_number = self._parse_bought_number(bought_text)

        return bought_text, bought_number

    def _parse_bought_number(self, text: str) -> int:
        """解析销量数字"""
        if not text:
            return 0

        try:
            match = re.search(r'(\d+(?:\.\d+)?)\s*([KkMm])?\s*\+?', text)
            if match:
                number = float(match.group(1))
                unit = match.group(2)

                if unit:
                    unit = unit.upper()
                    if unit == 'K':
                        number *= 1000
                    elif unit == 'M':
                        number *= 1000000

                return int(number)
        except Exception:
            pass

        return 0

    def _extract_bsr(self, soup: BeautifulSoup) -> Tuple[Optional[int], str]:
        """提取BSR排名"""
        bsr_selectors = [
            '#productDetails_detailBullets_sections1 tr',
            '#detailBullets_feature_div li',
            'table.prodDetTable tr',
            '#productDetails_db_sections tr',
        ]

        for selector in bsr_selectors:
            rows = soup.select(selector)
            for row in rows:
                text = row.get_text()
                if 'Best Sellers Rank' in text or 'Amazon Best Sellers Rank' in text:
                    match = re.search(r'#?([\d,]+)\s+in\s+([^()\n]+)', text)
                    if match:
                        try:
                            rank = int(match.group(1).replace(',', ''))
                            category = match.group(2).strip()
                            return rank, category
                        except ValueError:
                            continue

        return None, ""

    def _extract_sub_category_ranks(self, soup: BeautifulSoup) -> List[Dict]:
        """提取子类目排名"""
        ranks = []

        bsr_links = soup.select('#SalesRank a, #productDetails_detailBullets_sections1 a[href*="/gp/bestsellers/"]')

        for link in bsr_links:
            try:
                parent_text = link.find_parent(['li', 'td', 'span'])
                if parent_text:
                    text = parent_text.get_text()
                    match = re.search(r'#?([\d,]+)\s+in\s+', text)
                    if match:
                        rank = int(match.group(1).replace(',', ''))
                        category = link.get_text(strip=True)
                        if category and rank:
                            ranks.append({
                                'rank': rank,
                                'category': category,
                                'url': link.get('href', '')
                            })
            except Exception:
                continue

        seen = set()
        unique_ranks = []
        for r in ranks:
            key = (r['rank'], r['category'])
            if key not in seen:
                seen.add(key)
                unique_ranks.append(r)

        return unique_ranks

    def _extract_bullet_points(self, soup: BeautifulSoup) -> List[str]:
        """提取商品要点"""
        bullets = []

        selectors = [
            '#feature-bullets li:not(.aok-hidden) span.a-list-item',
            '#feature-bullets ul li span',
            'div[id="feature-bullets"] li span',
        ]

        for selector in selectors:
            elements = soup.select(selector)
            for elem in elements:
                text = elem.get_text(strip=True)
                if text and len(text) > 5 and text not in bullets:
                    bullets.append(text)
            if bullets:
                break

        return bullets[:10]

    def _extract_description(self, soup: BeautifulSoup) -> str:
        """提取商品描述"""
        selectors = [
            '#productDescription p',
            '#productDescription',
            'div[data-feature-name="productDescription"]',
        ]

        for selector in selectors:
            element = soup.select_one(selector)
            if element:
                text = element.get_text(strip=True)
                if text:
                    return text[:2000]

        return ""

    def _extract_images(self, soup: BeautifulSoup) -> List[str]:
        """提取图片"""
        images = []

        img_selectors = [
            '#altImages img',
            '#imageBlock img',
            'li.image.item img',
        ]

        for selector in img_selectors:
            elements = soup.select(selector)
            for img in elements:
                src = img.get('src', '') or img.get('data-old-hires', '')
                if src and 'sprite' not in src and 'grey-pixel' not in src:
                    if '_SS40_' in src:
                        src = src.replace('_SS40_', '_SL1500_')
                    elif '_AC_US40_' in src:
                        src = src.replace('_AC_US40_', '_AC_SL1500_')

                    if src not in images:
                        images.append(src)

        if not images:
            main_img = soup.select_one('#landingImage, #imgBlkFront')
            if main_img:
                dynamic = main_img.get('data-a-dynamic-image', '')
                if dynamic:
                    try:
                        img_dict = json.loads(dynamic)
                        images = list(img_dict.keys())
                    except json.JSONDecodeError:
                        pass

        return images[:10]

    def _extract_variants(self, soup: BeautifulSoup) -> List[Dict]:
        """提取变体信息"""
        variants = []

        twister_selectors = [
            '#twister li[data-defaultasin]',
            '#variation_color_name li',
            '#variation_size_name li',
            '.swatchesSquare li',
        ]

        for selector in twister_selectors:
            elements = soup.select(selector)
            for elem in elements:
                try:
                    variant = {
                        'asin': elem.get('data-defaultasin', '') or elem.get('data-asin', ''),
                        'value': elem.get('title', '') or elem.get_text(strip=True),
                        'available': 'unavailable' not in elem.get('class', []),
                    }
                    if variant['asin'] or variant['value']:
                        variants.append(variant)
                except Exception:
                    continue

        return variants

    def _extract_seller(self, soup: BeautifulSoup) -> str:
        """提取卖家信息"""
        seller_selectors = [
            '#sellerProfileTriggerId',
            '#merchant-info a',
            '#tabular-buybox-truncate-1 span a',
            '#tabular-buybox span.tabular-buybox-text a',
            'div[data-feature-name="merchantName"] a',
        ]

        for selector in seller_selectors:
            element = soup.select_one(selector)
            if element:
                seller = element.get_text(strip=True)
                if seller:
                    return seller

        merchant_info = soup.select_one('#merchant-info, #tabular-buybox')
        if merchant_info:
            text = merchant_info.get_text()
            if 'Amazon.com' in text:
                return 'Amazon.com'

        return ""

    def _is_fba(self, soup: BeautifulSoup) -> bool:
        """检测是否FBA"""
        fba_indicators = [
            '#SSOFpopoverLink',
            '#deliveryMessageMirId',
        ]

        for selector in fba_indicators:
            element = soup.select_one(selector)
            if element:
                text = element.get_text().lower()
                if 'amazon' in text or 'fulfilled by amazon' in text:
                    return True

        if soup.select_one('#primeExclusiveBadge, i.a-icon-prime'):
            return True

        return False

    def _has_aplus(self, soup: BeautifulSoup) -> bool:
        """检测是否有A+内容"""
        aplus_selectors = [
            '#aplus',
            '#aplus_feature_div',
            '.aplus-v2',
            '#aplusBrandStory_feature_div',
        ]

        for selector in aplus_selectors:
            if soup.select_one(selector):
                return True
        return False

    def _extract_first_available(self, soup: BeautifulSoup) -> str:
        """提取首次上架日期"""
        detail_selectors = [
            '#productDetails_detailBullets_sections1 tr',
            '#detailBullets_feature_div li',
        ]

        for selector in detail_selectors:
            rows = soup.select(selector)
            for row in rows:
                text = row.get_text()
                if 'Date First Available' in text:
                    match = re.search(r'(\w+\s+\d+,?\s*\d{4})', text)
                    if match:
                        return match.group(1)

        return ""

    def _extract_availability(self, soup: BeautifulSoup) -> str:
        """提取库存状态"""
        element = soup.select_one('#availability span')
        if element:
            return element.get_text(strip=True)
        return ""
