"""
Excel 导出器
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any

from config import config

logger = logging.getLogger(__name__)


class ExcelExporter:
    """Excel 导出器"""

    def __init__(self, output_dir: str = None):
        self.output_dir = Path(output_dir or config.OUTPUT_DIR)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def export_to_excel(self, data_dict: Dict[str, List[Dict]], filename: str = "amazon_data.xlsx", brand: str = None):
        """
        导出多个数据集到 Excel（多sheet）

        Args:
            data_dict: {'sheet名': 数据列表}
            filename: 输出文件名
            brand: 品牌（可选）
        """
        try:
            import pandas as pd
        except ImportError:
            logger.error("需要安装 pandas: pip install pandas openpyxl")
            return

        if brand:
            filename = f"{brand}_{filename}"

        filepath = self.output_dir / filename

        with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
            for sheet_name, data in data_dict.items():
                if data:
                    # 处理嵌套数据
                    processed = self._flatten_data(data)
                    df = pd.DataFrame(processed)
                    df.to_excel(writer, sheet_name=sheet_name[:31], index=False)
                    logger.info(f"导出 sheet: {sheet_name} ({len(data)} 条)")

        logger.info(f"Excel 文件已保存: {filepath}")

    def export_to_csv(self, data: List[Dict], filename: str, brand: str = None):
        """
        导出数据到 CSV

        Args:
            data: 数据列表
            filename: 输出文件名
            brand: 品牌（可选）
        """
        if not data:
            logger.warning(f"没有数据可导出: {filename}")
            return

        if brand:
            filename = f"{brand}_{filename}"

        import csv

        filepath = self.output_dir / filename

        # 获取所有键
        all_keys = set()
        for item in data:
            all_keys.update(item.keys())

        # 排序键
        keys = sorted(all_keys)
        if 'asin' in keys:
            keys.remove('asin')
            keys = ['asin'] + keys

        with open(filepath, 'w', encoding='utf-8-sig', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()

            for item in data:
                row = {}
                for k, v in item.items():
                    if isinstance(v, (list, dict)):
                        row[k] = json.dumps(v, ensure_ascii=False)
                    else:
                        row[k] = v
                writer.writerow(row)

        logger.info(f"CSV 文件已保存: {filepath} ({len(data)} 条)")

    def _flatten_data(self, data: List[Dict]) -> List[Dict]:
        """展平嵌套数据"""
        processed = []

        for item in data:
            new_item = {}

            for key, value in item.items():
                if isinstance(value, list):
                    if key == 'bullet_points':
                        new_item[key] = ' | '.join(value[:5])
                    elif key == 'images':
                        new_item['image_count'] = len(value)
                        new_item['main_image'] = value[0] if value else ''
                    elif key == 'variants':
                        new_item['variant_count'] = len(value)
                    elif key == 'sub_category_ranks':
                        new_item['sub_category_count'] = len(value)
                        for i, rank in enumerate(value[:3]):
                            new_item[f'sub_rank_{i + 1}'] = rank.get('rank', '')
                            new_item[f'sub_category_{i + 1}'] = rank.get('category', '')
                    else:
                        new_item[key] = json.dumps(value, ensure_ascii=False)
                elif isinstance(value, dict):
                    if key == 'rating_distribution':
                        for star, pct in value.items():
                            new_item[f'rating_{star}'] = pct
                    else:
                        new_item[key] = json.dumps(value, ensure_ascii=False)
                else:
                    new_item[key] = value

            processed.append(new_item)

        return processed

    def generate_report(self, db_storage) -> Dict[str, Any]:
        """
        生成数据分析报告

        Args:
            db_storage: DatabaseStorage 实例

        Returns:
            报告数据
        """
        report = {
            'generated_at': __import__('datetime').datetime.now().isoformat(),
            'stats': db_storage.db.get_stats(),
            'brand_distribution': db_storage.get_brand_stats(),
            'price_distribution': db_storage.get_price_distribution(),
            'review_stats': db_storage.get_review_stats(),
        }

        # 保存报告
        report_path = self.output_dir / 'analysis_report.json'
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        logger.info(f"分析报告已保存: {report_path}")
        return report

    def export_brand_data(self, db_storage, brand: str):
        """
        导出指定品牌的数据

        Args:
            db_storage: DatabaseStorage 实例
            brand: 品牌名称
        """
        logger.info(f"开始导出品牌数据: {brand}")

        # 获取该品牌的数据
        search_results = db_storage.db.execute(
            "SELECT * FROM search_results WHERE brand = ?", (brand,)
        )
        products = db_storage.get_products_by_brand(brand)
        reviews = db_storage.get_reviews_by_brand(brand)
        merged = db_storage.get_merged_data(brand)

        # 导出 Excel
        self.export_to_excel({
            '搜索结果': search_results,
            '商品详情': products,
            '评论': reviews,
            '综合数据': merged,
        }, 'amazon_data.xlsx', brand=brand)

        # 导出 CSV
        self.export_to_csv(merged, 'merged_data.csv', brand=brand)

        # 生成品牌报告
        brand_progress = db_storage.get_brand_progress(brand)
        brand_report = {
            'brand': brand,
            'progress': brand_progress,
            'brand_stats': [b for b in db_storage.get_brand_stats() if b['brand'] == brand],
            'generated_at': __import__('datetime').datetime.now().isoformat(),
        }

        report_path = self.output_dir / f'{brand}_report.json'
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(brand_report, f, ensure_ascii=False, indent=2)

        logger.info(f"品牌 {brand} 数据导出完成")
