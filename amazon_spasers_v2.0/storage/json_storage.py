"""
JSON 存储 - 文件存储
"""

import json
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Any, Dict

from config import config

logger = logging.getLogger(__name__)


class JsonStorage:
    """JSON 文件存储"""

    def __init__(self, output_dir: str = None):
        self.output_dir = Path(output_dir or config.OUTPUT_DIR)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def save(self, data: List[Any], filename: str, with_timestamp: bool = True):
        """
        保存数据到 JSON 文件

        Args:
            data: 数据列表
            filename: 文件名
            with_timestamp: 是否保存带时间戳的备份
        """
        if not data:
            logger.warning(f"没有数据可保存: {filename}")
            return

        # 转换为字典列表
        items = []
        for item in data:
            if hasattr(item, 'to_dict'):
                items.append(item.to_dict())
            elif isinstance(item, dict):
                items.append(item)
            else:
                items.append(str(item))

        # 保存主文件
        filepath = self.output_dir / filename
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(items, f, ensure_ascii=False, indent=2)

        logger.info(f"保存 JSON: {filepath} ({len(items)} 条)")

        # 保存带时间戳的备份
        if with_timestamp:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_name = filename.replace('.json', f'_{timestamp}.json')
            backup_path = self.output_dir / backup_name
            with open(backup_path, 'w', encoding='utf-8') as f:
                json.dump(items, f, ensure_ascii=False, indent=2)

    def load(self, filename: str) -> List[Dict]:
        """
        加载 JSON 文件

        Args:
            filename: 文件名

        Returns:
            数据列表
        """
        filepath = self.output_dir / filename

        if not filepath.exists():
            logger.warning(f"文件不存在: {filepath}")
            return []

        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        logger.info(f"加载 JSON: {filepath} ({len(data)} 条)")
        return data

    def append(self, data: Any, filename: str):
        """追加数据到 JSON 文件"""
        existing = self.load(filename)

        if hasattr(data, 'to_dict'):
            data = data.to_dict()

        if isinstance(data, list):
            existing.extend(data)
        else:
            existing.append(data)

        self.save(existing, filename, with_timestamp=False)

    def list_files(self, pattern: str = "*.json") -> List[Path]:
        """列出所有JSON文件"""
        return list(self.output_dir.glob(pattern))
