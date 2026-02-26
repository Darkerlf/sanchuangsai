"""
配置文件 - 集中管理所有配置项（重写版）
- 统一 BASE_DIR
- 路径使用 pathlib.Path
- 支持环境变量覆盖
- 自动创建必要目录
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any


def _env_bool(name: str, default: bool) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return v.strip().lower() in {"1", "true", "yes", "y", "on"}


def _env_int(name: str, default: int) -> int:
    v = os.getenv(name)
    if v is None:
        return default
    try:
        return int(v.strip())
    except ValueError:
        return default


def _env_str(name: str, default: str) -> str:
    v = os.getenv(name)
    return default if v is None else v


def _project_base_dir() -> Path:
    """
    计算项目根目录：
    - 优先用环境变量 PROJECT_BASE_DIR
    - 否则用当前文件所在目录
    """
    override = os.getenv("PROJECT_BASE_DIR")
    if override:
        return Path(override).expanduser().resolve()
    return Path(__file__).resolve().parent


@dataclass(slots=True)
class ScraperConfig:
    """爬虫配置"""

    # ==================== 项目根目录 ====================
    BASE_DIR: Path = field(default_factory=_project_base_dir)

    # ==================== 搜索配置 ====================
    SEARCH_KEYWORDS: List[str] = field(default_factory=lambda: [
        "PAUDIN knife",
        "PAUDIN knife set",
        "PAUDIN kitchen knife",
    ])
    SEARCH_MAX_PAGES: int = 100

    # ==================== 品牌过滤配置 ====================
    FILTER_BY_BRAND: bool = False
    TARGET_BRANDS: List[str] = field(default_factory=lambda: ["McCook"])
    BRAND_MATCH_MODE: str = "contains"  # "exact" / "contains"

    # ==================== 品牌管理配置 ====================
    BRANDS: List[str] = field(default_factory=lambda: ["imarku", "McCook", "PAUDIN"])
    CURRENT_BRAND: str = ""  # 空表示爬取所有品牌

    # ==================== 爬取阶段配置 ====================
    SCRAPE_STAGES: List[str] = field(default_factory=lambda: ["search", "products", "reviews"])
    SAVE_PROGRESS: bool = True
    PROGRESS_FILE: str = "scrape_progress.json"

    # ==================== 延迟配置（秒）====================
    PAGE_LOAD_DELAY: Tuple[float, float] = (2.0, 4.0)
    REQUEST_DELAY: Tuple[float, float] = (1.0, 3.0)
    SCROLL_DELAY: Tuple[float, float] = (0.3, 0.8)

    # ==================== 重试配置 ====================
    MAX_RETRIES: int = 5
    RETRY_DELAY: Tuple[float, float] = (3.0, 6.0)

    # ==================== 评论配置 ====================
    REVIEWS_MAX_PAGES: int = 10
    REVIEWS_TOP_N: int = 100

    # ==================== 浏览器配置 ====================
    HEADLESS: bool = field(default_factory=lambda: _env_bool("SCRAPER_HEADLESS", False))
    WINDOW_SIZE: Tuple[int, int] = (1920, 1080)

    # ==================== 数据库配置 ====================
    DATABASE_TYPE: str = field(default_factory=lambda: _env_str("DATABASE_TYPE", "sqlite"))
    MYSQL_HOST: str = field(default_factory=lambda: _env_str("MYSQL_HOST", "localhost"))
    MYSQL_PORT: int = field(default_factory=lambda: _env_int("MYSQL_PORT", 3306))
    MYSQL_USER: str = field(default_factory=lambda: _env_str("MYSQL_USER", "root"))
    MYSQL_PASSWORD: str = field(default_factory=lambda: _env_str("MYSQL_PASSWORD", "1234"))
    MYSQL_DATABASE: str = field(default_factory=lambda: _env_str("MYSQL_DATABASE", "amazon_scraper"))

    # ==================== 路径配置（Path 类型）====================
    OUTPUT_DIR: Path = field(init=False)
    LOG_DIR: Path = field(init=False)
    DEBUG_DIR: Path = field(init=False)
    BROWSER_DATA_DIR: Path = field(init=False)
    DATABASE_PATH: Path = field(init=False)

    # ==================== 日志配置 ====================
    LOG_LEVEL: str = field(default_factory=lambda: _env_str("LOG_LEVEL", "INFO"))
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    def __post_init__(self) -> None:
        # 统一在这里根据 BASE_DIR 生成路径，避免“默认值静态求值”坑
        self.OUTPUT_DIR = self.BASE_DIR / "data"
        self.LOG_DIR = self.BASE_DIR / "logs"
        self.DEBUG_DIR = self.BASE_DIR / "debug"
        self.BROWSER_DATA_DIR = self.BASE_DIR / "browser_data"
        self.DATABASE_PATH = self.OUTPUT_DIR / "amazon_data.db"

        self.ensure_dirs()

    def ensure_dirs(self) -> None:
        """创建必要目录"""
        for p in [self.OUTPUT_DIR, self.LOG_DIR, self.DEBUG_DIR, self.BROWSER_DATA_DIR]:
            p.mkdir(parents=True, exist_ok=True)

    def as_dict(self) -> Dict[str, Any]:
        """便于打印/调试：把 Path 转成 str"""
        d: Dict[str, Any] = {}
        for k, v in self.__dict__.items():
            if isinstance(v, Path):
                d[k] = str(v)
            else:
                d[k] = v
        return d


@dataclass(slots=True)
class UserAgents:
    """
    User-Agent 列表
    注意：你用的是 ChromeDriver + Chrome，建议只用 Chrome UA，避免站点返回不同浏览器脚本导致异常。
    """
    AGENTS: List[str] = field(default_factory=lambda: [
        # 尽量与你机器 Chrome 主版本接近（你现在 driver 看起来是 142）
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/142.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/141.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/140.0.0.0 Safari/537.36",
    ])


# 全局配置实例（保持你的原有 import 方式兼容）
config = ScraperConfig()
user_agents = UserAgents()
