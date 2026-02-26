# Amazon 商品研究工具

基于 Python + Selenium 的 Amazon 商品数据爬虫，支持搜索结果、商品详情、用户评论的批量抓取和分析。

## 功能特性

- ✅ 三阶段爬取流程（搜索 → 商品详情 → 评论）
- ✅ 品牌过滤和分阶段爬取
- ✅ 断点续传支持
- ✅ 反爬策略（随机延迟、User-Agent 轮换、浏览器重启）
- ✅ 多种数据导出格式（SQLite、JSON、Excel、CSV）
- ✅ 数据统计分析

## 项目结构

```
amazon_spasers_v2.0/
├── main.py              # 主程序入口
├── config.py            # 配置管理
├── browser.py           # 浏览器管理
├── scrapers/            # 爬虫模块
├── parsers/             # 页面解析器
├── storage/             # 数据存储
├── data/                # 数据输出目录
├── logs/                # 日志文件
└── browser_data/        # 浏览器用户数据
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 首次运行

```bash
# 完整流程（搜索 + 商品 + 评论 + 导出）
python main.py all
```

首次运行时会提示手动登录 Amazon 账号。

### 3. 登录引导

程序会自动打开浏览器，引导您完成登录：
1. 输入 Amazon 邮箱和密码
2. 处理验证码（如有）
3. 确认登录成功后按回车继续

## 使用方法

### 基本命令

```bash
# 完整流程
python main.py all

# 仅搜索
python main.py search -k "McCook knife" --brand imarku

# 仅抓取商品详情
python main.py products --brand imarku

# 仅抓取评论
python main.py reviews -n 50 --brand imarku

# 导出数据
python main.py export
python main.py export --export-brand imarku

# 查看统计
python main.py stats
python main.py stats --brand imarku
```

### 按品牌分阶段爬取

```bash
# 第一阶段：搜索
python main.py brand --brand imarku --stage search

# 第二阶段：商品详情
python main.py brand --brand imarku --stage products

# 第三阶段：评论
python main.py brand --brand imarku --stage reviews
```

### 常用参数

| 参数 | 说明 | 示例 |
|------|------|------|
| `-k, --keywords` | 搜索关键词 | `-k "kitchen knife"` |
| `-a, --asins` | 指定 ASIN | `-a B000PS2XI4` |
| `-n, --top-n` | 评论抓取数量 | `-n 100` |
| `--brand` | 目标品牌 | `--brand imarku` |
| `--brands` | 多品牌列表 | `--brands imarku McCook` |
| `--headless` | 无头模式 | `--headless` |
| `--no-brand-filter` | 禁用品牌过滤 | `--no-brand-filter` |

## 配置说明

在 `config.py` 中可以修改以下配置：

### 搜索配置
- `SEARCH_KEYWORDS`: 默认搜索关键词
- `SEARCH_MAX_PAGES`: 最大搜索页数

### 品牌过滤
- `FILTER_BY_BRAND`: 是否启用品牌过滤
- `TARGET_BRANDS`: 目标品牌列表
- `BRAND_MATCH_MODE`: 匹配模式 (`exact`/`contains`)

### 反爬配置
- `PAGE_LOAD_DELAY`: 页面加载延迟范围
- `REQUEST_DELAY`: 请求延迟范围
- `MAX_RETRIES`: 最大重试次数

### 评论配置
- `REVIEWS_MAX_PAGES`: 最大评论页数
- `REVIEWS_TOP_N`: 抓取评论的商品数量

### 数据库配置
- `DATABASE_TYPE`: 数据库类型 (`sqlite`/`mysql`)
- MySQL 配置（如使用 MySQL）

## 数据说明

### 数据库表结构

| 表名 | 字段 |
|------|------|
| `search_results` | asin, brand, title, price, rating, search_rank |
| `products` | asin, title, price, rating, bsr_rank, bought_count |
| `reviews` | asin, rating, title, content, verified_purchase |

### 导出文件

- `amazon_data.xlsx`: Excel 格式（多 Sheet）
- `merged_data.csv`: CSV 格式合并数据
- `analysis_report.json`: 数据分析报告

## 反爬策略

1. **随机延迟**: 页面加载、请求、滚动均使用随机延迟
2. **User-Agent 轮换**: 模拟不同浏览器版本
3. **反自动化检测**: 注入脚本隐藏 webdriver 特征
4. **分批处理**: 每 5 个商品重启一次浏览器
5. **狗狗页检测**: 自动检测并等待解封
6. **验证码处理**: 检测到验证码时暂停等待手动处理

## 注意事项

1. 首次使用需要手动登录 Amazon 账号
2. 建议使用有头模式以便处理验证码
3. 抓取频率不要过高，避免被封
4. 大批量抓取时启用分批处理和断点续传
5. 浏览器数据保存在 `browser_data/` 目录，保持登录状态

## 常见问题

### Q: 如何重置数据库？
```bash
python reset.py
```

### Q: 如何清理浏览器数据？
删除 `browser_data/` 目录内容即可。

### Q: 登录后每次都需要重新登录？
浏览器数据损坏，删除 `browser_data/` 重新登录。

### Q: 遇到验证码怎么办？
程序会自动暂停，在浏览器中手动处理后按回车继续。

### Q: 如何查看抓取进度？
```bash
python main.py stats
python main.py stats --brand imarku
```

## 系统要求

- Python 3.8+
- Chrome 浏览器（最新版）
- Windows/Linux/macOS

## 许可证

MIT License

## 免责声明

本工具仅供学习和研究使用，请遵守 Amazon 的服务条款和 robots.txt 规定。
