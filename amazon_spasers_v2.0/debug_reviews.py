"""
调试评论抓取问题
"""

import time
from pathlib import Path
from browser import BrowserManager


def debug_review_page():
    """调试评论页面"""

    # 使用有界面模式
    browser = BrowserManager(headless=False)
    driver = browser.create_driver()

    # 测试一个有评论的商品
    test_asin = "B000PS2XI4"  # 你可以换成其他 ASIN
    url = f"https://www.amazon.com/product-reviews/{test_asin}?pageNumber=1&sortBy=recent"

    print(f"\n访问评论页面: {url}")
    driver.get(url)
    time.sleep(5)

    # 保存页面截图
    debug_dir = Path("debug")
    debug_dir.mkdir(exist_ok=True)

    screenshot_path = debug_dir / f"review_page_{test_asin}.png"
    driver.save_screenshot(str(screenshot_path))
    print(f"📸 截图已保存: {screenshot_path}")

    # 保存 HTML
    html_path = debug_dir / f"review_page_{test_asin}.html"
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(driver.page_source)
    print(f"📄 HTML 已保存: {html_path}")

    # 检查页面内容
    page_source = driver.page_source

    print("\n" + "=" * 60)
    print("🔍 页面分析")
    print("=" * 60)

    # 检查是否需要登录
    if "Sign in" in page_source and "sign-in" in page_source.lower():
        print("⚠️ 检测到登录提示 - 可能需要登录才能查看评论")

    # 检查是否有验证码
    if "captcha" in page_source.lower() or "robot" in page_source.lower():
        print("⚠️ 检测到验证码")

    # 检查评论元素
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(page_source, 'html.parser')

    review_selectors = [
        ('div[data-hook="review"]', 'data-hook="review"'),
        ('div.review', 'class="review"'),
        ('div[id^="customer_review-"]', 'id="customer_review-"'),
        ('span[data-hook="review-body"]', 'data-hook="review-body"'),
        ('div.a-section.review', 'a-section review'),
    ]

    print("\n📋 评论元素检查:")
    for selector, desc in review_selectors:
        elements = soup.select(selector)
        status = "✅" if elements else "❌"
        print(f"  {status} {desc}: 找到 {len(elements)} 个")

    # 检查页面标题
    print(f"\n📰 页面标题: {driver.title}")

    # 检查 URL
    print(f"🔗 当前 URL: {driver.current_url}")

    print("\n" + "=" * 60)
    print("请查看浏览器和截图，确认问题原因")
    print("按回车键关闭浏览器...")
    print("=" * 60)

    input()
    browser.close()


if __name__ == '__main__':
    debug_review_page()
