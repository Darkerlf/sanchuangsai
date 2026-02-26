"""
手动登录脚本 - 登录后保持会话到浏览器数据目录
"""

import time
from pathlib import Path
from browser import BrowserManager


def manual_login():
    """手动登录并保持会话"""

    print("\n" + "=" * 60)
    print("🔐 Amazon 手动登录")
    print("=" * 60)

    # 使用非无头模式，显示浏览器
    browser = BrowserManager(headless=False)
    driver = browser.create_driver()

    # 访问 Amazon 首页（可能会自动跳转到登录）
    print("\n正在打开 Amazon...")
    driver.get("https://www.amazon.com")
    time.sleep(3)

    # 访问登录页面
    login_url = "https://www.amazon.com/ap/signin?openid.pape.max_auth_age=0&openid.return_to=https%3A%2F%2Fwww.amazon.com%2F&openid.identity=http%3A%2F%2Fspecs.openid.net%2Fauth%2F2.0%2Fidentifier_select&openid.assoc_handle=usflex&openid.mode=checkid_setup&openid.claimed_id=http%3A%2F%2Fspecs.openid.net%2Fauth%2F2.0%2Fidentifier_select&openid.ns=http%3A%2F%2Fspecs.openid.net%2Fauth%2F2.0"
    driver.get(login_url)

    print("\n" + "-" * 60)
    print("📝 请在浏览器中完成以下操作：")
    print("-" * 60)
    print("  1. 输入你的 Amazon 邮箱")
    print("  2. 输入密码")
    print("  3. 完成验证码（如果有）")
    print("  4. 完成二次验证（如果有）")
    print("  5. 确认看到 Amazon 首页")
    print("-" * 60)
    print("\n⏳ 完成登录后，按回车键继续...")
    input()

    # 验证登录状态
    print("\n正在验证登录状态...")
    driver.get("https://www.amazon.com")
    time.sleep(2)

    page_source = driver.page_source

    # 检查登录状态
    logged_in = False
    if "Hello, " in page_source:
        logged_in = True
    elif "Sign in" not in page_source or "Account & Lists" in page_source:
        logged_in = True

    if logged_in:
        print("\n✅ 登录成功！")

        # 测试评论页面
        print("\n正在测试评论页面访问...")
        test_asin = "B000PS2XI4"
        driver.get(f"https://www.amazon.com/product-reviews/{test_asin}")
        time.sleep(3)

        if "review" in driver.page_source.lower() and "ap/signin" not in driver.current_url:
            print("✅ 评论页面可以正常访问！")

            # 保存截图确认
            debug_dir = Path("debug")
            debug_dir.mkdir(exist_ok=True)
            driver.save_screenshot(str(debug_dir / "login_success_review.png"))
            print(f"📸 截图已保存: debug/login_success_review.png")
        else:
            print("⚠️ 评论页面访问异常，请检查截图")
            driver.save_screenshot(str(Path("debug") / "login_review_check.png"))
    else:
        print("\n❌ 登录可能未成功，请重试")

    print("\n" + "=" * 60)
    print("💡 会话已保存到 browser_data 目录")
    print("   现在可以运行爬虫: python main.py reviews")
    print("=" * 60)

    print("\n按回车键关闭浏览器...")
    input()

    browser.close()


if __name__ == '__main__':
    manual_login()
