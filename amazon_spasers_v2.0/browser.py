"""
浏览器管理器 - 处理 Selenium WebDriver 的创建和管理（稳定版）
"""

from __future__ import annotations

import logging
import random
import socket
import tempfile
from pathlib import Path
from typing import Optional

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.common.exceptions import TimeoutException, NoSuchElementException, WebDriverException
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from config import config, user_agents

logger = logging.getLogger(__name__)


def _pick_free_port(preferred: int = 9222, tries: int = 20) -> int:
    """尽量用 preferred，否则找一个空闲端口，避免 remote-debugging-port 冲突导致启动失败。"""
    candidates = [preferred] + list(range(preferred + 1, preferred + tries))
    for p in candidates:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(0.2)
            if s.connect_ex(("127.0.0.1", p)) != 0:
                return p
    # 最后兜底：让系统分配
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class BrowserManager:
    """浏览器管理器"""

    def __init__(self, headless: bool = False, persistent_profile: bool = True):
        """
        :param headless: 是否无头。None 则使用 config.HEADLESS
        :param persistent_profile: 是否使用持久化 profile（保持登录状态）。
                                  True：使用 config.BROWSER_DATA_DIR
                                  False：使用临时目录（更稳定，不保留登录）
        """
        self.headless = headless if headless is not None else config.HEADLESS
        self.persistent_profile = persistent_profile
        self.driver: Optional[webdriver.Chrome] = None

    def _build_options(self, user_data_dir: Path, clean_mode: bool = False) -> Options:
        """构建 Chrome Options。clean_mode=True 表示更干净、更少反检测参数，以提高启动成功率。"""
        options = Options()

        user_data_dir.mkdir(parents=True, exist_ok=True)
        options.add_argument(f"--user-data-dir={user_data_dir}")
        options.add_argument("--profile-directory=Default")

        # DevToolsActivePort 常见修复：指定远程调试端口（避免默认行为失败/端口冲突）
        port = _pick_free_port(9222)
        options.add_argument(f"--remote-debugging-port={port}")

        if self.headless:
            options.add_argument("--headless=new")

        options.add_argument(f"--window-size={config.WINDOW_SIZE[0]},{config.WINDOW_SIZE[1]}")
        options.add_argument("--lang=en-US")
        options.add_argument("--disable-gpu")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--disable-notifications")
        options.add_argument("--disable-infobars")

        # 更“干净”时尽量少动浏览器内核特征，提升稳定性
        if not clean_mode:
            options.add_argument("--disable-blink-features=AutomationControlled")

            # 随机 UA（只建议使用 Chrome UA）
            ua = random.choice(user_agents.AGENTS)
            options.add_argument(f"--user-agent={ua}")
            logger.debug(f"使用 User-Agent: {ua[:80]}")

            # 反自动化特征（可能导致某些版本不稳定；失败时会切 clean_mode 重试）
            options.add_experimental_option("excludeSwitches", ["enable-automation"])
            options.add_experimental_option("useAutomationExtension", False)

        # 密码提示/通知
        prefs = {
            "credentials_enable_service": False,
            "profile.password_manager_enabled": False,
            "profile.default_content_setting_values.notifications": 2,
        }
        options.add_experimental_option("prefs", prefs)

        # 可选：降低“页面加载等待”导致的卡死（看你业务，想稳就保留默认）
        # options.page_load_strategy = "eager"

        return options

    def _post_patch(self, driver: webdriver.Chrome) -> None:
        """注入脚本，减少被检测几率。"""
        try:
            driver.execute_cdp_cmd(
                "Page.addScriptToEvaluateOnNewDocument",
                {
                    "source": """
                        Object.defineProperty(navigator, 'webdriver', {get: () => undefined});
                        Object.defineProperty(navigator, 'plugins', {get: () => [1,2,3,4,5]});
                        Object.defineProperty(navigator, 'languages', {get: () => ['en-US','en']});
                        window.chrome = { runtime: {} };
                    """
                },
            )
        except Exception as e:
            logger.debug(f"注入反检测脚本失败（可忽略）: {e}")

    def create_driver(self) -> webdriver.Chrome:
        """
        创建并配置浏览器实例。
        启动策略：
        1) 首选：持久化 profile（如果启用 persistent_profile）
        2) 如果启动失败：使用更干净参数 clean_mode 重试
        3) 如果仍失败：降级到临时 profile（避免 profile 损坏/锁导致崩溃）
        """
        if self.driver is not None:
            return self.driver

        service = Service(ChromeDriverManager().install())

        # 方案 1：按配置选择 profile
        if self.persistent_profile:
            primary_profile = Path(config.BROWSER_DATA_DIR)
        else:
            primary_profile = Path(tempfile.mkdtemp(prefix="selenium_profile_"))

        attempts = [
            ("primary", primary_profile, False),
            ("primary_clean", primary_profile, True),
        ]

        # 方案 2：降级临时 profile（尤其适用于 DevToolsActivePort/profile lock/profile 损坏）
        fallback_profile = Path(tempfile.mkdtemp(prefix="selenium_profile_fallback_"))
        attempts.extend([
            ("fallback", fallback_profile, False),
            ("fallback_clean", fallback_profile, True),
        ])

        last_err: Optional[Exception] = None

        for name, profile_dir, clean_mode in attempts:
            try:
                options = self._build_options(profile_dir, clean_mode=clean_mode)

                logger.info(f"启动 Chrome: mode={name}, headless={self.headless}, profile={profile_dir}")
                driver = webdriver.Chrome(service=service, options=options)

                self._post_patch(driver)
                self.driver = driver
                logger.info("浏览器启动成功")
                return driver

            except WebDriverException as e:
                last_err = e
                logger.warning(f"启动失败({name}): {e.msg if hasattr(e, 'msg') else e}")
            except Exception as e:
                last_err = e
                logger.warning(f"启动失败({name}): {e}")

        # 全部失败
        raise RuntimeError(f"Chrome 启动失败，已重试 {len(attempts)} 次。最后错误: {last_err}")

    def get_driver(self) -> webdriver.Chrome:
        """获取浏览器实例（如果不存在则创建）"""
        if self.driver is None:
            return self.create_driver()
        return self.driver

    def close(self) -> None:
        """关闭浏览器"""
        if not self.driver:
            return
        try:
            self.driver.quit()
            logger.info("浏览器已关闭")
        except Exception as e:
            logger.warning(f"关闭浏览器时出错: {e}")
        finally:
            self.driver = None

    def wait_for_page_load(self, timeout: int = 10) -> None:
        """等待页面加载完成"""
        if not self.driver:
            return
        try:
            WebDriverWait(self.driver, timeout).until(
                lambda d: d.execute_script("return document.readyState") == "complete"
            )
        except TimeoutException:
            logger.warning("页面加载超时")

    def check_captcha(self) -> bool:
        """检测验证码"""
        if not self.driver:
            return False

        captcha_indicators = [
            "//input[@id='captchacharacters']",
            "//form[@action='/errors/validateCaptcha']",
            "//div[contains(@class, 'captcha')]",
            "//img[contains(@src, 'captcha')]",
        ]

        for xpath in captcha_indicators:
            try:
                self.driver.find_element(By.XPATH, xpath)
                logger.warning("⚠️ 检测到验证码！")
                return True
            except NoSuchElementException:
                continue
        return False

    def check_login_status(self) -> bool:
        if not self.driver:
            return False

        try:
            self.driver.get("https://www.amazon.com/")
            self.wait_for_page_load()

            # 1) 先看是否被重定向到登录/验证码页
            url = (self.driver.current_url or "").lower()
            if "ap/signin" in url:
                logger.warning("⚠️ Amazon 未登录（被重定向到登录页）")
                return False
            if "validatecaptcha" in url or self.check_captcha():
                logger.warning("⚠️ 遇到验证码页，无法确认登录状态")
                return False

            # 2) 读导航栏账户区域文本：已登录一般是 "Hello, Michael"，未登录是 "Hello, sign in"
            line1 = WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.ID, "nav-link-accountList-nav-line-1"))
            )
            text = (line1.text or "").strip().lower()

            is_logged_in = ("hello" in text) and ("sign in" not in text)

            if is_logged_in:
                logger.info(f"✅ Amazon 已登录（nav 文本: {line1.text!r}）")
            else:
                logger.warning(f"⚠️ Amazon 未登录（nav 文本: {line1.text!r}）")

            return is_logged_in

        except Exception as e:
            logger.error(f"检查登录状态时出错: {e}")
            return False

    def manual_login_guide(self) -> None:
        """引导用户手动登录"""
        if not self.driver:
            return
            
        print("\n" + "=" * 60)
        print("🔐 Amazon 手动登录")
        print("=" * 60)
        
        # 访问登录页面
        login_url = "https://www.amazon.com/ap/signin?openid.pape.max_auth_age=0&openid.return_to=https%3A%2F%2Fwww.amazon.com%2F&openid.identity=http%3A%2F%2Fspecs.openid.net%2Fauth%2F2.0%2Fidentifier_select&openid.assoc_handle=usflex&openid.mode=checkid_setup&openid.claimed_id=http%3A%2F%2Fspecs.openid.net%2Fauth%2F2.0%2Fidentifier_select&openid.ns=http%3A%2F%2Fspecs.openid.net%2Fauth%2F2.0"
        
        print("\n正在打开登录页面...")
        self.driver.get(login_url)
        
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
        if self.check_login_status():
            print("✅ 登录成功！")
            
            # 测试评论页面访问
            print("\n正在测试评论页面访问...")
            test_asin = "B000PS2XI4"
            self.driver.get(f"https://www.amazon.com/product-reviews/{test_asin}")
            self.wait_for_page_load()
            
            if "review" in self.driver.page_source.lower() and "ap/signin" not in self.driver.current_url:
                print("✅ 评论页面可以正常访问！")
                self.save_screenshot("login_success_review.png")
                print(f"📸 截图已保存: debug/login_success_review.png")
            else:
                print("⚠️ 评论页面访问异常，请检查截图")
                self.save_screenshot("login_review_check.png")
        else:
            print("❌ 登录可能未成功，请重试")
            
        print("\n" + "=" * 60)

    def scroll_page(self, scroll_pause: float = 0.5) -> None:
        """滚动页面以加载动态内容"""
        if not self.driver:
            return
        try:
            import time

            total_height = self.driver.execute_script("return document.body.scrollHeight")
            current_position = 0

            while current_position < total_height:
                current_position += random.randint(300, 600)
                self.driver.execute_script(f"window.scrollTo(0, {current_position});")
                time.sleep(random.uniform(0.2, scroll_pause))

                new_height = self.driver.execute_script("return document.body.scrollHeight")
                if new_height > total_height:
                    total_height = new_height

            self.driver.execute_script("window.scrollTo(0, 0);")
        except Exception as e:
            logger.debug(f"滚动页面时出错: {e}")

        # 在 browser.py 的 BrowserManager 类中添加
        def check_dog_page(self) -> bool:
            """检测是否遇到亚马逊变狗页 (503/404错误)"""
            if not self.driver:
                return False

            try:
                # 检查标题
                title = self.driver.title.lower()
                if "sorry" in title or "page not found" in title or "server busy" in title:
                    logger.warning("🐶 汪汪！检测到亚马逊狗狗页 (被反爬拦截)")
                    return True

                # 检查页面特定文本
                body_text = self.driver.find_element(By.TAG_NAME, "body").text.lower()
                if "sorry! something went wrong" in body_text or "we're sorry" in body_text:
                    logger.warning("🐶 汪汪！页面显示 Something went wrong")
                    return True

                return False
            except Exception:
                return False

    def save_screenshot(self, filename: str) -> None:
        """保存截图用于调试"""
        if not self.driver:
            return
        filepath = Path(config.DEBUG_DIR) / filename
        filepath.parent.mkdir(parents=True, exist_ok=True)
        self.driver.save_screenshot(str(filepath))
        logger.debug(f"截图已保存: {filepath}")

    def save_page_source(self, filename: str) -> None:
        """保存页面源码用于调试"""
        if not self.driver:
            return
        filepath = Path(config.DEBUG_DIR) / filename
        filepath.parent.mkdir(parents=True, exist_ok=True)
        filepath.write_text(self.driver.page_source, encoding="utf-8")
        logger.debug(f"页面源码已保存: {filepath}")

    def __enter__(self) -> "BrowserManager":
        self.create_driver()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
