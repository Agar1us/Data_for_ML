import os
import sys
import shutil
import time
import json
import urllib.parse
from datetime import datetime, timezone
from pathlib import Path

from tqdm import tqdm
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.common.exceptions import SessionNotCreatedException
from selenium.common.exceptions import WebDriverException
from selenium.common.exceptions import NoSuchElementException
from selenium.common.exceptions import ElementNotInteractableException
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.support.ui import WebDriverWait
from tools.runtime import artifacts_root


_EMPTY_PAGE_RETRIES = 2


class Parser:
    def __init__(
        self,
        headless=True,
        chrome_binary_path=None,
        chromedriver_path=None,
        profile_dir=None,
    ):
        self.headless = headless
        self.chrome_binary_path = chrome_binary_path or self.__find_chrome_binary()
        self.chromedriver_path = chromedriver_path or self.__find_chromedriver()
        self.profile_dir = profile_dir
        self.last_debug_info: dict = {}

    def query_search(self,
                     query: str,
                     limit: int = 100,
                     delay: float = 6.0,
                     manual_captcha_timeout: float = 0.0,
                     size: str = None,
                     orientation: str = None,
                     image_type: str = None,
                     color: str = None,
                     image_format: str = None,
                     site: str = None) -> list:
        """
        Description
        ---------
        Implements the search function by query in Yandex Images.

        Args
        ---------
        **query:** str
                Query text
        **limit:** int
                Required (maximum) number of images
        **delay:** float
                Delay time between requests (sec)
        **size:** Size
                Size (large, small, medium)
        **orientation:** Orientation
                Orientation (horizontal, vertical, square)
        **image_type:** ImageType
                The type of images you are looking for (photos, faces, cliparts, ... )
        **color:** Color
                Color scheme (b/w, colored, orange, blue, ... )
        **image_format:** Format
                Format (jpg, png, gif)
        **site:** str
                The site where the images are located

        Return value
        ---------
        list: A list of URL to images matching the query.
        """

        params = {"text": query,
                  "isize": size,
                  "iorient": orientation,
                  "type": image_type,
                  "icolor": color,
                  "itype": image_format,
                  "site": site,
                  "nomisspell": 1,
                  "noreask": 1,
                  "p": 0}

        return self.__get_images(
            params=params,
            limit=limit,
            delay=delay,
            manual_captcha_timeout=manual_captcha_timeout,
        )

    def __get_images(self, params: dict, limit: int, delay: float, manual_captcha_timeout: float) -> list:
        """
        Description
        ---------
        Returns the specified number of direct URL to
        images corresponding to the request parameters.

        Parameters
        ---------
        **params:** dict
                Request parameters
        **limit:** int
                Required (maximum) number of images
        **delay:** float
                Delay time between requests (sec)

        Return value
        ---------
        list: A list of URL to images.
        """

        request_url = self.__build_search_url(params)
        self.last_debug_info = {
            "request_url": request_url,
            "query_params": params,
            "resolved_urls": 0,
            "driver_title": "",
            "driver_url": "",
            "debug_html_path": "",
            "debug_screenshot_path": "",
            "captcha_suspected": False,
            "wait_timed_out": False,
            "manual_captcha_waited": False,
            "status": "started",
            "profile_dir": self.profile_dir or "",
        }
        driver = None

        try:
            options = webdriver.ChromeOptions()

            if self.headless:
                options.add_argument("--headless=new")

            options.add_argument("--disable-gpu")
            options.add_argument("--disable-dev-shm-usage")
            options.add_argument("--no-sandbox")
            options.add_argument("--window-size=1600,1200")
            options.add_argument("--lang=en-US")

            if self.chrome_binary_path:
                options.binary_location = self.chrome_binary_path
            if self.profile_dir:
                resolved_profile_dir = str(Path(self.profile_dir).expanduser().resolve())
                Path(resolved_profile_dir).mkdir(parents=True, exist_ok=True)
                options.add_argument(f"--user-data-dir={resolved_profile_dir}")

            service = Service(executable_path=self.chromedriver_path)
            driver = webdriver.Chrome(service=service, options=options)

        except SessionNotCreatedException as e:
            raise RuntimeError(
                f"Chrome bootstrap failed: {e.msg}"
            ) from e
        except FileNotFoundError as e:
            raise RuntimeError(f"Chrome bootstrap failed: {e}") from e

        try:
            driver.get(request_url)
        except WebDriverException as e:
            raise RuntimeError(f"WebDriver navigation failed: {e.msg}") from e

        images = []
        html = ""
        try:
            try:
                WebDriverWait(driver, max(10, int(delay * 3))).until(
                    lambda current_driver: current_driver.execute_script("return document.readyState") == "complete"
                )
            except TimeoutException:
                self.last_debug_info["wait_timed_out"] = True

            self.__maybe_wait_for_manual_captcha_resolution(driver, delay=delay, manual_captcha_timeout=manual_captcha_timeout)
            time.sleep(delay)
            pbar = tqdm(total=limit)

            while True:
                html = driver.page_source
                images = self.__parse_html(html)

                if len(images) == 0:
                    # Give the dynamic page a few extra chances to render results before declaring failure.
                    recovered = False
                    for _ in range(_EMPTY_PAGE_RETRIES):
                        time.sleep(delay)
                        html = driver.page_source
                        images = self.__parse_html(html)
                        if images:
                            recovered = True
                            break
                    if not recovered:
                        pbar.set_postfix_str("Something went wrong... no images found.")
                        break

                pbar.n = len(images) if len(images) <= limit else limit
                pbar.refresh()

                if len(images) >= limit:
                    break

                old_page_height = driver.execute_script("return document.body.scrollHeight")
                driver.execute_script("window.scrollTo(0, document.body.scrollHeight)")
                time.sleep(delay)
                new_page_height = driver.execute_script("return document.body.scrollHeight")

                if old_page_height == new_page_height:
                    try:
                        button = driver.find_element(
                            By.XPATH,
                            "//div[starts-with(@class, 'FetchListButton')]//button[starts-with(@class, 'Button2')]",
                        )
                        button.click()
                        time.sleep(delay)
                    except NoSuchElementException:
                        break
                    except ElementNotInteractableException:
                        pbar.set_postfix_str("Fewer images found")
                        break
                    except Exception:
                        break
        finally:
            if driver is not None:
                self.last_debug_info["driver_title"] = driver.title
                self.last_debug_info["driver_url"] = driver.current_url
                self.last_debug_info["resolved_urls"] = int(len(images))
                self.last_debug_info["captcha_suspected"] = self.__captcha_suspected(html, driver.title, driver.current_url)
                if self.last_debug_info["captcha_suspected"]:
                    self.last_debug_info["status"] = "captcha_blocked"
                elif len(images) == 0:
                    self.last_debug_info["status"] = "no_results"
                else:
                    self.last_debug_info["status"] = "ok"
                if len(images) == 0 and html:
                    debug_paths = self.__save_debug_snapshot(driver, html, params)
                    self.last_debug_info["debug_html_path"] = debug_paths["html"]
                    self.last_debug_info["debug_screenshot_path"] = debug_paths["screenshot"]
                driver.quit()

        return images[:limit]

    def __maybe_wait_for_manual_captcha_resolution(self, driver, delay: float, manual_captcha_timeout: float) -> None:
        if self.headless or manual_captcha_timeout <= 0:
            return
        if not self.__captcha_suspected(driver.page_source, driver.title, driver.current_url):
            return

        print(
            "Captcha detected. Solve it in the opened browser window. "
            f"Waiting up to {manual_captcha_timeout:.0f} seconds..."
        )
        self.last_debug_info["manual_captcha_waited"] = True
        deadline = time.time() + manual_captcha_timeout
        while time.time() < deadline:
            if not self.__captcha_suspected(driver.page_source, driver.title, driver.current_url):
                time.sleep(delay)
                return
            time.sleep(1.0)

    def __captcha_suspected(self, html: str, title: str, current_url: str) -> bool:
        haystack = " ".join([html or "", title or "", current_url or ""]).casefold()
        markers = [
            "captcha",
            "smartcaptcha",
            "are you robot",
            "are you a robot",
            "доступ ограничен",
            "подтвердите, что запросы отправляли вы",
            "робот",
        ]
        return any(marker in haystack for marker in markers)

    def __debug_root(self) -> Path:
        return artifacts_root() / "yandex_debug"

    def __save_debug_snapshot(self, driver, html: str, params: dict) -> dict[str, str]:
        debug_root = self.__debug_root()
        debug_root.mkdir(parents=True, exist_ok=True)
        query_text = params.get("text") or params.get("url") or "yandex"
        slug = self.__slugify(query_text)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        debug_path = debug_root / f"{timestamp}_{slug}.html"
        screenshot_path = debug_root / f"{timestamp}_{slug}.png"
        debug_path.write_text(html, encoding="utf-8")
        try:
            driver.save_screenshot(str(screenshot_path))
            screenshot = str(screenshot_path)
        except Exception:
            screenshot = ""
        return {"html": str(debug_path), "screenshot": screenshot}

    def __slugify(self, value: str) -> str:
        sanitized = "".join(char.lower() if char.isalnum() else "_" for char in str(value).strip())
        compact = "_".join(part for part in sanitized.split("_") if part)
        return compact or "yandex"

    def __find_chrome_binary(self):
        env_binary = os.environ.get("CHROME_BIN")
        candidate_paths = [
            env_binary,
            shutil.which("google-chrome"),
            shutil.which("google-chrome-stable"),
            shutil.which("chromium-browser"),
            shutil.which("chromium"),
            "/usr/bin/google-chrome",
            "/usr/bin/google-chrome-stable",
            "/usr/bin/chromium-browser",
            "/usr/bin/chromium",
            "/snap/bin/chromium",
            "/mnt/c/Program Files/Google/Chrome/Application/chrome.exe",
            "/mnt/c/Program Files (x86)/Google/Chrome/Application/chrome.exe",
        ]

        for candidate in candidate_paths:
            if candidate and os.path.exists(candidate):
                return candidate

        raise FileNotFoundError(
            "Chrome/Chromium binary not found. Set CHROME_BIN or install chromium/chrome inside WSL2."
        )

    def __find_chromedriver(self):
        env_driver = os.environ.get("CHROMEDRIVER")
        candidate_paths = [
            env_driver,
            shutil.which("chromedriver"),
            "/usr/bin/chromedriver",
            "/usr/local/bin/chromedriver",
        ]

        for candidate in candidate_paths:
            if candidate and os.path.exists(candidate):
                return candidate

        raise FileNotFoundError(
            "chromedriver not found. Set CHROMEDRIVER or install chromedriver inside WSL2."
        )

    def __build_search_url(self, params: dict) -> str:
        clean_params = {key: value for key, value in params.items() if value is not None}
        return "https://yandex.ru/images/search?" + urllib.parse.urlencode(clean_params)

    def __parse_html(self, html: str) -> list:
        """
        Description
        ---------
        Extracts direct links to images from the html code of the page.

        Args
        ---------
        **html:** str
                html code of the page with images.

        Return value
        ---------
        list: A list of direct URL to images from the page.
        """

        soup = BeautifulSoup(html, "lxml")
        pictures_place = soup.find("div", {"class": "SerpList"})

        if pictures_place is not None:
            urls = []
            pictures = pictures_place.find_all("div", {"class": "SerpItem"})

            for pic in pictures:
                image_url = self.__extract_serpitem_image_url(pic)
                if image_url:
                    urls.append(image_url)

            return urls

        else:
            pictures_place = soup.find("div", {"class": "cbir-page-layout__main-content"})
            urls = []
            try:
                pictures = pictures_place.find_all("div", {"class": "serp-item"})

                for pic in pictures:
                    data = json.loads(pic.get("data-bem"))
                    image = data['serp-item']['img_href']
                    urls.append(image)

                return urls
            except AttributeError:
                return urls

    def __extract_serpitem_image_url(self, pic) -> str | None:
        direct_link = pic.select_one('a[href*="img_url="]')
        if direct_link is not None:
            href = direct_link.get("href") or ""
            parsed = urllib.parse.urlparse(href)
            query = urllib.parse.parse_qs(parsed.query)
            image_urls = query.get("img_url") or query.get("url")
            if image_urls:
                return image_urls[0]

        for tag in pic.find_all(attrs={"data-bem": True}):
            raw_data = tag.get("data-bem")
            if not raw_data:
                continue
            try:
                data = json.loads(raw_data)
            except (TypeError, json.JSONDecodeError):
                continue

            serp_item_data = data.get("serp-item") if isinstance(data, dict) else None
            if isinstance(serp_item_data, dict):
                image_url = serp_item_data.get("img_href") or serp_item_data.get("preview", {}).get("url")
                if image_url:
                    return image_url

        image_tag = pic.find("img")
        if image_tag is not None:
            return image_tag.get("src") or image_tag.get("data-src")

        return None
