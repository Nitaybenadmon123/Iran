import json
import random
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# Default cookies path
DEFAULT_COOKIES_PATH = Path(__file__).parent.parent / "x_cookies.json"


def _human_delay(min_s=0.5, max_s=2.0):
    """Random delay to mimic human behavior."""
    time.sleep(random.uniform(min_s, max_s))


def _human_type(element, text, min_delay=0.05, max_delay=0.15):
    """Type text character by character like a human."""
    for char in text:
        element.send_keys(char)
        time.sleep(random.uniform(min_delay, max_delay))


def _create_driver(headless=True):
    """Create a Chrome driver that looks like a real browser."""
    options = Options()
    if headless:
        options.add_argument("--headless=new")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    options.add_argument("--window-size=1920,1080")
    options.add_argument("--lang=en-US")
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_experimental_option("useAutomationExtension", False)
    options.add_argument(
        "user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )

    driver = webdriver.Chrome(options=options)

    # Remove webdriver flag to avoid detection
    driver.execute_cdp_cmd("Page.addScriptToEvaluateOnNewDocument", {
        "source": """
            Object.defineProperty(navigator, 'webdriver', {get: () => undefined});
            Object.defineProperty(navigator, 'languages', {get: () => ['en-US', 'en']});
            Object.defineProperty(navigator, 'plugins', {get: () => [1, 2, 3, 4, 5]});
        """
    })

    return driver


def _human_scroll(driver, distance=None):
    """Scroll like a human — variable distance with slight randomness."""
    if distance is None:
        distance = random.randint(400, 900)
    driver.execute_script(f"window.scrollBy(0, {distance});")
    _human_delay(0.3, 0.8)


def login_twitter(driver, username, password):
    """
    Log into Twitter/X via the real login flow.
    Returns True if login succeeded.
    """
    print("  [login] Navigating to login page...")
    driver.get("https://x.com/i/flow/login")
    _human_delay(3, 5)

    # Step 1: Enter username
    try:
        print("  [login] Entering username...")
        username_input = WebDriverWait(driver, 15).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, 'input[autocomplete="username"]'))
        )
        _human_delay(0.5, 1.5)
        username_input.click()
        _human_delay(0.3, 0.7)
        _human_type(username_input, username)
        _human_delay(0.5, 1.0)

        # Click Next button
        next_btn = driver.find_element(
            By.XPATH, "//span[text()='Next']/ancestor::button | //span[text()='הבא']/ancestor::button"
        )
        next_btn.click()
        _human_delay(2, 4)
    except Exception as e:
        print(f"  [login] Username step failed: {e}")
        return False

    # Step 2: Handle possible phone/email verification prompt
    try:
        verify_input = WebDriverWait(driver, 5).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, 'input[data-testid="ocfEnterTextTextInput"]'))
        )
        print("  [login] Verification prompt detected - entering username as confirmation...")
        _human_type(verify_input, username)
        _human_delay(0.5, 1.0)
        next_btn2 = driver.find_element(
            By.XPATH, "//span[text()='Next']/ancestor::button | //span[text()='הבא']/ancestor::button"
        )
        next_btn2.click()
        _human_delay(2, 3)
    except Exception:
        pass  # No verification step needed

    # Step 3: Enter password
    try:
        print("  [login] Entering password...")
        password_input = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, 'input[name="password"]'))
        )
        _human_delay(0.5, 1.0)
        password_input.click()
        _human_delay(0.3, 0.7)
        _human_type(password_input, password)
        _human_delay(0.5, 1.5)

        # Click Log in button
        login_btn = driver.find_element(
            By.XPATH, "//span[text()='Log in']/ancestor::button | //span[text()='התחברות']/ancestor::button"
        )
        login_btn.click()
        _human_delay(4, 6)
    except Exception as e:
        print(f"  [login] Password step failed: {e}")
        return False

    # Verify login succeeded
    try:
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, 'a[data-testid="AppTabBar_Home_Link"]'))
        )
        print("  [login] Login successful!")
        return True
    except Exception:
        body = driver.find_element(By.TAG_NAME, "body").text[:300]
        print(f"  [login] Login may have failed. Page text: {body[:200]}")
        # Check if we're on the home timeline anyway
        if "Home" in body[:100]:
            print("  [login] Appears logged in (Home visible)")
            return True
        return False


def scrape_twitter_profile(username: str):
    url = f"https://twitter.com/{username}"

    driver = _create_driver()
    driver.get(url)
    time.sleep(5)

    for _ in range(4):
        driver.execute_script("window.scrollBy(0, 500);")
        time.sleep(1)

    data = {
        "user_name": username, "name": "", "bio": "", "location": "",
        "url": "", "joined_date": "", "followers": "", "following": "",
        "verified": False, "profile_url": url,
    }

    try:
        name_el = driver.find_element(By.XPATH, '//div[@data-testid="UserName"]//span[1]')
        data["name"] = name_el.text.strip()
    except: pass
    try:
        bio_el = driver.find_element(By.XPATH, "//div[@data-testid='UserDescription']")
        data["bio"] = bio_el.text.strip()
    except: pass
    try:
        loc_el = driver.find_element(By.XPATH, "//span[@data-testid='UserLocation']")
        data["location"] = loc_el.text.strip()
    except: pass
    try:
        url_el = driver.find_element(By.XPATH, "//a[@data-testid='UserUrl']")
        data["url"] = url_el.get_attribute("href").strip()
    except: pass
    try:
        join_el = driver.find_element(By.XPATH, "//span[contains(text(),'Joined')]")
        data["joined_date"] = join_el.text.replace("Joined", "").strip()
    except: pass
    try:
        img_el = driver.find_element(By.XPATH, "//img[contains(@alt,'Image') or contains(@src,'profile_images')]")
        data["profile_image"] = img_el.get_attribute("src")
    except: pass

    for _ in range(6):
        counts = driver.execute_script("""
            let result = {};
            let fwing = document.querySelector("a[href$='/following'] span span");
            let fwers = document.querySelector("a[href$='/followers'] span span");
            result.following = fwing ? fwing.textContent.trim() : '';
            result.followers = fwers ? fwers.textContent.trim() : '';
            return result;
        """)
        if counts.get("followers") or counts.get("following"):
            data["followers"] = counts.get("followers", "")
            data["following"] = counts.get("following", "")
            break
        time.sleep(1)

    if not data["followers"] or not data["following"]:
        body_text = driver.find_element(By.TAG_NAME, "body").text
        m_followers = re.search(r'(\d[\d,]*) Followers', body_text)
        m_following = re.search(r'(\d[\d,]*) Following', body_text)
        if m_followers: data["followers"] = m_followers.group(1)
        if m_following: data["following"] = m_following.group(1)

    data["verified"] = "Verified account" in driver.page_source
    driver.quit()
    return data


def _parse_engagement_count(text):
    """Parse engagement count strings like '1.2K', '3M', '42' to int."""
    if not text:
        return 0
    text = text.strip().replace(",", "")
    try:
        if text.upper().endswith("K"):
            return int(float(text[:-1]) * 1000)
        elif text.upper().endswith("M"):
            return int(float(text[:-1]) * 1_000_000)
        return int(text)
    except (ValueError, TypeError):
        return 0


def _extract_tweets_from_page(driver):
    """Extract all visible tweet data from the current page state."""
    tweets = []
    articles = driver.find_elements(By.CSS_SELECTOR, 'article[data-testid="tweet"]')

    for article in articles:
        tweet = {}

        try:
            text_el = article.find_element(By.CSS_SELECTOR, 'div[data-testid="tweetText"]')
            tweet["text"] = text_el.text.strip()
        except Exception:
            tweet["text"] = ""

        try:
            time_el = article.find_element(By.CSS_SELECTOR, "time")
            tweet["created_at"] = time_el.get_attribute("datetime")
            link_el = time_el.find_element(By.XPATH, "./..")
            href = link_el.get_attribute("href") or ""
            m = re.search(r"/status/(\d+)", href)
            tweet["tweet_id"] = m.group(1) if m else ""
            tweet["tweet_url"] = href
        except Exception:
            tweet["created_at"] = ""
            tweet["tweet_id"] = ""
            tweet["tweet_url"] = ""

        try:
            groups = article.find_elements(By.CSS_SELECTOR, 'div[role="group"] button')
            counts = []
            for btn in groups:
                aria = btn.get_attribute("aria-label") or ""
                m = re.search(r"([\d,.]+[KMkm]?)", aria)
                counts.append(_parse_engagement_count(m.group(1)) if m else 0)
            tweet["replies"] = counts[0] if len(counts) > 0 else 0
            tweet["retweets"] = counts[1] if len(counts) > 1 else 0
            tweet["likes"] = counts[2] if len(counts) > 2 else 0
        except Exception:
            tweet["replies"] = 0
            tweet["retweets"] = 0
            tweet["likes"] = 0

        try:
            social_ctx = article.find_elements(By.CSS_SELECTOR, 'span[data-testid="socialContext"]')
            tweet["is_retweet"] = any("reposted" in s.text.lower() or "retweeted" in s.text.lower()
                                      for s in social_ctx)
        except Exception:
            tweet["is_retweet"] = False

        if tweet.get("tweet_id"):
            tweets.append(tweet)

    return tweets


def scrape_user_tweets(username, since_date="2022-05-01", until_date="2026-01-01",
                       max_scrolls=300, scroll_pause=1.5, driver=None):
    """
    Scrape tweets from a user's timeline within a date range.
    Accepts a pre-logged-in driver to reuse the session across users.

    Parameters
    ----------
    username : str
        Twitter username (without @).
    since_date, until_date : str
        Date range in YYYY-MM-DD format.
    max_scrolls : int
        Maximum scroll iterations.
    scroll_pause : float
        Base seconds between scrolls (randomized +/- 30%).
    driver : webdriver.Chrome or None
        Pre-logged-in driver. If None, creates a new one (no login).

    Returns
    -------
    list[dict]
        List of tweet dicts.
    """
    since_dt = datetime.strptime(since_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    until_dt = datetime.strptime(until_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)

    own_driver = driver is None
    if own_driver:
        driver = _create_driver()

    # Navigate to user profile
    url = f"https://x.com/{username}"
    driver.get(url)
    _human_delay(3, 5)

    # Wait for tweets to appear (or detect errors)
    for wait_attempt in range(3):
        articles = driver.find_elements(By.CSS_SELECTOR, 'article[data-testid="tweet"]')
        if articles:
            break
        body_text = driver.find_element(By.TAG_NAME, "body").text
        if "Something went wrong" in body_text:
            _human_delay(2, 4)
            driver.refresh()
            _human_delay(3, 5)
        elif "protected" in body_text.lower():
            if own_driver:
                driver.quit()
            return []
        else:
            _human_delay(2, 3)

    seen_ids = set()
    all_tweets = []
    no_new_count = 0
    last_height = 0

    for scroll_i in range(max_scrolls):
        raw_tweets = _extract_tweets_from_page(driver)

        new_count = 0
        reached_before_range = False

        for tw in raw_tweets:
            tid = tw.get("tweet_id", "")
            if not tid or tid in seen_ids:
                continue
            seen_ids.add(tid)
            new_count += 1

            try:
                tw_dt = datetime.fromisoformat(tw["created_at"].replace("Z", "+00:00"))
            except (ValueError, TypeError):
                tw_dt = None

            if tw_dt:
                if tw_dt < since_dt:
                    reached_before_range = True
                    continue
                if tw_dt > until_dt:
                    continue

            tw["username"] = username
            all_tweets.append(tw)

        if reached_before_range:
            break

        if new_count == 0:
            no_new_count += 1
            if no_new_count >= 8:
                break
        else:
            no_new_count = 0

        # Human-like scroll with random variation
        scroll_dist = random.randint(500, 1200)
        driver.execute_script(f"window.scrollBy(0, {scroll_dist});")
        actual_pause = scroll_pause * random.uniform(0.7, 1.4)
        time.sleep(actual_pause)

        new_height = driver.execute_script("return document.body.scrollHeight")
        if new_height == last_height:
            no_new_count += 1
        last_height = new_height

    if own_driver:
        driver.quit()

    return all_tweets


if __name__ == "__main__":
    result1 = scrape_twitter_profile("Margarita100313")
    result2 = scrape_twitter_profile("oksanatserkovna")
    print(result1)
    print(result2)
