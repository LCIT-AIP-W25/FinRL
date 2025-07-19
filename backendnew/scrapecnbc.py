import time
from bs4 import BeautifulSoup
from db_config import get_connection
from psycopg2.extras import execute_values
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
from webdriver_manager.chrome import ChromeDriverManager

def create_driver(headless=True):
    options = webdriver.ChromeOptions()
    if headless:
        options.add_argument("--headless")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.page_load_strategy = "eager"
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    driver.set_page_load_timeout(60)
    driver.set_script_timeout(60)
    return driver

def safe_get(driver, url, retries=1):
    for attempt in range(retries + 1):
        try:
            driver.get(url)
            return True
        except TimeoutException:
            print(f"Timeout loading {url}, attempt {attempt + 1}")
    return False

def wait_for_element(driver, by, value, timeout=15):
    return WebDriverWait(driver, timeout).until(EC.presence_of_element_located((by, value)))

def scrape_cnbc(driver, ticker):
    url = f"https://www.cnbc.com/quotes/{ticker}?tab=news"
    if not safe_get(driver, url):
        return []
    try:
        wait_for_element(driver, By.CSS_SELECTOR, "div.QuoteNewsFeed-headline a", timeout=20)
    except:
        return []
    soup = BeautifulSoup(driver.page_source, "html.parser")
    articles = []
    for tag in soup.select("div.QuoteNewsFeed-headline a"):
        title = tag.get_text(strip=True)
        link = tag.get("href", "")
        if title and link:
            articles.append({"source": "CNBC", "ticker": ticker, "title": title, "url": link})
    return articles

def insert_articles(conn, articles):
    with conn.cursor() as cur:
        tuples = [(a["source"], a["ticker"], a["title"], a["url"]) for a in articles]
        sql = """
            INSERT INTO finance_news (source, ticker, title, url)
            VALUES %s
            ON CONFLICT DO NOTHING
        """
        execute_values(cur, sql, tuples)
    conn.commit()

def main():
    tickers = ["AAPL", "TSLA", "MSFT"]
    conn = get_connection()
    driver = create_driver(headless=False)
    all_articles = []

    for idx, ticker in enumerate(tickers, 1):
        print(f"[{idx}/{len(tickers)}] Scraping Yahoo for {ticker}...")
        try:
            all_articles += scrape_cnbc(driver, ticker)
        except Exception as e:
            print(f"Error: {e}")
        time.sleep(1)

    driver.quit()

    if all_articles:
        insert_articles(conn, all_articles)
        print(f"✅ Inserted {len(all_articles)} Yahoo articles.")
    else:
        print("⚠️ No Yahoo articles scraped.")

    conn.close()

if __name__ == "__main__":
    main()
