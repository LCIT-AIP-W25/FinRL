# scrape.py
import requests
from bs4 import BeautifulSoup
from datetime import datetime
from psycopg2.extras import execute_values
from db_config import get_connection
import time
import feedparser

def scrape_yahoo_news_rss(ticker):
    url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US"
    feed = feedparser.parse(url)
    articles = []
    for entry in feed.entries:
        articles.append({
            "source": "Yahoo Finance",
            "ticker": ticker,
            "title": entry.title,
            "url": entry.link,
            "scraped_at": datetime.utcnow()
        })
    return articles

def insert_articles_to_db(conn, articles):
    if not articles:
        print("⚠️ No articles to insert.")
        return

    with conn.cursor() as cur:
        tuples = [(a["source"], a["ticker"], a["title"], a["url"], a["scraped_at"]) for a in articles]
        sql = """
            INSERT INTO finance_news (source, ticker, title, url, scraped_at)
            VALUES %s
            ON CONFLICT DO NOTHING
        """
        execute_values(cur, sql, tuples)
    conn.commit()
    print(f"✅ Inserted {len(articles)} articles into the database.")

def main():
    tickers = ["AAPL", "TSLA", "MSFT", "GOOG", "AMZN", "NVDA", "META", "NFLX"]
    conn = get_connection()
    all_articles = []

    for i, ticker in enumerate(tickers, 1):
        print(f"[{i}/{len(tickers)}] Scraping news for {ticker}...")
        try:
            articles = scrape_yahoo_news_rss(ticker)
            if articles:
                print(f"Found {len(articles)} articles for {ticker}:")
                for article in articles:
                    print(f"- {article['title']} ({article['url']})")
                all_articles.extend(articles)
            else:
                print(f"No articles found for {ticker}.")
            time.sleep(1)
        except Exception as e:
            print(f"❌ Error scraping {ticker}: {e}")

    insert_articles_to_db(conn, all_articles)
    conn.close()

if __name__ == "__main__":
    main()
