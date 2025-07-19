import psycopg2
from psycopg2.extras import execute_values
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import sys
import os

# Add the parent directory to the path to find db_config
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from db_config import get_connection

def add_sentiment_columns_if_not_exists(conn):
    """Adds 'title_sentiment' and 'sentiment_label' columns if they don't exist."""
    with conn.cursor() as cur:
        try:
            # Add numerical sentiment score column
            cur.execute("ALTER TABLE finance_news ADD COLUMN title_sentiment REAL;")
            print("✅ 'title_sentiment' column added.")
        except psycopg2.errors.DuplicateColumn:
            print("ℹ️ 'title_sentiment' column already exists.")
            conn.rollback() # Important to rollback the failed transaction
        
        try:
            # Add text label column
            cur.execute("ALTER TABLE finance_news ADD COLUMN sentiment_label TEXT;")
            print("✅ 'sentiment_label' column added.")
        except psycopg2.errors.DuplicateColumn:
            print("ℹ️ 'sentiment_label' column already exists.")
            conn.rollback()
        
        conn.commit()


def get_sentiment_label(score):
    """Categorizes a sentiment score into a text label."""
    if score >= 0.05:
        return 'Positive'
    elif score <= -0.05:
        return 'Negative'
    else:
        return 'Neutral'

def analyze_and_store_sentiment():
    """
    Fetches news titles, calculates sentiment score and label,
    and updates the database.
    """
    conn = get_connection()
    if not conn:
        return

    try:
        add_sentiment_columns_if_not_exists(conn)
        
        analyzer = SentimentIntensityAnalyzer()
        
        with conn.cursor() as cur:
            # Fetch articles where sentiment has not been calculated yet
            cur.execute("SELECT id, title FROM finance_news WHERE title_sentiment IS NULL")
            news_to_analyze = cur.fetchall()
            
            if not news_to_analyze:
                print("✅ No new news articles to analyze.")
                return

            print(f"Found {len(news_to_analyze)} new articles to analyze for sentiment.")
            
            updates = []
            for news_id, title in news_to_analyze:
                sentiment_score = analyzer.polarity_scores(title)['compound']
                sentiment_label = get_sentiment_label(sentiment_score)
                updates.append((sentiment_score, sentiment_label, news_id))
            
            # Perform a bulk update for both columns
            update_query = """
                UPDATE finance_news 
                SET title_sentiment = data.title_sentiment, sentiment_label = data.sentiment_label
                FROM (VALUES %s) AS data(title_sentiment, sentiment_label, id) 
                WHERE finance_news.id = data.id
            """
            execute_values(cur, update_query, updates)
            
            conn.commit()
            print(f"✅ Successfully updated sentiment for {len(updates)} articles.")

    except Exception as e:
        print(f"❌ An error occurred: {e}")
    finally:
        if conn:
            conn.close()

if __name__ == "__main__":
    analyze_and_store_sentiment() 