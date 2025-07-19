import pandas as pd
from db_config import get_connection

def populate_tickers_table():
    # Fetch S&P 500 tickers and company names from Wikipedia
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    df = pd.read_html(url)[0]
    df['Symbol'] = df['Symbol'].str.replace('.', '-', regex=False)
    tickers = df['Symbol'].tolist()
    companies = df['Security'].tolist()

    conn = get_connection()
    cur = conn.cursor()
    # Create the tickers table if it doesn't exist
    cur.execute('''
        CREATE TABLE IF NOT EXISTS tickers (
            ticker VARCHAR(10) PRIMARY KEY,
            company VARCHAR(255) NOT NULL
        );
    ''')
    # Insert tickers and company names
    for ticker, company in zip(tickers, companies):
        cur.execute(
            "INSERT INTO tickers (ticker, company) VALUES (%s, %s) ON CONFLICT (ticker) DO NOTHING;",
            (ticker, company)
        )
    conn.commit()
    cur.close()
    conn.close()
    print(f"Inserted {len(tickers)} tickers into the tickers table.")

if __name__ == "__main__":
    populate_tickers_table() 