# db_config.py

import psycopg2

def get_connection():
    return psycopg2.connect(
        dbname="postgres",
        user="postgres.ukepmwoqxybhauovasry",
        password="FinAnswer@Loyalist",
        host="aws-0-ca-central-1.pooler.supabase.com",
        port=5432,
        sslmode="require"
    )
