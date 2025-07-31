
# db_config.py

import os
from psycopg2 import pool
from dotenv import load_dotenv
import psycopg2

# Load environment variables
load_dotenv()

# Global connection pool
_connection_pool = None

def get_connection_pool():
    global _connection_pool
    if _connection_pool is None:
        try:
            _connection_pool = pool.SimpleConnectionPool(
                minconn=1,
                maxconn=10,
                user=os.getenv("DB_USER", "postgres.ukepmwoqxybhauovasry"),
                password=os.getenv("DB_PASSWORD", "FinAnswer@Loyalist"),
                host=os.getenv("DB_HOST", "aws-0-ca-central-1.pooler.supabase.com"),
                port=os.getenv("DB_PORT", 5432),
                database=os.getenv("DB_NAME", "postgres"),
                sslmode="require"
            )
            print("✅ Connection pool created successfully.")
        except Exception as e:
            print("❌ Failed to create connection pool:", e)
            raise
    return _connection_pool

def get_connection():
    try:
        conn = get_connection_pool().getconn()
        # Override the close method to use return_connection
        original_close = conn.close
        def custom_close():
            return_connection(conn)
        conn.close = custom_close
        return conn
    except Exception as e:
        print("❌ Failed to get connection from pool:", e)
        return None

def return_connection(conn):
    try:
        get_connection_pool().putconn(conn)
    except Exception as e:
        print("❌ Failed to return connection to pool:", e)
