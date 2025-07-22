# db_config.py

import psycopg2
from psycopg2 import pool

# Simple connection pool
_connection_pool = None

def get_connection_pool():
    global _connection_pool
    if _connection_pool is None:
        _connection_pool = psycopg2.pool.SimpleConnectionPool(
            minconn=1,
            maxconn=10,
            dbname="postgres",
            user="postgres.ukepmwoqxybhauovasry",
            password="FinAnswer@Loyalist",
            host="aws-0-ca-central-1.pooler.supabase.com",
            port=5432,
            sslmode="require"
        )
    return _connection_pool

def get_connection():
    pool = get_connection_pool()
    return pool.getconn()

def return_connection(conn):
    pool = get_connection_pool()
    pool.putconn(conn)
