
# db_config.py

import os
import time
from psycopg2 import pool
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv
import psycopg2

# Load environment variables
load_dotenv()

# Global connection pool
_connection_pool = None
_pool_lock = None

def get_connection_pool():
    global _connection_pool, _pool_lock
    if _connection_pool is None:
        try:
            _connection_pool = pool.SimpleConnectionPool(
                minconn=0,  # No minimum connections - create on demand
                maxconn=10,  # Only 1 connection at a time to avoid pool exhaustion
                user=os.getenv("DB_USER", "postgres.ukepmwoqxybhauovasry"),
                password=os.getenv("DB_PASSWORD", "FinAnswer@Loyalist"),
                host=os.getenv("DB_HOST", "aws-0-ca-central-1.pooler.supabase.com"),
                port=int(os.getenv("DB_PORT", 5432)),
                database=os.getenv("DB_NAME", "postgres"),
                sslmode="require",
                # Connection timeout settings
                connect_timeout=5,  # Slightly longer timeout
                # Keep connections alive
                keepalives_idle=60,
                keepalives_interval=10,
                keepalives_count=3
            )
            print("✅ Connection pool created successfully.")
        except Exception as e:
            print("❌ Failed to create connection pool:", e)
            raise
    return _connection_pool

def get_connection(max_retries=3):
    """Get a connection from the pool with retry logic."""
    for attempt in range(max_retries):
        try:
            pool = get_connection_pool()
            conn = pool.getconn()
            
            # Test the connection with timeout
            cursor = conn.cursor()
            cursor.execute("SELECT 1")
            cursor.fetchone()
            cursor.close()
            
            print(f"✅ Connection successful (attempt {attempt + 1})")
            return conn
        except Exception as e:
            print(f"❌ Connection attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(1)  # Shorter wait time
            else:
                print("❌ All connection attempts failed")
                return None

def return_connection(conn):
    """Return a connection to the pool with error handling."""
    if conn is None:
        return
        
    try:
        # Test if connection is still valid before returning
        cursor = conn.cursor()
        cursor.execute("SELECT 1")
        cursor.fetchone()
        cursor.close()
        
        get_connection_pool().putconn(conn)
    except Exception as e:
        print(f"❌ Connection is invalid, closing: {e}")
        try:
            conn.close()
        except:
            pass

def check_connection_health():
    """Check if the database connection is healthy."""
    try:
        conn = get_connection()
        if conn:
            cursor = conn.cursor()
            cursor.execute("SELECT 1")
            cursor.fetchone()
            cursor.close()
            return_connection(conn)
            return True
        return False
    except Exception as e:
        print(f"❌ Connection health check failed: {e}")
        return False
