from db_config import get_connection

def debug_stock_data():
    try:
        conn = get_connection()
        cursor = conn.cursor()
        
        # Check if table exists
        cursor.execute("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_name = 'stock_data'
            );
        """)
        table_exists = cursor.fetchone()[0]
        print(f"stock_data table exists: {table_exists}")
        
        if table_exists:
            # Get table structure
            cursor.execute("""
                SELECT column_name, data_type 
                FROM information_schema.columns 
                WHERE table_name = 'stock_data'
                ORDER BY ordinal_position;
            """)
            columns = cursor.fetchall()
            print("\nTable structure:")
            for col in columns:
                print(f"  {col[0]}: {col[1]}")
            
            # Get sample data
            cursor.execute("SELECT * FROM stock_data LIMIT 5")
            sample_data = cursor.fetchall()
            print(f"\nSample data ({len(sample_data)} rows):")
            for row in sample_data:
                print(f"  {row}")
            
            # Get column names from cursor description
            cursor.execute("SELECT * FROM stock_data LIMIT 1")
            column_names = [desc[0] for desc in cursor.description]
            print(f"\nColumn names from cursor: {column_names}")
            
            # Try to get unique symbols using the correct column name
            if 'symbol' in column_names:
                cursor.execute("SELECT DISTINCT symbol FROM stock_data ORDER BY symbol")
                symbols = cursor.fetchall()
                print(f"\nUnique symbols ({len(symbols)}):")
                for symbol in symbols:
                    print(f"  {symbol[0]}")
            else:
                print("\nNo 'symbol' column found. Available columns:")
                for col in column_names:
                    print(f"  {col}")
            
            # Get count by symbol
            cursor.execute("""
                SELECT symbol, COUNT(*) as count 
                FROM stock_data 
                GROUP BY symbol 
                ORDER BY count DESC 
                LIMIT 10
            """)
            symbol_counts = cursor.fetchall()
            print(f"\nTop 10 symbols by data count:")
            for symbol, count in symbol_counts:
                print(f"  {symbol}: {count} rows")
        
        cursor.close()
        conn.close()
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    debug_stock_data() 