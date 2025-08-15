import mysql.connector
import mysql.connector
import mysql.connector
import config

def fetch_data(query, params=None):
    """Executes a SELECT query and returns results, printing a confirmation when data is from MySQL."""
    conn = None
    try:
        conn = config.get_db_connection()  # Ensure this function exists in config.py
        cursor = conn.cursor(dictionary=True)
        cursor.execute(query, params or ())
        result = cursor.fetchall()

        if result:
            print("✅ Data fetched from MySQL")  # Print confirmation
        else:
            print("⚠️ Query executed, but no matching data found.")

        return result

    except mysql.connector.Error as err:
        print(f"❌ Database Error: {err}")
        return []

    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()

def execute_query(query, params=None):
    """Executes an INSERT, UPDATE, or DELETE query."""
    try:
        conn = config.get_db_connection()
        cursor = conn.cursor()
        cursor.execute(query, params or ())
        conn.commit()
        return True
    except mysql.connector.Error as err:
        print(f"❌ Database Error: {err}")
        return False
    finally:
        if 'conn' in locals() and conn.is_connected():
            cursor.close()
            conn.close()
