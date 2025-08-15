from config import get_db_connection
import re

# Mapping keywords to database tables
TABLE_MAP = {
    "units": ["unit", "apartment", "villa", "property", "house"],
    "developers": ["developer", "company", "real estate firm"],
    "compounds": ["compound", "gated community"],
    "property_settings": ["property setting", "configuration", "settings"]
}

COLUMN_MAP = {
    "units": ["name_en", "price", "Bedrooms", "Bathrooms", "apartment_area"],
    "developers": ["name_en", "country_id"],
    "compounds": ["name_en", "developer_id"]
}



def detect_table(user_input):
    """
    Determines which table the user is referring to.
    """
    for table, keywords in TABLE_MAP.items():
        for keyword in keywords:
            if keyword in user_input.lower():
                return table
    return None


def detect_columns(user_input, table):
    """
    Identifies columns based on user input.
    """
    if table in COLUMN_MAP:
        selected_columns = []
        for column in COLUMN_MAP[table]:
            if column in user_input.lower():
                selected_columns.append(column)
        return selected_columns if selected_columns else ["*"]  # Select all if no column specified
    return ["*"]


def generate_query(user_input):
    """Generates an SQL query dynamically based on user input."""
    table = detect_table(user_input)
    if not table:
        return None, "I couldn't determine which table to query. Can you clarify?"

    columns = detect_columns(user_input, table)
    column_str = ", ".join(columns)

    # Handle WHERE conditions
    conditions = []
    price_range = re.findall(r'\d+', user_input)
    if "price" in user_input.lower() and len(price_range) == 2:
        conditions.append(f"max_price BETWEEN {price_range[0]} AND {price_range[1]}")

    if "location" in user_input.lower():
        conditions.append("area_id IN (SELECT id FROM cities WHERE LOWER(name_en) = %s)")

    where_clause = f" WHERE {' AND '.join(conditions)}" if conditions else ""

    query = f"SELECT {column_str} FROM {table}{where_clause} LIMIT 5;"
    return query, None


def execute_query(query):
    """
    Runs the SQL query and fetches results.
    """
    conn = get_db_connection()
    if conn:
        cursor = conn.cursor(dictionary=True)
        cursor.execute(query)
        result = cursor.fetchall()
        cursor.close()
        conn.close()
        return result
    return []
