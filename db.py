"""
src/db.py
=========
Conexion a la base de datos SQLite de Northwind.
El esquema se define de forma fija y exacta para evitar que el LLM
invente columnas que no existen.
"""
from sqlalchemy import create_engine, text
import pandas as pd

_engine = None


def get_engine(db_path: str = "northwind.db"):
    global _engine
    if _engine is None:
        _engine = create_engine(f"sqlite:///{db_path}", echo=False)
    return _engine


# Esquema exacto verificado contra la BD real.
# Se define hardcodeado para garantizar precision total al LLM.
_SCHEMA = """
Database: Northwind Traders (SQLite)

Tables:

categories (
    category_id   INTEGER PK,
    category_name VARCHAR(25),   -- e.g. Beverages, Condiments, Seafood, Dairy Products
    description   VARCHAR(255)
)

suppliers (
    supplier_id   INTEGER PK,
    supplier_name VARCHAR(50),
    contact_name  VARCHAR(50),
    address       VARCHAR(50),
    city          VARCHAR(20),
    postal_code   VARCHAR(10),
    country       VARCHAR(15),
    phone         VARCHAR(15)
)

shippers (
    shipper_id   INTEGER PK,
    shipper_name VARCHAR(25),    -- Speedy Express, United Package, Federal Shipping
    phone        VARCHAR(15)
)

customers (
    customer_id   INTEGER PK,
    customer_name VARCHAR(50),   -- use this column to filter by customer name
    contact_name  VARCHAR(50),
    address       VARCHAR(50),
    city          VARCHAR(20),
    postal_code   VARCHAR(10),
    country       VARCHAR(15)
)

employees (
    employee_id INTEGER PK,
    last_name   VARCHAR(15),
    first_name  VARCHAR(15),
    birth_date  DATETIME,
    photo       VARCHAR(25),
    notes       VARCHAR(1024)
)

products (
    product_id   INTEGER PK,
    product_name VARCHAR(50),
    supplier_id  INTEGER FK -> suppliers.supplier_id,
    category_id  INTEGER FK -> categories.category_id,
    unit         VARCHAR(25),
    price        NUMERIC       -- unit price of the product
)

orders (
    order_id    INTEGER PK,
    customer_id INTEGER FK -> customers.customer_id,
    employee_id INTEGER FK -> employees.employee_id,
    order_date  DATETIME,
    shipper_id  INTEGER FK -> shippers.shipper_id
)

order_details (
    order_detail_id INTEGER PK,
    order_id        INTEGER FK -> orders.order_id,
    product_id      INTEGER FK -> products.product_id,
    quantity        INTEGER
)

CRITICAL RULES — MUST FOLLOW:
- Revenue = order_details.quantity * products.price  (NO unit_price column exists anywhere)
- Employee full name: first_name || ' ' || last_name
- Customer search: use customer_name column (NOT company_name, that column does not exist)
- Date formatting: strftime('%Y-%m', order_date)
- String concatenation: use || operator (NOT CONCAT)
- All table and column names are lowercase snake_case
- There is NO discount column in order_details
"""


def get_schema(db_path: str = "northwind.db") -> str:
    """Retorna el esquema exacto de la base de datos Northwind."""
    return _SCHEMA


def run_query(sql: str, db_path: str = "northwind.db") -> pd.DataFrame:
    """Ejecuta una query y retorna un DataFrame. Lanza excepcion si falla."""
    engine = get_engine(db_path)
    with engine.connect() as conn:
        return pd.read_sql_query(text(sql), conn)
