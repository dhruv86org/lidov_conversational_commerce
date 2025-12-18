"""
Database module for order persistence.

Provides SQL database schema, connection management, and CRUD operations
for orders using SQLite with parameterized queries for security.
"""

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional
from contextlib import contextmanager

from src.models import Order, OrderItem, CustomerInfo, OrderStatus


# Default database path
DEFAULT_DB_PATH = Path(__file__).parent.parent / "db" / "orders.db"


class DatabaseManager:
    """
    Manages SQLite database connections and operations for order persistence.
    
    Uses parameterized queries to prevent SQL injection and context managers
    for proper resource handling.
    """
    
    def __init__(self, db_path: Optional[str] = None):
        """
        Initialize database manager with optional custom path.
        
        Args:
            db_path: Path to SQLite database file. Uses default if not provided.
        """
        self.db_path = Path(db_path) if db_path else DEFAULT_DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize_schema()
    
    @contextmanager
    def _get_connection(self):
        """Context manager for database connections."""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception as e:
            conn.rollback()
            raise e
        finally:
            conn.close()
    
    def _initialize_schema(self):
        """Create database tables if they don't exist."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Orders table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS orders (
                    order_id TEXT PRIMARY KEY,
                    customer_name TEXT NOT NULL,
                    customer_email TEXT,
                    customer_phone TEXT,
                    shipping_address TEXT,
                    total_amount REAL NOT NULL CHECK(total_amount > 0),
                    status TEXT NOT NULL DEFAULT 'pending',
                    notes TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
            """)
            
            # Order items table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS order_items (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    order_id TEXT NOT NULL,
                    product_id TEXT NOT NULL,
                    product_name TEXT NOT NULL,
                    quantity INTEGER NOT NULL CHECK(quantity >= 1),
                    unit_price REAL NOT NULL CHECK(unit_price > 0),
                    subtotal REAL NOT NULL CHECK(subtotal > 0),
                    FOREIGN KEY (order_id) REFERENCES orders(order_id)
                        ON DELETE CASCADE
                )
            """)
            
            # Index for faster order lookups
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_orders_status 
                ON orders(status)
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_order_items_order_id 
                ON order_items(order_id)
            """)
    
    def create_order(self, order: Order) -> str:
        """
        Persist a new order to the database.
        
        Args:
            order: Validated Order object to save
            
        Returns:
            The order_id of the created order
            
        Raises:
            sqlite3.IntegrityError: If order_id already exists
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Insert order
            cursor.execute("""
                INSERT INTO orders (
                    order_id, customer_name, customer_email, customer_phone,
                    shipping_address, total_amount, status, notes,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                order.order_id,
                order.customer.name,
                order.customer.email,
                order.customer.phone,
                order.customer.shipping_address,
                order.total_amount,
                order.status.value,
                order.notes,
                order.created_at.isoformat(),
                order.updated_at.isoformat()
            ))
            
            # Insert order items
            for item in order.items:
                cursor.execute("""
                    INSERT INTO order_items (
                        order_id, product_id, product_name,
                        quantity, unit_price, subtotal
                    ) VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    order.order_id,
                    item.product_id,
                    item.product_name,
                    item.quantity,
                    item.unit_price,
                    item.subtotal
                ))
        
        return order.order_id
    
    def get_order_by_id(self, order_id: str) -> Optional[Order]:
        """
        Retrieve an order by its ID.
        
        Args:
            order_id: Unique order identifier
            
        Returns:
            Order object if found, None otherwise
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Get order
            cursor.execute("""
                SELECT * FROM orders WHERE order_id = ?
            """, (order_id,))
            row = cursor.fetchone()
            
            if not row:
                return None
            
            # Get order items
            cursor.execute("""
                SELECT * FROM order_items WHERE order_id = ?
            """, (order_id,))
            item_rows = cursor.fetchall()
            
            return self._row_to_order(row, item_rows)
    
    def get_orders_by_status(self, status: OrderStatus) -> List[Order]:
        """
        Retrieve all orders with a specific status.
        
        Args:
            status: Order status to filter by
            
        Returns:
            List of matching Order objects
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT * FROM orders WHERE status = ? ORDER BY created_at DESC
            """, (status.value,))
            order_rows = cursor.fetchall()
            
            orders = []
            for order_row in order_rows:
                cursor.execute("""
                    SELECT * FROM order_items WHERE order_id = ?
                """, (order_row['order_id'],))
                item_rows = cursor.fetchall()
                orders.append(self._row_to_order(order_row, item_rows))
            
            return orders
    
    def get_all_orders(self, limit: int = 100) -> List[Order]:
        """
        Retrieve all orders with optional limit.
        
        Args:
            limit: Maximum number of orders to return
            
        Returns:
            List of Order objects
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT * FROM orders ORDER BY created_at DESC LIMIT ?
            """, (limit,))
            order_rows = cursor.fetchall()
            
            orders = []
            for order_row in order_rows:
                cursor.execute("""
                    SELECT * FROM order_items WHERE order_id = ?
                """, (order_row['order_id'],))
                item_rows = cursor.fetchall()
                orders.append(self._row_to_order(order_row, item_rows))
            
            return orders
    
    def update_order_status(self, order_id: str, status: OrderStatus) -> bool:
        """
        Update the status of an existing order.
        
        Args:
            order_id: Order to update
            status: New status
            
        Returns:
            True if order was updated, False if not found
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                UPDATE orders 
                SET status = ?, updated_at = ?
                WHERE order_id = ?
            """, (status.value, datetime.now(timezone.utc).isoformat(), order_id))
            
            return cursor.rowcount > 0
    
    def delete_order(self, order_id: str) -> bool:
        """
        Delete an order and its items.
        
        Args:
            order_id: Order to delete
            
        Returns:
            True if order was deleted, False if not found
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Delete order items first (or rely on CASCADE)
            cursor.execute("""
                DELETE FROM order_items WHERE order_id = ?
            """, (order_id,))
            
            cursor.execute("""
                DELETE FROM orders WHERE order_id = ?
            """, (order_id,))
            
            return cursor.rowcount > 0
    
    def get_order_count(self) -> int:
        """Get total number of orders in database."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM orders")
            return cursor.fetchone()[0]
    
    def _row_to_order(self, order_row: sqlite3.Row, item_rows: List[sqlite3.Row]) -> Order:
        """Convert database rows to Order object."""
        items = [
            OrderItem(
                product_id=row['product_id'],
                product_name=row['product_name'],
                quantity=row['quantity'],
                unit_price=row['unit_price'],
                subtotal=row['subtotal']
            )
            for row in item_rows
        ]
        
        customer = CustomerInfo(
            name=order_row['customer_name'],
            email=order_row['customer_email'],
            phone=order_row['customer_phone'],
            shipping_address=order_row['shipping_address']
        )
        
        return Order(
            order_id=order_row['order_id'],
            customer=customer,
            items=items,
            total_amount=order_row['total_amount'],
            status=OrderStatus(order_row['status']),
            created_at=datetime.fromisoformat(order_row['created_at']),
            updated_at=datetime.fromisoformat(order_row['updated_at']),
            notes=order_row['notes']
        )


# SQL Schema documentation for reference
SQL_SCHEMA = """
-- Orders Database Schema

-- Main orders table storing order header information
CREATE TABLE orders (
    order_id TEXT PRIMARY KEY,           -- UUID string for unique identification
    customer_name TEXT NOT NULL,          -- Customer's full name
    customer_email TEXT,                  -- Optional email address
    customer_phone TEXT,                  -- Optional phone number
    shipping_address TEXT,                -- Optional delivery address
    total_amount REAL NOT NULL,           -- Total order value (must be > 0)
    status TEXT NOT NULL DEFAULT 'pending', -- Order status enum
    notes TEXT,                           -- Optional order notes
    created_at TEXT NOT NULL,             -- ISO format timestamp
    updated_at TEXT NOT NULL              -- ISO format timestamp
);

-- Order line items table with foreign key to orders
CREATE TABLE order_items (
    id INTEGER PRIMARY KEY AUTOINCREMENT, -- Auto-increment ID
    order_id TEXT NOT NULL,               -- Reference to parent order
    product_id TEXT NOT NULL,             -- Product identifier
    product_name TEXT NOT NULL,           -- Product name at time of order
    quantity INTEGER NOT NULL,            -- Quantity ordered (>= 1)
    unit_price REAL NOT NULL,             -- Price per unit (> 0)
    subtotal REAL NOT NULL,               -- quantity * unit_price
    FOREIGN KEY (order_id) REFERENCES orders(order_id) ON DELETE CASCADE
);

-- Indexes for query optimization
CREATE INDEX idx_orders_status ON orders(status);
CREATE INDEX idx_order_items_order_id ON order_items(order_id);
"""


def get_database() -> DatabaseManager:
    """Get the default database manager instance."""
    return DatabaseManager()
