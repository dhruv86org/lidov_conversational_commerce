-- Conversational Commerce Chatbot - Orders Database Schema
-- SQLite Database

-- =============================================================================
-- Orders Table
-- =============================================================================
-- Main orders table storing order header information

CREATE TABLE IF NOT EXISTS orders (
    order_id TEXT PRIMARY KEY,              -- UUID string for unique identification
    customer_name TEXT NOT NULL,             -- Customer's full name (required)
    customer_email TEXT,                     -- Optional email address
    customer_phone TEXT,                     -- Optional phone number
    shipping_address TEXT,                   -- Optional delivery address
    total_amount REAL NOT NULL               -- Total order value
        CHECK(total_amount > 0),             -- Must be positive
    status TEXT NOT NULL DEFAULT 'pending',  -- Order status enum value
    notes TEXT,                              -- Optional order notes/instructions
    created_at TEXT NOT NULL,                -- ISO 8601 timestamp
    updated_at TEXT NOT NULL                 -- ISO 8601 timestamp
);

-- =============================================================================
-- Order Items Table
-- =============================================================================
-- Order line items table with foreign key to orders

CREATE TABLE IF NOT EXISTS order_items (
    id INTEGER PRIMARY KEY AUTOINCREMENT,    -- Auto-increment ID
    order_id TEXT NOT NULL,                  -- Reference to parent order
    product_id TEXT NOT NULL,                -- Product identifier
    product_name TEXT NOT NULL,              -- Product name at time of order
    quantity INTEGER NOT NULL                -- Quantity ordered
        CHECK(quantity >= 1),                -- Must be at least 1
    unit_price REAL NOT NULL                 -- Price per unit
        CHECK(unit_price > 0),               -- Must be positive
    subtotal REAL NOT NULL                   -- quantity * unit_price
        CHECK(subtotal > 0),                 -- Must be positive
    FOREIGN KEY (order_id) REFERENCES orders(order_id)
        ON DELETE CASCADE                    -- Cascade delete items when order deleted
);

-- =============================================================================
-- Indexes
-- =============================================================================
-- Indexes for query optimization

-- Index for filtering orders by status
CREATE INDEX IF NOT EXISTS idx_orders_status 
    ON orders(status);

-- Index for looking up order items by order_id
CREATE INDEX IF NOT EXISTS idx_order_items_order_id 
    ON order_items(order_id);

-- Index for finding orders by customer name
CREATE INDEX IF NOT EXISTS idx_orders_customer_name 
    ON orders(customer_name);

-- Index for date-based queries
CREATE INDEX IF NOT EXISTS idx_orders_created_at 
    ON orders(created_at);

-- =============================================================================
-- Views
-- =============================================================================
-- Useful views for reporting

-- Order summary view with item count and total
CREATE VIEW IF NOT EXISTS order_summary AS
SELECT 
    o.order_id,
    o.customer_name,
    o.total_amount,
    o.status,
    o.created_at,
    COUNT(oi.id) as item_count
FROM orders o
LEFT JOIN order_items oi ON o.order_id = oi.order_id
GROUP BY o.order_id;

-- Recent orders view (last 30 days)
CREATE VIEW IF NOT EXISTS recent_orders AS
SELECT *
FROM orders
WHERE datetime(created_at) >= datetime('now', '-30 days')
ORDER BY created_at DESC;

-- =============================================================================
-- Order Status Enum Values
-- =============================================================================
-- Valid status values:
-- - pending: Order created, awaiting confirmation
-- - confirmed: Order confirmed by customer
-- - processing: Order being prepared
-- - shipped: Order shipped to customer
-- - delivered: Order delivered
-- - cancelled: Order cancelled

-- =============================================================================
-- Example Queries
-- =============================================================================

-- Get order with items
-- SELECT o.*, oi.*
-- FROM orders o
-- JOIN order_items oi ON o.order_id = oi.order_id
-- WHERE o.order_id = 'uuid-here';

-- Get orders by status
-- SELECT * FROM orders WHERE status = 'pending';

-- Get total sales
-- SELECT SUM(total_amount) as total_sales FROM orders WHERE status != 'cancelled';

-- Get top selling products
-- SELECT product_id, product_name, SUM(quantity) as total_sold
-- FROM order_items
-- GROUP BY product_id
-- ORDER BY total_sold DESC
-- LIMIT 10;
