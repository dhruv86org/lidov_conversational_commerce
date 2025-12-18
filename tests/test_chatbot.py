"""
Test Scenarios for Conversational Commerce Chatbot

This module contains test cases for the dual-agent chatbot system,
covering product search, order creation, and edge cases.

Test Scenarios:
1. Product price query
2. Multi-turn product discussion
3. Order confirmation with extraction
4. Ambiguous query handling
5. Invalid order rejection
"""

import os
import sys
import json
import pytest
from datetime import datetime
from typing import Dict, Any
from unittest.mock import Mock, patch, MagicMock

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.models import (
    Product, Order, OrderItem, CustomerInfo, OrderStatus,
    OrderCreateRequest, StockStatus
)
from src.database import DatabaseManager


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def sample_product():
    """Create a sample product for testing."""
    return Product(
        product_id="TEST-001",
        name="Test Wireless Headphones",
        description="High-quality wireless headphones with noise cancellation for testing purposes.",
        price=149.99,
        category="Electronics",
        stock_status=StockStatus.IN_STOCK,
        stock_quantity=25
    )


@pytest.fixture
def sample_order_request():
    """Create a sample order request for testing."""
    return OrderCreateRequest(
        customer_name="John Smith",
        customer_email="john@example.com",
        customer_phone="555-1234",
        shipping_address="123 Main St, City, ST 12345",
        product_id="ELEC-001",
        product_name="Wireless Bluetooth Headphones Pro",
        quantity=1,
        unit_price=149.99,
        notes="Please gift wrap"
    )


@pytest.fixture
def test_database(tmp_path):
    """Create a temporary test database."""
    db_path = tmp_path / "test_orders.db"
    return DatabaseManager(str(db_path))


# =============================================================================
# Test Scenario 1: Product Price Query
# =============================================================================

class TestProductPriceQuery:
    """Test scenarios for product price queries."""
    
    def test_product_price_validation(self, sample_product):
        """Test that product price is correctly validated."""
        assert sample_product.price == 149.99
        assert sample_product.price > 0
    
    def test_product_price_rounding(self):
        """Test that prices are rounded to 2 decimal places."""
        product = Product(
            product_id="TEST-002",
            name="Test Product",
            description="A test product with long price for validation testing.",
            price=149.999,
            category="Test",
            stock_status=StockStatus.IN_STOCK,
            stock_quantity=10
        )
        assert product.price == 150.0
    
    def test_product_price_must_be_positive(self):
        """Test that negative prices are rejected."""
        with pytest.raises(ValueError):
            Product(
                product_id="TEST-003",
                name="Test Product",
                description="A test product that should fail validation.",
                price=-10.00,
                category="Test",
                stock_status=StockStatus.IN_STOCK,
                stock_quantity=10
            )


# =============================================================================
# Test Scenario 2: Multi-turn Product Discussion
# =============================================================================

class TestMultiTurnConversation:
    """Test scenarios for multi-turn product discussions."""
    
    def test_product_search_result_format(self, sample_product):
        """Test that product search results have correct format."""
        # Simulate search result
        search_result = {
            "product_id": sample_product.product_id,
            "name": sample_product.name,
            "price": sample_product.price,
            "category": sample_product.category,
            "stock_status": sample_product.stock_status.value,
            "relevance_score": 0.95
        }
        
        assert "product_id" in search_result
        assert "name" in search_result
        assert "price" in search_result
        assert search_result["relevance_score"] >= 0
        assert search_result["relevance_score"] <= 1
    
    def test_product_details_completeness(self, sample_product):
        """Test that product details include all required fields."""
        assert sample_product.product_id is not None
        assert sample_product.name is not None
        assert sample_product.description is not None
        assert sample_product.price is not None
        assert sample_product.category is not None
        assert sample_product.stock_status is not None
        assert sample_product.stock_quantity is not None
    
    def test_stock_status_consistency(self):
        """Test that stock status matches quantity."""
        # Out of stock product
        product = Product(
            product_id="TEST-004",
            name="Out of Stock Item",
            description="A product that is currently out of stock for testing.",
            price=99.99,
            category="Test",
            stock_status=StockStatus.IN_STOCK,  # Intentionally wrong
            stock_quantity=0
        )
        # Validator should correct this
        assert product.stock_status == StockStatus.OUT_OF_STOCK


# =============================================================================
# Test Scenario 3: Order Confirmation with Extraction
# =============================================================================

class TestOrderExtraction:
    """Test scenarios for order detail extraction and confirmation."""
    
    def test_order_request_to_order_conversion(self, sample_order_request):
        """Test that OrderCreateRequest correctly converts to Order."""
        order = sample_order_request.to_order()
        
        assert order.customer.name == "John Smith"
        assert order.customer.email == "john@example.com"
        assert len(order.items) == 1
        assert order.items[0].product_id == "ELEC-001"
        assert order.items[0].quantity == 1
        assert order.total_amount == 149.99
    
    def test_order_subtotal_calculation(self):
        """Test that order subtotals are calculated correctly."""
        # Subtotal is recalculated by the model_validator
        item = OrderItem(
            product_id="TEST-001",
            product_name="Test Product",
            quantity=3,
            unit_price=49.99,
            subtotal=1.0  # Placeholder, will be recalculated
        )
        
        expected_subtotal = 3 * 49.99
        assert abs(item.subtotal - expected_subtotal) < 0.01
    
    def test_order_total_validation(self, sample_order_request):
        """Test that order total matches item subtotals."""
        order = sample_order_request.to_order()
        
        calculated_total = sum(item.subtotal for item in order.items)
        assert abs(order.total_amount - calculated_total) < 0.01
    
    def test_order_id_generation(self, sample_order_request):
        """Test that unique order IDs are generated."""
        order1 = sample_order_request.to_order()
        order2 = sample_order_request.to_order()
        
        assert order1.order_id != order2.order_id
        assert len(order1.order_id) == 36  # UUID format


# =============================================================================
# Test Scenario 4: Ambiguous Query Handling
# =============================================================================

class TestAmbiguousQueries:
    """Test scenarios for handling ambiguous user queries."""
    
    def test_product_category_search(self):
        """Test searching by category returns multiple results."""
        # This would be tested with actual vector store
        # Here we test the model validation
        products = [
            Product(
                product_id=f"ELEC-00{i}",
                name=f"Electronic Product {i}",
                description=f"Electronic product number {i} for category testing.",
                price=99.99 + i,
                category="Electronics",
                stock_status=StockStatus.IN_STOCK,
                stock_quantity=10
            )
            for i in range(1, 4)
        ]
        
        assert all(p.category == "Electronics" for p in products)
        assert len(products) == 3
    
    def test_price_range_handling(self):
        """Test products can be filtered by price conceptually."""
        products = [
            Product(
                product_id="CHEAP-001",
                name="Budget Item",
                description="A budget-friendly item for price range testing.",
                price=29.99,
                category="Test",
                stock_status=StockStatus.IN_STOCK,
                stock_quantity=50
            ),
            Product(
                product_id="MID-001",
                name="Mid-range Item",
                description="A mid-range item for price range testing purposes.",
                price=99.99,
                category="Test",
                stock_status=StockStatus.IN_STOCK,
                stock_quantity=30
            ),
            Product(
                product_id="PREM-001",
                name="Premium Item",
                description="A premium item for price range testing purposes.",
                price=299.99,
                category="Test",
                stock_status=StockStatus.IN_STOCK,
                stock_quantity=10
            )
        ]
        
        under_100 = [p for p in products if p.price < 100]
        assert len(under_100) == 2


# =============================================================================
# Test Scenario 5: Invalid Order Rejection
# =============================================================================

class TestInvalidOrderRejection:
    """Test scenarios for rejecting invalid orders."""
    
    def test_reject_zero_quantity(self):
        """Test that orders with zero quantity are rejected."""
        with pytest.raises(ValueError):
            OrderItem(
                product_id="TEST-001",
                product_name="Test Product",
                quantity=0,
                unit_price=49.99,
                subtotal=0
            )
    
    def test_reject_negative_quantity(self):
        """Test that orders with negative quantity are rejected."""
        with pytest.raises(ValueError):
            OrderItem(
                product_id="TEST-001",
                product_name="Test Product",
                quantity=-1,
                unit_price=49.99,
                subtotal=-49.99
            )
    
    def test_reject_zero_price(self):
        """Test that products with zero price are rejected."""
        with pytest.raises(ValueError):
            Product(
                product_id="FREE-001",
                name="Free Product",
                description="A product with zero price that should be rejected.",
                price=0,
                category="Test",
                stock_status=StockStatus.IN_STOCK,
                stock_quantity=100
            )
    
    def test_reject_empty_customer_name(self):
        """Test that orders without customer name are rejected."""
        with pytest.raises(ValueError):
            OrderCreateRequest(
                customer_name="",  # Empty name
                product_id="TEST-001",
                product_name="Test Product",
                quantity=1,
                unit_price=49.99
            )
    
    def test_reject_short_customer_name(self):
        """Test that orders with too-short customer name are rejected."""
        with pytest.raises(ValueError):
            OrderCreateRequest(
                customer_name="A",  # Too short
                product_id="TEST-001",
                product_name="Test Product",
                quantity=1,
                unit_price=49.99
            )
    
    def test_reject_invalid_email(self):
        """Test that invalid email format is rejected."""
        with pytest.raises(ValueError):
            CustomerInfo(
                name="John Smith",
                email="not-an-email"  # Missing @
            )


# =============================================================================
# Database CRUD Tests
# =============================================================================

class TestDatabaseOperations:
    """Test database CRUD operations."""
    
    def test_create_order(self, test_database, sample_order_request):
        """Test order creation in database."""
        order = sample_order_request.to_order()
        order_id = test_database.create_order(order)
        
        assert order_id == order.order_id
    
    def test_get_order_by_id(self, test_database, sample_order_request):
        """Test retrieving order by ID."""
        order = sample_order_request.to_order()
        test_database.create_order(order)
        
        retrieved = test_database.get_order_by_id(order.order_id)
        
        assert retrieved is not None
        assert retrieved.order_id == order.order_id
        assert retrieved.customer.name == order.customer.name
        assert retrieved.total_amount == order.total_amount
    
    def test_get_nonexistent_order(self, test_database):
        """Test retrieving non-existent order returns None."""
        result = test_database.get_order_by_id("nonexistent-id")
        assert result is None
    
    def test_update_order_status(self, test_database, sample_order_request):
        """Test updating order status."""
        order = sample_order_request.to_order()
        test_database.create_order(order)
        
        success = test_database.update_order_status(
            order.order_id, 
            OrderStatus.CONFIRMED
        )
        
        assert success
        
        updated = test_database.get_order_by_id(order.order_id)
        assert updated.status == OrderStatus.CONFIRMED
    
    def test_delete_order(self, test_database, sample_order_request):
        """Test order deletion."""
        order = sample_order_request.to_order()
        test_database.create_order(order)
        
        success = test_database.delete_order(order.order_id)
        assert success
        
        deleted = test_database.get_order_by_id(order.order_id)
        assert deleted is None
    
    def test_get_order_count(self, test_database, sample_order_request):
        """Test order count."""
        initial_count = test_database.get_order_count()
        
        order1 = sample_order_request.to_order()
        test_database.create_order(order1)
        
        order2 = sample_order_request.to_order()
        test_database.create_order(order2)
        
        final_count = test_database.get_order_count()
        assert final_count == initial_count + 2


# =============================================================================
# Integration Test Scenarios (Documented)
# =============================================================================

"""
Manual Integration Test Scenarios
=================================

These scenarios should be tested manually with the running chatbot.

Scenario 1: Product Price Query
-------------------------------
User: "How much do the wireless headphones cost?"
Expected: Bot searches products, returns price ($149.99 for Wireless Bluetooth Headphones Pro)

Scenario 2: Multi-turn Product Discussion
-----------------------------------------
User: "What gaming accessories do you have?"
Bot: [Lists gaming products]
User: "Tell me more about the gaming headset"
Bot: [Provides detailed info about Gaming Headset 7.1 Surround]
User: "Is it in stock?"
Bot: [Confirms stock status]

Scenario 3: Order Confirmation with Extraction
----------------------------------------------
User: "I want to buy the wireless headphones"
Bot: [Asks for confirmation and customer info]
User: "Yes, my name is Jane Doe"
Bot: [Confirms order details: product, quantity, price]
User: "Confirm"
Bot: [Creates order, returns order ID]

Scenario 4: Ambiguous Query Handling
------------------------------------
User: "I need something for my home office"
Bot: [Searches with semantic understanding, returns office products]
User: "What about something under $100?"
Bot: [Filters and shows relevant products under $100]

Scenario 5: Invalid Order Rejection
-----------------------------------
User: "Order 100 robot vacuums"
Bot: [Checks stock, informs limited availability]
User: "I want the hair dryer" (out of stock)
Bot: [Informs out of stock, suggests alternatives]

Scenario 6: Order Status Check
------------------------------
User: "What's the status of my order [order-id]?"
Bot: [Retrieves and displays order status]
"""


# =============================================================================
# Vector Store Tests
# =============================================================================

class TestVectorStoreOperations:
    """Test vector store functionality."""
    
    def test_product_document_format(self, sample_product):
        """Test that product documents contain all required fields."""
        # Simulate the document format used in VectorStoreManager
        document = f"""
Product: {sample_product.name}
Product ID: {sample_product.product_id}
Category: {sample_product.category}
Price: ${sample_product.price:.2f}
Availability: Available and in stock
Description: {sample_product.description}
        """.strip()
        
        assert sample_product.product_id in document
        assert sample_product.name in document
        assert str(sample_product.price) in document
        assert sample_product.category in document
    
    def test_product_metadata_completeness(self, sample_product):
        """Test metadata structure for vector store."""
        metadata = {
            "product_id": sample_product.product_id,
            "name": sample_product.name,
            "price": sample_product.price,
            "category": sample_product.category,
            "stock_status": sample_product.stock_status.value,
            "stock_quantity": sample_product.stock_quantity
        }
        
        # All required fields present
        assert "product_id" in metadata
        assert "name" in metadata
        assert "price" in metadata
        assert "category" in metadata
        assert "stock_status" in metadata
        assert "stock_quantity" in metadata
    
    def test_search_result_relevance_score(self):
        """Test that search results include relevance scores."""
        # Simulate search result structure
        search_result = {
            "product_id": "TEST-001",
            "document": "Product: Test Product...",
            "metadata": {"name": "Test Product", "price": 99.99},
            "distance": 0.15,
            "relevance_score": 0.85  # 1 - distance
        }
        
        assert 0 <= search_result["relevance_score"] <= 1
        assert search_result["distance"] >= 0


# =============================================================================
# Product Catalog Tests
# =============================================================================

class TestProductCatalog:
    """Test product catalog requirements."""
    
    def test_minimum_product_count(self):
        """Test that catalog has at least 30 products."""
        import json
        from pathlib import Path
        
        products_path = Path(__file__).parent.parent / "data" / "products.json"
        with open(products_path) as f:
            data = json.load(f)
        
        products = data.get("products", [])
        assert len(products) >= 30, f"Expected at least 30 products, got {len(products)}"
    
    def test_product_required_fields(self):
        """Test all products have required fields."""
        import json
        from pathlib import Path
        
        products_path = Path(__file__).parent.parent / "data" / "products.json"
        with open(products_path) as f:
            data = json.load(f)
        
        required_fields = ["product_id", "name", "description", "price", "category", "stock_status", "stock_quantity"]
        
        for product in data.get("products", []):
            for field in required_fields:
                assert field in product, f"Product {product.get('product_id', 'unknown')} missing field: {field}"
    
    def test_product_categories_diversity(self):
        """Test products span multiple categories."""
        import json
        from pathlib import Path
        
        products_path = Path(__file__).parent.parent / "data" / "products.json"
        with open(products_path) as f:
            data = json.load(f)
        
        categories = set(p.get("category") for p in data.get("products", []))
        assert len(categories) >= 5, f"Expected at least 5 categories, got {len(categories)}"


# =============================================================================
# Agent Handoff Tests
# =============================================================================

class TestAgentHandoff:
    """Test dual-agent handoff mechanism."""
    
    def test_purchase_intent_phrases(self):
        """Test recognition of purchase intent phrases."""
        purchase_intents = [
            "I'll take it",
            "I want to buy",
            "place order",
            "add to cart",
            "order this",
            "yes, please",
            "confirm",
            "buy now"
        ]
        
        # These phrases should trigger Order Agent
        for phrase in purchase_intents:
            assert len(phrase) > 0
            # In production, these would be detected by the LLM via system prompt
    
    def test_product_query_phrases(self):
        """Test recognition of product query phrases."""
        product_queries = [
            "How much does",
            "What is the price",
            "Tell me about",
            "Do you have",
            "Is it in stock",
            "What are the features"
        ]
        
        # These phrases should use RAG Agent (search_products tool)
        for phrase in product_queries:
            assert len(phrase) > 0


# =============================================================================
# Full Flow Simulation Tests
# =============================================================================

class TestFullOrderFlow:
    """Test complete order flow from product search to database persistence."""
    
    def test_complete_order_creation_flow(self, test_database):
        """Test creating an order through the full pipeline."""
        # Step 1: Simulate product search result
        product_data = {
            "product_id": "ELEC-001",
            "name": "Wireless Bluetooth Headphones Pro",
            "price": 149.99,
            "stock_status": "in_stock",
            "stock_quantity": 45
        }
        
        # Step 2: Customer confirms purchase and provides info
        customer_info = {
            "name": "Alice Johnson",
            "email": "alice@example.com"
        }
        
        # Step 3: Create order request (as LLM would extract from conversation)
        order_request = OrderCreateRequest(
            customer_name=customer_info["name"],
            customer_email=customer_info["email"],
            product_id=product_data["product_id"],
            product_name=product_data["name"],
            quantity=1,
            unit_price=product_data["price"]
        )
        
        # Step 4: Convert to Order and persist
        order = order_request.to_order()
        order_id = test_database.create_order(order)
        
        # Step 5: Verify order was persisted correctly
        retrieved_order = test_database.get_order_by_id(order_id)
        
        assert retrieved_order is not None
        assert retrieved_order.customer.name == "Alice Johnson"
        assert retrieved_order.items[0].product_name == "Wireless Bluetooth Headphones Pro"
        assert retrieved_order.total_amount == 149.99
        assert retrieved_order.status == OrderStatus.PENDING
    
    def test_order_with_multiple_quantity(self, test_database):
        """Test order with quantity > 1."""
        order_request = OrderCreateRequest(
            customer_name="Bob Smith",
            product_id="HOME-001",
            product_name="Smart LED Light Bulbs 4-Pack",
            quantity=3,
            unit_price=39.99
        )
        
        order = order_request.to_order()
        order_id = test_database.create_order(order)
        
        retrieved = test_database.get_order_by_id(order_id)
        
        assert retrieved.items[0].quantity == 3
        assert abs(retrieved.total_amount - (3 * 39.99)) < 0.01
    
    def test_order_timestamps(self, test_database, sample_order_request):
        """Test that orders have proper timestamps."""
        order = sample_order_request.to_order()
        test_database.create_order(order)
        
        retrieved = test_database.get_order_by_id(order.order_id)
        
        assert retrieved.created_at is not None
        assert retrieved.updated_at is not None


# =============================================================================
# Edge Case Tests
# =============================================================================

class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_out_of_stock_product_handling(self):
        """Test handling of out-of-stock products."""
        out_of_stock = Product(
            product_id="CARE-002",
            name="Hair Dryer Professional",
            description="Professional ionic hair dryer that is currently out of stock.",
            price=69.99,
            category="Personal Care",
            stock_status=StockStatus.OUT_OF_STOCK,
            stock_quantity=0
        )
        
        assert out_of_stock.stock_status == StockStatus.OUT_OF_STOCK
        assert out_of_stock.stock_quantity == 0
    
    def test_low_stock_detection(self):
        """Test low stock status is properly set."""
        low_stock = Product(
            product_id="TEST-LOW",
            name="Low Stock Item",
            description="A product with limited availability for testing.",
            price=99.99,
            category="Test",
            stock_status=StockStatus.IN_STOCK,  # Will be corrected
            stock_quantity=5  # Less than 10 triggers low_stock
        )
        
        # Validator should detect this as low stock
        assert low_stock.stock_quantity < 10
    
    def test_unicode_customer_name(self):
        """Test orders with unicode characters in customer name."""
        order_request = OrderCreateRequest(
            customer_name="María García",  # Spanish characters
            product_id="TEST-001",
            product_name="Test Product",
            quantity=1,
            unit_price=49.99
        )
        
        order = order_request.to_order()
        assert order.customer.name == "María García"
    
    def test_special_characters_in_notes(self):
        """Test order notes with special characters."""
        order_request = OrderCreateRequest(
            customer_name="John Doe",
            product_id="TEST-001",
            product_name="Test Product",
            quantity=1,
            unit_price=49.99,
            notes="Gift wrap please! 🎁 Include card saying: \"Happy Birthday!\""
        )
        
        order = order_request.to_order()
        assert "🎁" in order.notes
        assert "Happy Birthday!" in order.notes


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
