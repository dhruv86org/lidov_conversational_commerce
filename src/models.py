"""
Pydantic models for the Conversational Commerce System.

Defines data validation schemas for products, orders, and customer information
with proper field constraints and custom validators for business logic.
"""

from datetime import datetime, timezone
from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field, field_validator, model_validator
import uuid


class StockStatus(str, Enum):
    """Enumeration for product stock status."""
    IN_STOCK = "in_stock"
    LOW_STOCK = "low_stock"
    OUT_OF_STOCK = "out_of_stock"


class OrderStatus(str, Enum):
    """Enumeration for order status."""
    PENDING = "pending"
    CONFIRMED = "confirmed"
    PROCESSING = "processing"
    SHIPPED = "shipped"
    DELIVERED = "delivered"
    CANCELLED = "cancelled"


class Product(BaseModel):
    """
    Product model with validation constraints.
    
    Attributes:
        product_id: Unique identifier for the product
        name: Product name (minimum 2 characters)
        description: Detailed product description
        price: Product price (must be greater than 0)
        category: Product category
        stock_status: Current stock availability status
        stock_quantity: Number of items in stock (must be >= 0)
    """
    product_id: str = Field(..., min_length=1, description="Unique product identifier")
    name: str = Field(..., min_length=2, description="Product name")
    description: str = Field(..., min_length=10, description="Product description")
    price: float = Field(..., gt=0, description="Product price in USD")
    category: str = Field(..., min_length=1, description="Product category")
    stock_status: StockStatus = Field(..., description="Stock availability status")
    stock_quantity: int = Field(..., ge=0, description="Quantity in stock")

    @field_validator('price')
    @classmethod
    def validate_price(cls, v: float) -> float:
        """Ensure price has at most 2 decimal places."""
        return round(v, 2)

    @model_validator(mode='after')
    def validate_stock_consistency(self) -> 'Product':
        """Ensure stock_status matches stock_quantity."""
        if self.stock_quantity == 0 and self.stock_status != StockStatus.OUT_OF_STOCK:
            self.stock_status = StockStatus.OUT_OF_STOCK
        elif self.stock_quantity > 0 and self.stock_status == StockStatus.OUT_OF_STOCK:
            self.stock_status = StockStatus.LOW_STOCK if self.stock_quantity < 10 else StockStatus.IN_STOCK
        return self


class CustomerInfo(BaseModel):
    """
    Customer information model for order processing.
    
    Attributes:
        name: Customer's full name
        email: Customer's email address (optional)
        phone: Customer's phone number (optional)
        shipping_address: Delivery address
    """
    name: str = Field(..., min_length=2, description="Customer full name")
    email: Optional[str] = Field(None, description="Customer email")
    phone: Optional[str] = Field(None, description="Customer phone number")
    shipping_address: Optional[str] = Field(None, description="Shipping address")

    @field_validator('email')
    @classmethod
    def validate_email(cls, v: Optional[str]) -> Optional[str]:
        """Basic email format validation."""
        if v is not None and '@' not in v:
            raise ValueError('Invalid email format')
        return v


class OrderItem(BaseModel):
    """
    Individual item within an order.
    
    Attributes:
        product_id: Reference to the product
        product_name: Name of the product
        quantity: Number of items ordered (must be >= 1)
        unit_price: Price per unit (must be > 0)
        subtotal: Total for this line item
    """
    product_id: str = Field(..., min_length=1, description="Product identifier")
    product_name: str = Field(..., min_length=2, description="Product name")
    quantity: int = Field(..., ge=1, description="Quantity ordered")
    unit_price: float = Field(..., gt=0, description="Price per unit")
    subtotal: float = Field(..., gt=0, description="Line item total")

    @model_validator(mode='after')
    def calculate_subtotal(self) -> 'OrderItem':
        """Auto-calculate subtotal from quantity and unit_price."""
        self.subtotal = round(self.quantity * self.unit_price, 2)
        return self


class Order(BaseModel):
    """
    Complete order model with all details for database persistence.
    
    Attributes:
        order_id: Unique order identifier (auto-generated UUID)
        customer: Customer information
        items: List of ordered items
        total_amount: Total order amount
        status: Current order status
        created_at: Order creation timestamp
        updated_at: Last update timestamp
        notes: Optional order notes or special instructions
    """
    order_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique order ID")
    customer: CustomerInfo = Field(..., description="Customer details")
    items: List[OrderItem] = Field(..., min_length=1, description="Order items")
    total_amount: float = Field(..., gt=0, description="Total order amount")
    status: OrderStatus = Field(default=OrderStatus.PENDING, description="Order status")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="Creation timestamp")
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="Last update timestamp")
    notes: Optional[str] = Field(None, description="Order notes")

    @model_validator(mode='after')
    def validate_total(self) -> 'Order':
        """Validate that total_amount matches sum of item subtotals."""
        calculated_total = sum(item.subtotal for item in self.items)
        if abs(self.total_amount - calculated_total) > 0.01:
            self.total_amount = round(calculated_total, 2)
        return self


class OrderCreateRequest(BaseModel):
    """
    Request model for creating a new order from chat context.
    
    This model is used to validate extracted order information
    before creating a full Order object.
    """
    customer_name: str = Field(..., min_length=2, description="Customer name")
    customer_email: Optional[str] = Field(None, description="Customer email")
    customer_phone: Optional[str] = Field(None, description="Customer phone")
    shipping_address: Optional[str] = Field(None, description="Shipping address")
    product_id: str = Field(..., min_length=1, description="Product to order")
    product_name: str = Field(..., min_length=2, description="Product name")
    quantity: int = Field(..., ge=1, description="Quantity to order")
    unit_price: float = Field(..., gt=0, description="Unit price")
    notes: Optional[str] = Field(None, description="Special instructions")

    def to_order(self) -> Order:
        """Convert request to full Order object."""
        customer = CustomerInfo(
            name=self.customer_name,
            email=self.customer_email,
            phone=self.customer_phone,
            shipping_address=self.shipping_address
        )
        
        item = OrderItem(
            product_id=self.product_id,
            product_name=self.product_name,
            quantity=self.quantity,
            unit_price=self.unit_price,
            subtotal=round(self.quantity * self.unit_price, 2)
        )
        
        return Order(
            customer=customer,
            items=[item],
            total_amount=item.subtotal,
            notes=self.notes
        )


class ProductSearchResult(BaseModel):
    """
    Model for product search results from the vector store.
    """
    product: Product
    relevance_score: float = Field(..., ge=0, le=1, description="Search relevance score")


class ChatMessage(BaseModel):
    """
    Model for individual chat messages in conversation history.
    """
    role: str = Field(..., pattern="^(user|assistant|system)$", description="Message role")
    content: str = Field(..., min_length=1, description="Message content")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="Message timestamp")


class ConversationContext(BaseModel):
    """
    Model for tracking conversation state and extracted information.
    """
    messages: List[ChatMessage] = Field(default_factory=list, description="Chat history")
    detected_intent: Optional[str] = Field(None, description="Current user intent")
    selected_product: Optional[Product] = Field(None, description="Product being discussed")
    customer_info: Optional[CustomerInfo] = Field(None, description="Extracted customer info")
    pending_order: Optional[OrderCreateRequest] = Field(None, description="Order being prepared")
