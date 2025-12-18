"""
Conversational Commerce System

A dual-agent chatbot system for product discovery and order processing
using RAG (Retrieval-Augmented Generation) and Function Calling.
"""

from src.models import (
    Product,
    Order,
    OrderItem,
    CustomerInfo,
    OrderStatus,
    StockStatus,
    OrderCreateRequest
)
from src.database import DatabaseManager, get_database
from src.initialize_vector_store import VectorStoreManager, get_vector_store
from src.chatbot import ConversationalCommerceChatbot

__version__ = "1.0.0"
__all__ = [
    "Product",
    "Order",
    "OrderItem",
    "CustomerInfo",
    "OrderStatus",
    "StockStatus",
    "OrderCreateRequest",
    "DatabaseManager",
    "get_database",
    "VectorStoreManager",
    "get_vector_store",
    "ConversationalCommerceChatbot",
]
