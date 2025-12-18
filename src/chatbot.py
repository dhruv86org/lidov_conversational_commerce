"""
Conversational Commerce Chatbot - Main Application

Dual-agent system with:
1. RAG Agent: Retrieves product information from vector store
2. Order Agent: Handles order creation and persistence

Uses OpenAI Function Calling for autonomous tool selection based on
conversation context. Integrates with OpenRouter API endpoints.
"""

import json
import os
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from dotenv import load_dotenv
import openai

from src.models import (
    Product, Order, OrderItem, CustomerInfo, OrderStatus,
    OrderCreateRequest, ChatMessage, StockStatus
)
from src.database import DatabaseManager, get_database
from src.initialize_vector_store import VectorStoreManager, get_vector_store

# Load environment variables
load_dotenv()

# Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://openrouter.ai/api/v1")
CHAT_MODEL = os.getenv("CHAT_MODEL", "openai/gpt-4o-mini")


# =============================================================================
# Function Calling Tool Definitions
# =============================================================================

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_products",
            "description": "Search for products in the catalog using natural language queries. Use this to find products by name, category, features, price range, or any product attributes. Returns relevant products with prices, descriptions, and availability.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural language search query describing the products to find. Examples: 'wireless headphones under $150', 'gaming accessories', 'smart home devices'"
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of products to return (default: 5)",
                        "default": 5
                    },
                    "in_stock_only": {
                        "type": "boolean",
                        "description": "If true, only return products that are in stock",
                        "default": False
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_product_details",
            "description": "Get detailed information about a specific product by its ID. Use this when a user asks for more details about a specific product.",
            "parameters": {
                "type": "object",
                "properties": {
                    "product_id": {
                        "type": "string",
                        "description": "The unique product identifier (e.g., 'ELEC-001')"
                    }
                },
                "required": ["product_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "create_order",
            "description": "Create and persist a new order to the database. Use this when a customer confirms they want to purchase a product and you have gathered their information (name at minimum). Extract order details from the conversation history.",
            "parameters": {
                "type": "object",
                "properties": {
                    "customer_name": {
                        "type": "string",
                        "description": "Full name of the customer"
                    },
                    "customer_email": {
                        "type": "string",
                        "description": "Customer's email address (optional)"
                    },
                    "customer_phone": {
                        "type": "string",
                        "description": "Customer's phone number (optional)"
                    },
                    "shipping_address": {
                        "type": "string",
                        "description": "Delivery address (optional)"
                    },
                    "product_id": {
                        "type": "string",
                        "description": "ID of the product to order"
                    },
                    "product_name": {
                        "type": "string",
                        "description": "Name of the product"
                    },
                    "quantity": {
                        "type": "integer",
                        "description": "Number of items to order",
                        "minimum": 1,
                        "default": 1
                    },
                    "unit_price": {
                        "type": "number",
                        "description": "Price per unit"
                    },
                    "notes": {
                        "type": "string",
                        "description": "Special instructions or notes for the order"
                    }
                },
                "required": ["customer_name", "product_id", "product_name", "quantity", "unit_price"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_order_status",
            "description": "Retrieve the status and details of an existing order by order ID. Use when customer asks about their order.",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {
                        "type": "string",
                        "description": "The unique order identifier (UUID format)"
                    }
                },
                "required": ["order_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "check_stock_availability",
            "description": "Check if a specific product is available in stock and get current quantity. Use before confirming orders.",
            "parameters": {
                "type": "object",
                "properties": {
                    "product_id": {
                        "type": "string",
                        "description": "The product ID to check"
                    }
                },
                "required": ["product_id"]
            }
        }
    }
]


# =============================================================================
# System Prompts
# =============================================================================

SYSTEM_PROMPT = """You are a helpful shopping assistant for an online store. Your role is to:

1. **Product Discovery (RAG Agent)**:
   - Help customers find products using the search_products tool
   - Provide detailed product information, prices, and availability
   - Answer questions about product features and specifications
   - Compare products when asked

2. **Order Processing (Order Agent)**:
   - When a customer expresses purchase intent (e.g., "I want to buy", "I'll take it", "add to cart", "order this"), begin the ordering process
   - Collect customer information naturally through conversation (name is required, email/phone/address are optional)
   - Confirm order details before creating the order
   - Use the create_order tool to persist orders to the database
   - Provide order confirmation with order ID

**Guidelines:**
- Be friendly, helpful, and concise
- Always mention prices in USD format ($XX.XX)
- If a product is out of stock, suggest alternatives
- For orders, always confirm the product, quantity, and total before processing
- Extract all order information from conversation context - don't ask for information already provided
- If the customer provides their name during conversation, remember it for the order
- Never ask the user to re-provide information they've already shared

**Agent Handoff:**
- When you detect purchase intent, smoothly transition from product assistance to order processing
- This happens automatically based on conversation context - no explicit handoff is needed
- The same conversation flow handles both product questions and order placement

**Order Confirmation:**
When creating an order, always confirm with the customer first by summarizing:
- Product name and ID
- Quantity
- Unit price and total
- Customer name
Only call create_order after the customer confirms.
"""


# =============================================================================
# Conversation Commerce Chatbot
# =============================================================================

class ConversationalCommerceChatbot:
    """
    Dual-agent chatbot for product search and order processing.
    
    Uses OpenAI Function Calling to autonomously select between:
    - RAG Agent: Product search and information retrieval
    - Order Agent: Order creation and persistence
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        chat_model: Optional[str] = None
    ):
        """
        Initialize the chatbot with API configuration.
        
        Args:
            api_key: OpenAI/OpenRouter API key
            base_url: API base URL
            chat_model: Model to use for chat completion
        """
        self.api_key = api_key or OPENAI_API_KEY
        self.base_url = base_url or OPENAI_BASE_URL
        self.chat_model = chat_model or CHAT_MODEL
        
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")
        
        # Initialize OpenAI client
        self.client = openai.OpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )
        
        # Initialize components
        self.vector_store = get_vector_store()
        self.database = get_database()
        
        # Conversation history
        self.messages: List[Dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT}
        ]
        
        # Tracking
        self.current_product: Optional[Dict[str, Any]] = None
        self.pending_order: Optional[Dict[str, Any]] = None
    
    def _execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        """
        Execute a tool function and return the result.
        
        Args:
            tool_name: Name of the tool to execute
            arguments: Tool arguments
            
        Returns:
            JSON string result from the tool
        """
        try:
            if tool_name == "search_products":
                return self._search_products(
                    query=arguments.get("query", ""),
                    max_results=arguments.get("max_results", 5),
                    in_stock_only=arguments.get("in_stock_only", False)
                )
            
            elif tool_name == "get_product_details":
                return self._get_product_details(
                    product_id=arguments.get("product_id", "")
                )
            
            elif tool_name == "create_order":
                return self._create_order(arguments)
            
            elif tool_name == "get_order_status":
                return self._get_order_status(
                    order_id=arguments.get("order_id", "")
                )
            
            elif tool_name == "check_stock_availability":
                return self._check_stock_availability(
                    product_id=arguments.get("product_id", "")
                )
            
            else:
                return json.dumps({"error": f"Unknown tool: {tool_name}"})
        
        except Exception as e:
            return json.dumps({"error": str(e)})
    
    def _search_products(
        self,
        query: str,
        max_results: int = 5,
        in_stock_only: bool = False
    ) -> str:
        """Execute product search using vector store."""
        results = self.vector_store.search_products(
            query=query,
            n_results=max_results,
            filter_in_stock=in_stock_only
        )
        
        # Format results for the model
        formatted_results = []
        for r in results:
            metadata = r.get("metadata", {})
            formatted_results.append({
                "product_id": metadata.get("product_id"),
                "name": metadata.get("name"),
                "price": metadata.get("price"),
                "category": metadata.get("category"),
                "stock_status": metadata.get("stock_status"),
                "relevance_score": round(r.get("relevance_score", 0), 3),
                "description": r.get("document", "")
            })
        
        # Track the first result as current product
        if formatted_results:
            self.current_product = formatted_results[0]
        
        return json.dumps({
            "query": query,
            "results_count": len(formatted_results),
            "products": formatted_results
        }, indent=2)
    
    def _get_product_details(self, product_id: str) -> str:
        """Get detailed product information."""
        product = self.vector_store.get_product_by_id(product_id)
        
        if not product:
            return json.dumps({"error": f"Product {product_id} not found"})
        
        self.current_product = {
            "product_id": product_id,
            **product.get("metadata", {})
        }
        
        return json.dumps({
            "product_id": product_id,
            "details": product.get("document", ""),
            "metadata": product.get("metadata", {})
        }, indent=2)
    
    def _check_stock_availability(self, product_id: str) -> str:
        """Check stock status for a product."""
        product = self.vector_store.get_product_by_id(product_id)
        
        if not product:
            return json.dumps({"error": f"Product {product_id} not found"})
        
        metadata = product.get("metadata", {})
        stock_status = metadata.get("stock_status", "unknown")
        stock_quantity = metadata.get("stock_quantity", 0)
        
        return json.dumps({
            "product_id": product_id,
            "name": metadata.get("name"),
            "stock_status": stock_status,
            "stock_quantity": stock_quantity,
            "available": stock_status != "out_of_stock"
        }, indent=2)
    
    def _create_order(self, order_data: Dict[str, Any]) -> str:
        """Create and persist an order."""
        try:
            # Validate with Pydantic
            order_request = OrderCreateRequest(
                customer_name=order_data.get("customer_name"),
                customer_email=order_data.get("customer_email"),
                customer_phone=order_data.get("customer_phone"),
                shipping_address=order_data.get("shipping_address"),
                product_id=order_data.get("product_id"),
                product_name=order_data.get("product_name"),
                quantity=order_data.get("quantity", 1),
                unit_price=order_data.get("unit_price"),
                notes=order_data.get("notes")
            )
            
            # Convert to Order
            order = order_request.to_order()
            
            # Persist to database
            order_id = self.database.create_order(order)
            
            return json.dumps({
                "success": True,
                "order_id": order_id,
                "customer_name": order.customer.name,
                "product_name": order.items[0].product_name,
                "quantity": order.items[0].quantity,
                "total_amount": order.total_amount,
                "status": order.status.value,
                "created_at": order.created_at.isoformat(),
                "message": f"Order {order_id} created successfully!"
            }, indent=2)
        
        except Exception as e:
            return json.dumps({
                "success": False,
                "error": str(e)
            }, indent=2)
    
    def _get_order_status(self, order_id: str) -> str:
        """Get order status by ID."""
        order = self.database.get_order_by_id(order_id)
        
        if not order:
            return json.dumps({"error": f"Order {order_id} not found"})
        
        return json.dumps({
            "order_id": order.order_id,
            "status": order.status.value,
            "customer_name": order.customer.name,
            "items": [
                {
                    "product_name": item.product_name,
                    "quantity": item.quantity,
                    "subtotal": item.subtotal
                }
                for item in order.items
            ],
            "total_amount": order.total_amount,
            "created_at": order.created_at.isoformat()
        }, indent=2)
    
    def chat(self, user_message: str) -> str:
        """
        Process a user message and return the assistant's response.
        
        Uses OpenAI Function Calling to autonomously select appropriate
        tools based on conversation context.
        
        Args:
            user_message: The user's input message
            
        Returns:
            The assistant's response
        """
        # Add user message to history
        self.messages.append({"role": "user", "content": user_message})
        
        # Get initial response with potential tool calls
        response = self.client.chat.completions.create(
            model=self.chat_model,
            messages=self.messages,
            tools=TOOLS,
            tool_choice="auto"
        )
        
        assistant_message = response.choices[0].message
        
        # Process tool calls if any
        while assistant_message.tool_calls:
            # Add assistant message with tool calls to history
            self.messages.append({
                "role": "assistant",
                "content": assistant_message.content,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments
                        }
                    }
                    for tc in assistant_message.tool_calls
                ]
            })
            
            # Execute each tool call
            for tool_call in assistant_message.tool_calls:
                tool_name = tool_call.function.name
                arguments = json.loads(tool_call.function.arguments)
                
                # Execute tool
                result = self._execute_tool(tool_name, arguments)
                
                # Add tool result to history
                self.messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result
                })
            
            # Get next response
            response = self.client.chat.completions.create(
                model=self.chat_model,
                messages=self.messages,
                tools=TOOLS,
                tool_choice="auto"
            )
            
            assistant_message = response.choices[0].message
        
        # Add final response to history
        final_response = assistant_message.content or ""
        self.messages.append({"role": "assistant", "content": final_response})
        
        return final_response
    
    def get_chat_history(self) -> List[Dict[str, Any]]:
        """Get the current conversation history."""
        return [
            msg for msg in self.messages 
            if msg.get("role") in ["user", "assistant"] and msg.get("content")
        ]
    
    def reset_conversation(self):
        """Reset conversation history."""
        self.messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        self.current_product = None
        self.pending_order = None
    
    def get_recent_orders(self, limit: int = 10) -> List[Order]:
        """Get recent orders from database."""
        return self.database.get_all_orders(limit=limit)


# =============================================================================
# CLI Interface
# =============================================================================

def run_cli():
    """Run the chatbot in command-line interface mode."""
    print("=" * 60)
    print("Welcome to the Conversational Commerce Chatbot!")
    print("=" * 60)
    print("\nI can help you find products and place orders.")
    print("Type 'quit' or 'exit' to end the conversation.")
    print("Type 'reset' to start a new conversation.")
    print("Type 'orders' to view recent orders.")
    print("-" * 60)
    
    try:
        chatbot = ConversationalCommerceChatbot()
    except Exception as e:
        print(f"\nError initializing chatbot: {e}")
        print("Make sure you have set up your environment variables correctly.")
        print("See .env.example for required configuration.")
        return
    
    while True:
        try:
            user_input = input("\nYou: ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() in ['quit', 'exit']:
                print("\nThank you for shopping with us! Goodbye!")
                break
            
            if user_input.lower() == 'reset':
                chatbot.reset_conversation()
                print("\nConversation reset. How can I help you?")
                continue
            
            if user_input.lower() == 'orders':
                orders = chatbot.get_recent_orders(5)
                if not orders:
                    print("\nNo orders found.")
                else:
                    print("\n--- Recent Orders ---")
                    for order in orders:
                        print(f"  Order ID: {order.order_id[:8]}...")
                        print(f"  Customer: {order.customer.name}")
                        print(f"  Total: ${order.total_amount:.2f}")
                        print(f"  Status: {order.status.value}")
                        print(f"  Created: {order.created_at.strftime('%Y-%m-%d %H:%M')}")
                        print("-" * 30)
                continue
            
            # Get chatbot response
            print("\nAssistant: ", end="")
            response = chatbot.chat(user_input)
            print(response)
        
        except KeyboardInterrupt:
            print("\n\nThank you for shopping with us! Goodbye!")
            break
        except Exception as e:
            print(f"\nError: {e}")
            print("Please try again.")


if __name__ == "__main__":
    run_cli()
