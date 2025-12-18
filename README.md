# Conversational Commerce Chatbot

A production-grade dual-agent conversational commerce system that combines RAG (Retrieval-Augmented Generation) for product discovery with autonomous order processing. Built with OpenAI Function Calling and integrated with OpenRouter API endpoints.

## Overview

This system solves the real-world problem of handling both pre-purchase questions and checkout in a single conversational flow. It mirrors production requirements at companies like Shopify, Amazon, and AI-native e-commerce platforms.

### Key Capabilities

- **Product Discovery**: Natural language search across 40+ products using semantic similarity
- **Intelligent Handoff**: Automatic transition from product questions to order processing
- **Order Persistence**: SQL database with full CRUD operations
- **Context Awareness**: Extracts order details from conversation without re-asking users

## Architecture

The system uses a single LLM with Function Calling for autonomous agent selection:

1. **User asks about products** → LLM calls `search_products` (RAG Agent behavior)
2. **User expresses purchase intent** → LLM recognizes intent, asks for confirmation
3. **User confirms and provides name** → LLM calls `create_order` (Order Agent behavior)

No explicit handoff code is needed - the LLM autonomously selects the appropriate tool based on conversation context.

### Components

| Component | Technology | Purpose |
|-----------|------------|---------|
| Chat Model | OpenRouter/GPT-4o-mini | Conversation handling and tool selection |
| Embeddings | OpenAI text-embedding-3-small | Product document embeddings |
| Vector Store | ChromaDB | Semantic product search |
| Database | SQLite | Order persistence |
| Validation | Pydantic | Data validation and schemas |

## Features

### Product Catalog (40 Products)

Categories include Electronics, Home and Kitchen, Sports and Fitness, Office, Audio and Entertainment, Gaming, Travel and Outdoor, and Personal Care.

### Function Calling Tools

| Tool | Description | Agent |
|------|-------------|-------|
| `search_products` | Semantic search across product catalog | RAG |
| `get_product_details` | Retrieve specific product information | RAG |
| `check_stock_availability` | Verify product availability | RAG |
| `create_order` | Persist order to database | Order |
| `get_order_status` | Retrieve order by ID | Order |

## Setup

### Prerequisites

- Python 3.10 or higher
- OpenRouter API key (get one at https://openrouter.ai/keys)

### Installation

1. Create virtual environment: `python -m venv venv && source venv/bin/activate`
2. Install dependencies: `pip install -r requirements.txt`
3. Configure environment: `cp .env.example .env` and add your API key
4. Initialize vector store: `python -m src.initialize_vector_store`

### Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `OPENAI_API_KEY` | Yes | - | OpenRouter API key |
| `OPENAI_BASE_URL` | No | `https://openrouter.ai/api/v1` | API endpoint |
| `CHAT_MODEL` | No | `openai/gpt-4o-mini` | Chat model |
| `EMBEDDING_MODEL` | No | `openai/text-embedding-3-small` | Embedding model |
| `VECTOR_STORE_PATH` | No | `./vector_store` | ChromaDB path |
| `DATABASE_PATH` | No | `./db/orders.db` | SQLite path |

## Usage

### Running the Chatbot

Run `python -m src.chatbot` to start the CLI interface.

### CLI Commands

- `quit` or `exit`: End the conversation
- `reset`: Start a new conversation
- `orders`: View recent orders

### Example Conversation

User asks about products, the RAG agent searches the vector store. User decides to buy, the Order agent collects info and persists the order to SQLite.

## Technical Decisions

### Why RAG for Products (100+ words)

Retrieval-Augmented Generation (RAG) was chosen for product search because it enables semantic understanding beyond keyword matching. Traditional SQL LIKE queries fail when customers use natural language like "something for my home office" or "gifts under $100." RAG converts product descriptions into dense vector embeddings using OpenAI's text-embedding-3-small model, capturing semantic meaning. ChromaDB was selected as the vector store for its simplicity, persistence support, and cosine similarity search. The embedding dimension (1536) provides good accuracy while remaining efficient. Each product document combines name, description, price, category, and stock status to enable multi-faceted retrieval. This approach handles typos, synonyms, and conceptual queries that would fail with exact matching.

### Why Function Calling for Agent Orchestration

OpenAI Function Calling was chosen over manual intent detection or keyword routing for several reasons:

1. **Autonomous Decision Making**: The LLM decides which tool to use based on complete conversation context, not brittle regex patterns
2. **Structured Outputs**: Function parameters are automatically validated against JSON schemas
3. **Multi-tool Support**: The LLM can chain multiple tools in a single turn (search → check stock → create order)
4. **Future-proof**: Adding new tools requires only schema definition, no routing logic changes

This approach eliminates the need for separate intent classification models and provides more natural conversation flow.

### Agent Handoff Mechanism

The "handoff" between RAG Agent and Order Agent is implicit rather than explicit:

1. Both agents are actually the same LLM with different tools available
2. The system prompt instructs the LLM on when to use each tool
3. Purchase intent phrases trigger the transition naturally
4. All context (product discussed, customer name mentioned) flows through chat history
5. The LLM extracts order details from prior messages without re-asking

This unified approach avoids complex state machines while maintaining clear separation of concerns.

### Database Choice: SQLite

SQLite was chosen for order persistence because:

1. **Zero Configuration**: No separate database server needed
2. **ACID Compliance**: Full transaction support for order integrity
3. **Persistence**: Data survives program restarts
4. **Portability**: Single file database, easy to backup/migrate
5. **Sufficient Scale**: Handles thousands of orders efficiently

For production with higher concurrency, the DatabaseManager class could be swapped for PostgreSQL with minimal code changes thanks to the abstraction layer.

## Database Schema

### Orders Table

| Column | Type | Description |
|--------|------|-------------|
| order_id | TEXT PRIMARY KEY | UUID string |
| customer_name | TEXT NOT NULL | Customer full name |
| customer_email | TEXT | Optional email |
| customer_phone | TEXT | Optional phone |
| shipping_address | TEXT | Optional address |
| total_amount | REAL | Order total |
| status | TEXT | Order status enum |
| notes | TEXT | Special instructions |
| created_at | TEXT | ISO timestamp |
| updated_at | TEXT | ISO timestamp |

### Order Items Table

| Column | Type | Description |
|--------|------|-------------|
| id | INTEGER PRIMARY KEY | Auto-increment |
| order_id | TEXT | Foreign key to orders |
| product_id | TEXT | Product identifier |
| product_name | TEXT | Product name at order time |
| quantity | INTEGER | Quantity ordered |
| unit_price | REAL | Price per unit |
| subtotal | REAL | Line item total |

## Testing

See `tests/test_chatbot.py` for test scenarios covering:

1. Product price query
2. Multi-turn product discussion
3. Order confirmation with extraction
4. Ambiguous query handling
5. Invalid order rejection

Run tests with: `python -m pytest tests/`

## Project Structure

```
conversational-commerce/
├── data/
│   └── products.json          # 40 products catalog
├── db/
│   └── orders.db              # SQLite database (created on first run)
├── src/
│   ├── __init__.py
│   ├── models.py              # Pydantic models
│   ├── database.py            # CRUD operations
│   ├── initialize_vector_store.py  # Vector store setup
│   └── chatbot.py             # Main application
├── tests/
│   └── test_chatbot.py        # Test scenarios
├── vector_store/              # ChromaDB persistence (created on init)
├── .env.example               # Environment template
├── requirements.txt           # Dependencies
└── README.md                  # This file
```

## License

MIT License
