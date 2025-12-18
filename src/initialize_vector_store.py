"""
Vector Store Initialization Module.

Loads product documents, generates embeddings using OpenAI API,
and stores them in a ChromaDB vector database for semantic retrieval.
"""

import json
import os
from pathlib import Path
from typing import List, Dict, Any, Optional
import hashlib

from dotenv import load_dotenv
import chromadb
from chromadb.config import Settings
import openai

from src.models import Product, StockStatus

# Load environment variables
load_dotenv()

# Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://openrouter.ai/api/v1")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "openai/text-embedding-3-small")
VECTOR_STORE_PATH = os.getenv("VECTOR_STORE_PATH", "./vector_store")
PRODUCTS_PATH = os.getenv("PRODUCTS_PATH", "./data/products.json")

# Collection name for products
PRODUCTS_COLLECTION = "products"


class VectorStoreManager:
    """
    Manages the ChromaDB vector store for product embeddings.
    
    Handles document chunking, embedding generation via OpenRouter/OpenAI,
    and semantic similarity search for product retrieval.
    """
    
    def __init__(
        self,
        persist_directory: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        embedding_model: Optional[str] = None
    ):
        """
        Initialize the vector store manager.
        
        Args:
            persist_directory: Path to store vector database
            api_key: OpenAI/OpenRouter API key
            base_url: API base URL (OpenRouter by default)
            embedding_model: Model to use for embeddings
        """
        self.persist_directory = persist_directory or VECTOR_STORE_PATH
        self.api_key = api_key or OPENAI_API_KEY
        self.base_url = base_url or OPENAI_BASE_URL
        self.embedding_model = embedding_model or EMBEDDING_MODEL
        
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")
        
        # Initialize OpenAI client for OpenRouter
        self.openai_client = openai.OpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )
        
        # Initialize ChromaDB with persistence
        self.chroma_client = chromadb.PersistentClient(
            path=self.persist_directory,
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Get or create the products collection
        self.collection = self.chroma_client.get_or_create_collection(
            name=PRODUCTS_COLLECTION,
            metadata={"hnsw:space": "cosine"}
        )
    
    def generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for a text using OpenAI/OpenRouter API.
        
        Args:
            text: Text to embed
            
        Returns:
            List of embedding floats
        """
        response = self.openai_client.embeddings.create(
            model=self.embedding_model,
            input=text
        )
        return response.data[0].embedding
    
    def generate_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts in a batch.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
        """
        response = self.openai_client.embeddings.create(
            model=self.embedding_model,
            input=texts
        )
        return [item.embedding for item in response.data]
    
    def product_to_document(self, product: Product) -> str:
        """
        Convert a product to a text document for embedding.
        
        Creates a rich text representation combining all product attributes
        for better semantic retrieval.
        
        Args:
            product: Product object
            
        Returns:
            Text document for embedding
        """
        stock_text = {
            StockStatus.IN_STOCK: "Available and in stock",
            StockStatus.LOW_STOCK: "Low stock - limited availability",
            StockStatus.OUT_OF_STOCK: "Currently out of stock"
        }.get(product.stock_status, "Unknown availability")
        
        document = f"""
Product: {product.name}
Product ID: {product.product_id}
Category: {product.category}
Price: ${product.price:.2f}
Availability: {stock_text}
Description: {product.description}
        """.strip()
        
        return document
    
    def load_products_from_file(self, file_path: Optional[str] = None) -> List[Product]:
        """
        Load products from JSON file.
        
        Args:
            file_path: Path to products JSON file
            
        Returns:
            List of validated Product objects
        """
        path = Path(file_path or PRODUCTS_PATH)
        
        with open(path, 'r') as f:
            data = json.load(f)
        
        products = []
        for item in data.get('products', []):
            try:
                product = Product(**item)
                products.append(product)
            except Exception as e:
                print(f"Warning: Skipping invalid product {item.get('product_id', 'unknown')}: {e}")
        
        return products
    
    def initialize_from_products(self, products: List[Product]) -> int:
        """
        Initialize vector store with product embeddings.
        
        Args:
            products: List of Product objects
            
        Returns:
            Number of products indexed
        """
        if not products:
            print("No products to index")
            return 0
        
        # Prepare documents and metadata
        documents = []
        metadatas = []
        ids = []
        
        for product in products:
            doc = self.product_to_document(product)
            documents.append(doc)
            
            metadata = {
                "product_id": product.product_id,
                "name": product.name,
                "price": product.price,
                "category": product.category,
                "stock_status": product.stock_status.value,
                "stock_quantity": product.stock_quantity
            }
            metadatas.append(metadata)
            ids.append(product.product_id)
        
        print(f"Generating embeddings for {len(documents)} products...")
        
        # Generate embeddings in batches to avoid API limits
        batch_size = 20
        all_embeddings = []
        
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            embeddings = self.generate_embeddings_batch(batch)
            all_embeddings.extend(embeddings)
            print(f"  Processed {min(i + batch_size, len(documents))}/{len(documents)} products")
        
        # Upsert to collection (updates if exists, inserts if not)
        self.collection.upsert(
            documents=documents,
            embeddings=all_embeddings,
            metadatas=metadatas,
            ids=ids
        )
        
        print(f"Successfully indexed {len(products)} products in vector store")
        return len(products)
    
    def search_products(
        self,
        query: str,
        n_results: int = 5,
        filter_in_stock: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Search for products using semantic similarity.
        
        Args:
            query: Search query text
            n_results: Maximum number of results
            filter_in_stock: Only return in-stock products
            
        Returns:
            List of matching products with scores
        """
        # Generate query embedding
        query_embedding = self.generate_embedding(query)
        
        # Build where filter if needed
        where_filter = None
        if filter_in_stock:
            where_filter = {"stock_status": {"$ne": "out_of_stock"}}
        
        # Search collection
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where_filter,
            include=["documents", "metadatas", "distances"]
        )
        
        # Format results
        products = []
        if results['ids'] and results['ids'][0]:
            for i, product_id in enumerate(results['ids'][0]):
                product_data = {
                    "product_id": product_id,
                    "document": results['documents'][0][i] if results['documents'] else None,
                    "metadata": results['metadatas'][0][i] if results['metadatas'] else {},
                    "distance": results['distances'][0][i] if results['distances'] else None,
                    "relevance_score": 1 - (results['distances'][0][i] if results['distances'] else 0)
                }
                products.append(product_data)
        
        return products
    
    def get_product_by_id(self, product_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a specific product by ID.
        
        Args:
            product_id: Product identifier
            
        Returns:
            Product data if found, None otherwise
        """
        results = self.collection.get(
            ids=[product_id],
            include=["documents", "metadatas"]
        )
        
        if results['ids']:
            return {
                "product_id": results['ids'][0],
                "document": results['documents'][0] if results['documents'] else None,
                "metadata": results['metadatas'][0] if results['metadatas'] else {}
            }
        return None
    
    def get_all_products(self) -> List[Dict[str, Any]]:
        """
        Retrieve all products from the vector store.
        
        Returns:
            List of all products with metadata
        """
        results = self.collection.get(include=["documents", "metadatas"])
        
        products = []
        for i, product_id in enumerate(results['ids']):
            products.append({
                "product_id": product_id,
                "document": results['documents'][i] if results['documents'] else None,
                "metadata": results['metadatas'][i] if results['metadatas'] else {}
            })
        
        return products
    
    def get_product_count(self) -> int:
        """Get the number of products in the vector store."""
        return self.collection.count()
    
    def clear_collection(self):
        """Remove all products from the collection."""
        self.chroma_client.delete_collection(PRODUCTS_COLLECTION)
        self.collection = self.chroma_client.create_collection(
            name=PRODUCTS_COLLECTION,
            metadata={"hnsw:space": "cosine"}
        )


def initialize_vector_store(
    products_path: Optional[str] = None,
    vector_store_path: Optional[str] = None,
    force_reinitialize: bool = False
) -> VectorStoreManager:
    """
    Initialize the vector store with products from file.
    
    Args:
        products_path: Path to products JSON file
        vector_store_path: Path for vector store persistence
        force_reinitialize: If True, clear and rebuild the store
        
    Returns:
        Initialized VectorStoreManager
    """
    manager = VectorStoreManager(persist_directory=vector_store_path)
    
    # Check if already initialized
    current_count = manager.get_product_count()
    
    if current_count > 0 and not force_reinitialize:
        print(f"Vector store already contains {current_count} products. Use force_reinitialize=True to rebuild.")
        return manager
    
    if force_reinitialize:
        print("Clearing existing vector store...")
        manager.clear_collection()
    
    # Load and index products
    products = manager.load_products_from_file(products_path)
    print(f"Loaded {len(products)} products from file")
    
    manager.initialize_from_products(products)
    
    return manager


def get_vector_store() -> VectorStoreManager:
    """Get the vector store manager with default settings."""
    return VectorStoreManager()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Initialize product vector store")
    parser.add_argument(
        "--products",
        type=str,
        default=PRODUCTS_PATH,
        help="Path to products JSON file"
    )
    parser.add_argument(
        "--vector-store",
        type=str,
        default=VECTOR_STORE_PATH,
        help="Path for vector store persistence"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force reinitialization of vector store"
    )
    parser.add_argument(
        "--test-search",
        type=str,
        help="Test search query after initialization"
    )
    
    args = parser.parse_args()
    
    print("=" * 50)
    print("Initializing Product Vector Store")
    print("=" * 50)
    
    manager = initialize_vector_store(
        products_path=args.products,
        vector_store_path=args.vector_store,
        force_reinitialize=args.force
    )
    
    print(f"\nVector store initialized with {manager.get_product_count()} products")
    
    if args.test_search:
        print(f"\nTesting search with query: '{args.test_search}'")
        results = manager.search_products(args.test_search, n_results=3)
        
        for i, result in enumerate(results, 1):
            print(f"\n--- Result {i} ---")
            print(f"Product: {result['metadata'].get('name', 'Unknown')}")
            print(f"Price: ${result['metadata'].get('price', 0):.2f}")
            print(f"Relevance: {result['relevance_score']:.3f}")
