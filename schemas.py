"""
Database Schemas

Define your MongoDB collection schemas here using Pydantic models.
These schemas are used for data validation in your application.

Each Pydantic model represents a collection in your database.
Model name is converted to lowercase for the collection name:
- User -> "user" collection
- Product -> "product" collection
- BlogPost -> "blogs" collection
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any

# Example schemas (replace with your own):

class User(BaseModel):
    """
    Users collection schema
    Collection name: "user" (lowercase of class name)
    """
    name: str = Field(..., description="Full name")
    email: str = Field(..., description="Email address")
    address: str = Field(..., description="Address")
    age: Optional[int] = Field(None, ge=0, le=120, description="Age in years")
    is_active: bool = Field(True, description="Whether user is active")

class Product(BaseModel):
    """
    Products collection schema
    Collection name: "product" (lowercase of class name)
    """
    title: str = Field(..., description="Product title")
    description: Optional[str] = Field(None, description="Product description")
    price: float = Field(..., ge=0, description="Price in dollars")
    category: str = Field(..., description="Product category")
    in_stock: bool = Field(True, description="Whether product is in stock")

# Agentic RAG specific schemas

class Document(BaseModel):
    """
    Ingested documents metadata
    Collection: "document"
    """
    title: str = Field(..., description="Human-friendly title (usually file name)")
    source_type: str = Field(..., description="pdf|docx|txt|other")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Arbitrary metadata such as size, pages, etc.")

class Chunk(BaseModel):
    """
    Text chunks derived from documents
    Collection: "chunk"
    """
    document_id: str = Field(..., description="Reference to parent document id")
    content: str = Field(..., description="Chunk text content")
    chunk_index: int = Field(..., ge=0, description="Order of the chunk within the document")
    tokens: Optional[int] = Field(default=None, description="Approximate token/word count")
    embedding: Optional[List[float]] = Field(default=None, description="Vector embedding for the chunk")

# Add your own schemas here:
# --------------------------------------------------

# Note: The Flames database viewer will automatically:
# 1. Read these schemas from GET /schema endpoint
# 2. Use them for document validation when creating/editing
# 3. Handle all database operations (CRUD) directly
# 4. You don't need to create any database endpoints!
