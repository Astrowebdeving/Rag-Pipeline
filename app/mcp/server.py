"""
MCP Server for RAG System

This module implements a Model Context Protocol (MCP) server that exposes
the RAG system's functionality as standardized tools that can be used by
AI assistants and other MCP-compatible clients.

The server acts as a bridge between MCP clients and the existing Flask API,
providing tools for document management and knowledge querying.
"""

import logging
import httpx
from typing import Optional, Dict, Any, List
from mcp.server.fastmcp import FastMCP, Context
from mcp.server.fastmcp.prompts import base
from pydantic import BaseModel, Field

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
API_BASE_URL = "http://localhost:5000/api"
DEFAULT_TIMEOUT = 30.0

# Get API key from environment
import os
API_KEY = os.environ.get('RAG_API_KEY', '')

def get_headers():
    """Get headers with API key if available."""
    headers = {}
    if API_KEY:
        headers['X-API-Key'] = API_KEY
    return headers

class DocumentInfo(BaseModel):
    """Information about a processed document."""
    document_id: str = Field(description="The unique document identifier")
    filename: str = Field(description="The original filename")
    chunks_created: int = Field(description="Number of chunks created from the document")
    processing_status: str = Field(description="Status of document processing")

class QueryResult(BaseModel):
    """Result from a knowledge query."""
    answer: str = Field(description="The generated answer based on retrieved knowledge")
    chunks_used: int = Field(description="Number of knowledge chunks used to generate the answer")
    source_documents: List[str] = Field(description="List of source document IDs used")

class SystemStatus(BaseModel):
    """System health and status information."""
    status: str = Field(description="Overall system status")
    api_available: bool = Field(description="Whether the API is responding")
    embedding_method: str = Field(description="Current embedding method")
    embedding_dimensions: int = Field(description="Embedding vector dimensions")

class ChunkPage(BaseModel):
    """A page of stored chunks."""
    page: int
    page_size: int
    total: int
    total_pages: int
    chunks: List[Dict[str, Any]]

# Create the MCP server
mcp = FastMCP("RAG System MCP Server")

@mcp.tool()
async def upload_document(
    file_path: str,
    filename: Optional[str] = None,
    chunking_method: str = "semantic",
    embedding_method: str = "sentence_transformer",
    ctx: Context = None
) -> DocumentInfo:
    """
    Upload and process a document into the RAG system.
    
    Args:
        file_path: Path to the document file to upload
        filename: Optional custom filename (defaults to file_path basename)
        chunking_method: Method for text chunking (semantic, fixed_length, adaptive)
        embedding_method: Method for creating embeddings (sentence_transformer, tfidf, huggingface)
    """
    if ctx:
        await ctx.info(f"Starting document upload: {file_path}")
    
    try:
        # Read the file
        with open(file_path, 'rb') as f:
            file_content = f.read()
        
        # Prepare the upload
        if not filename:
            import os
            filename = os.path.basename(file_path)
        
        files = {
            'file': (filename, file_content, 'application/octet-stream')
        }
        
        data = {
            'chunking_method': chunking_method,
            'embedding_method': embedding_method
        }
        
        # Upload to the Flask API
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            response = await client.post(
                f"{API_BASE_URL}/documents/upload",
                files=files,
                data=data,
                headers=get_headers()
            )
            response.raise_for_status()
            result = response.json()
        
        if ctx:
            await ctx.info(f"Document processed successfully: {result.get('chunks_created', 0)} chunks created")
        
        return DocumentInfo(
            document_id=result.get('document_id', 'unknown'),
            filename=filename,
            chunks_created=result.get('chunks_created', 0),
            processing_status="completed"
        )
        
    except Exception as e:
        if ctx:
            await ctx.error(f"Failed to upload document: {str(e)}")
        raise Exception(f"Document upload failed: {str(e)}")

@mcp.tool()
async def ingest_text(
    content: str,
    document_name: str,
    chunking_method: str = "semantic",
    embedding_method: str = "sentence_transformer",
    ctx: Context = None
) -> DocumentInfo:
    """
    Ingest text content directly into the RAG system.
    
    Args:
        content: The text content to ingest
        document_name: Name to assign to this document
        chunking_method: Method for text chunking (semantic, fixed_length, adaptive)
        embedding_method: Method for creating embeddings (sentence_transformer, tfidf, huggingface)
    """
    if ctx:
        await ctx.info(f"Starting text ingestion: {document_name}")
    
    try:
        data = {
            'text': content,
            'document_name': document_name,
            'chunking_method': chunking_method,
            'embedding_method': embedding_method
        }
        
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            response = await client.post(
                f"{API_BASE_URL}/documents/ingest",
                json=data,
                headers=get_headers()
            )
            response.raise_for_status()
            result = response.json()
        
        if ctx:
            await ctx.info(f"Text ingested successfully: {result.get('chunks_created', 0)} chunks created")
        
        return DocumentInfo(
            document_id=result.get('document_id', 'unknown'),
            filename=document_name,
            chunks_created=result.get('chunks_created', 0),
            processing_status="completed"
        )
        
    except Exception as e:
        if ctx:
            await ctx.error(f"Failed to ingest text: {str(e)}")
        raise Exception(f"Text ingestion failed: {str(e)}")

@mcp.tool()
async def query_knowledge(
    question: str,
    top_k: int = 5,
    retrieval_method: str = "similarity",
    ctx: Context = None
) -> QueryResult:
    """
    Query the RAG system's knowledge base to get answers based on ingested documents.
    
    Args:
        question: The question to ask
        top_k: Number of top chunks to retrieve for context
        retrieval_method: Method for retrieving relevant chunks (similarity, dense, hybrid, advanced)
    """
    if ctx:
        await ctx.info(f"Processing query: {question[:50]}...")
    
    try:
        data = {
            'query': question,
            'top_k': top_k,
            'retrieval_method': retrieval_method
        }
        
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            response = await client.post(
                f"{API_BASE_URL}/query",
                json=data,
                headers=get_headers()
            )
            response.raise_for_status()
            result = response.json()
        
        if ctx:
            await ctx.info(f"Query processed successfully, used {result.get('chunks_used', 0)} chunks")
        
        return QueryResult(
            answer=result.get('answer', 'No answer generated'),
            chunks_used=result.get('chunks_used', 0),
            source_documents=result.get('source_documents', [])
        )
        
    except Exception as e:
        if ctx:
            await ctx.error(f"Failed to process query: {str(e)}")
        raise Exception(f"Query processing failed: {str(e)}")

@mcp.tool()
async def get_system_status(ctx: Context = None) -> SystemStatus:
    """
    Get the current status and health of the RAG system.
    """
    if ctx:
        await ctx.info("Checking system status...")
    
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(f"{API_BASE_URL}/health")
            response.raise_for_status()
            result = response.json()
        
        return SystemStatus(
            status=result.get('status', 'unknown'),
            api_available=True,
            embedding_method=result.get('embedding_method', 'unknown'),
            embedding_dimensions=result.get('embedding_dimensions', 0)
        )
        
    except Exception as e:
        if ctx:
            await ctx.warning(f"API health check failed: {str(e)}")
        
        return SystemStatus(
            status="unhealthy",
            api_available=False,
            embedding_method="unknown",
            embedding_dimensions=0
        )

@mcp.resource("rag://system/status")
def get_system_status_resource() -> str:
    """Get current system status as a resource."""
    try:
        import httpx
        response = httpx.get(f"{API_BASE_URL}/health", timeout=10.0)
        response.raise_for_status()
        result = response.json()
        
        return f"""# RAG System Status

**Status**: {result.get('status', 'unknown')}
**API Available**: Yes
**Embedding Method**: {result.get('embedding_method', 'unknown')}
**Embedding Dimensions**: {result.get('embedding_dimensions', 0)}
**Last Updated**: {result.get('timestamp', 'unknown')}
"""
    except Exception as e:
        return f"""# RAG System Status

**Status**: unhealthy
**API Available**: No
**Error**: {str(e)}
"""

@mcp.prompt()
def create_knowledge_query(topic: str, context: str = "general") -> List[base.Message]:
    """
    Generate an effective prompt for querying the RAG system about a specific topic.
    
    Args:
        topic: The main topic or subject to query about
        context: Additional context for the query (technical, business, general, etc.)
    """
    
    context_instructions = {
        "technical": "Focus on technical details, implementation specifics, and precise definitions.",
        "business": "Emphasize business implications, use cases, and practical applications.",
        "general": "Provide a comprehensive overview suitable for a general audience.",
        "academic": "Include theoretical background, research findings, and scholarly perspectives."
    }
    
    instruction = context_instructions.get(context, context_instructions["general"])
    
    return [
        base.UserMessage(f"""Please provide detailed information about: {topic}

Context: {instruction}

Please include:
1. Key concepts and definitions
2. Important details and specifics
3. Relevant examples or use cases
4. Any relationships to related topics

Query: {topic}""")
    ]

@mcp.prompt()
def create_document_summary_request(document_focus: str = "main topics") -> List[base.Message]:
    """
    Generate a prompt for summarizing documents in the RAG system.
    
    Args:
        document_focus: What to focus on when summarizing (main topics, technical details, key findings, etc.)
    """
    
    return [
        base.UserMessage(f"""Please provide a comprehensive summary of the available documents, focusing on: {document_focus}

Include:
1. Main themes and topics covered
2. Key information and insights
3. Document relationships and connections
4. Important details relevant to the focus area

Please organize the summary in a clear, structured format.""")
    ]

# Configure the server
def get_mcp_server() -> FastMCP:
    """Get the configured MCP server instance."""
    logger.info("RAG System MCP Server initialized")
    return mcp 


@mcp.tool()
async def list_chunks(page: int = 1, page_size: int = 5, ctx: Context = None) -> ChunkPage:
    """
    List currently stored chunks in pages.

    Args:
        page: 1-based page index (default 1)
        page_size: number of chunks per page (default 5)
    """
    if ctx:
        await ctx.info(f"Listing chunks page {page} (size {page_size})")
    try:
        params = { 'page': page, 'page_size': page_size }
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            response = await client.get(
                f"{API_BASE_URL}/chunks",
                params=params,
                headers=get_headers(),
            )
            response.raise_for_status()
            data = response.json()
        return ChunkPage(**data)
    except Exception as e:
        if ctx:
            await ctx.error(f"Failed to list chunks: {str(e)}")
        raise Exception(f"List chunks failed: {str(e)}")