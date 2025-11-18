import os
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from io import BytesIO
from database import db, create_document, get_documents
from schemas import Document as DocumentSchema, Chunk as ChunkSchema

app = FastAPI(title="Agentic RAG API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def read_root():
    return {"message": "Agentic RAG Backend Running"}


class IngestResponse(BaseModel):
    document_id: str
    chunks_created: int


def simple_token_count(text: str) -> int:
    # Very rough token estimate (words)
    return len(text.split())


def optimal_chunk(text: str, max_tokens: int = 350, overlap_tokens: int = 50) -> List[str]:
    """
    Simple, pragmatic chunking that respects sentence boundaries when possible.
    - Splits by paragraphs then sentences.
    - Packs sentences into chunks up to max_tokens with an overlap.
    """
    import re

    # Normalize new lines
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

    sentences = []
    for p in paragraphs:
        # naive sentence split; good enough for many docs
        parts = re.split(r"(?<=[.!?])\s+", p)
        sentences.extend([s.strip() for s in parts if s.strip()])

    chunks: List[str] = []
    current: List[str] = []
    current_tokens = 0

    for s in sentences:
        s_tokens = simple_token_count(s)
        if current_tokens + s_tokens <= max_tokens:
            current.append(s)
            current_tokens += s_tokens
        else:
            if current:
                chunks.append(" ".join(current))
            # start new with overlap from previous
            if overlap_tokens > 0 and chunks:
                prev = chunks[-1].split()
                overlap = prev[-overlap_tokens:]
                current = [" ".join(overlap), s]
                current_tokens = len(overlap) + s_tokens
            else:
                current = [s]
                current_tokens = s_tokens

    if current:
        chunks.append(" ".join(current))

    # Safety: ensure no empty chunks
    chunks = [c.strip() for c in chunks if c.strip()]
    return chunks


# Embedding stub using a simple hashing projection to keep requirements minimal.
# In a real deployment you would swap this with an actual embedding model/service.
import hashlib

def cheap_embedding(text: str, dim: int = 128) -> List[float]:
    m = hashlib.sha256(text.encode("utf-8")).digest()
    # Repeat/trim to reach dim, scale to [0,1)
    arr = list(m) * ((dim // len(m)) + 1)
    arr = arr[:dim]
    return [v / 255.0 for v in arr]


def l2_normalize(vec: List[float]) -> List[float]:
    import math
    norm = math.sqrt(sum(v*v for v in vec)) or 1.0
    return [v / norm for v in vec]


# ----------------------------
# File parsing for PDF/DOCX/TXT
# ----------------------------

def parse_text_from_pdf(content: bytes) -> str:
    try:
        from pdfminer.high_level import extract_text
        return extract_text(BytesIO(content)) or ""
    except Exception as e:
        return f"[pdf-parse-error] {e}"


def parse_text_from_docx(content: bytes) -> str:
    try:
        from docx import Document as DocxDocument
        doc = DocxDocument(BytesIO(content))
        parts = []
        for p in doc.paragraphs:
            if p.text:
                parts.append(p.text)
        return "\n".join(parts)
    except Exception as e:
        return f"[docx-parse-error] {e}"


def extract_text_from_upload(filename: str, content: bytes, content_type: Optional[str]) -> str:
    ext = filename.split(".")[-1].lower() if "." in filename else ""
    if ext == "pdf" or (content_type and "pdf" in content_type.lower()):
        text = parse_text_from_pdf(content)
        if text.strip():
            return text
    if ext in ("docx",) or (content_type and "officedocument.wordprocessingml" in content_type.lower()):
        text = parse_text_from_docx(content)
        if text.strip():
            return text
    # Fallback to UTF-8 decode for txt/unknown
    try:
        return content.decode("utf-8")
    except Exception:
        try:
            return content.decode("latin-1")
        except Exception:
            return ""


@app.post("/ingest", response_model=IngestResponse)
async def ingest_document(
    file: UploadFile = File(...),
    title: Optional[str] = Form(None),
    max_tokens: int = Form(350),
    overlap_tokens: int = Form(50),
):
    if db is None:
        raise HTTPException(status_code=500, detail="Database not configured")

    # Read bytes
    content_bytes = await file.read()
    if not content_bytes:
        raise HTTPException(status_code=400, detail="Empty file")

    # Extract text using appropriate parser
    text = extract_text_from_upload(file.filename, content_bytes, file.content_type)
    if not text.strip():
        raise HTTPException(status_code=400, detail="Unable to extract text from file. Unsupported or empty content.")

    # Create document record
    doc = DocumentSchema(
        title=title or file.filename,
        source_type=(file.filename.split(".")[-1].lower() if "." in file.filename else "other"),
        metadata={"size": len(content_bytes), "content_type": file.content_type},
    )
    document_id = create_document("document", doc)

    # Chunk
    chunks = optimal_chunk(text, max_tokens=max_tokens, overlap_tokens=overlap_tokens)

    # Store chunks with cheap embeddings
    created = 0
    embeddings: List[List[float]] = []
    for idx, c in enumerate(chunks):
        emb = l2_normalize(cheap_embedding(c))
        embeddings.append(emb)
        chunk_doc = ChunkSchema(
            document_id=document_id,
            content=c,
            chunk_index=idx,
            tokens=simple_token_count(c),
            embedding=emb,
        )
        create_document("chunk", chunk_doc)
        created += 1

    # Build a simple document vector index entry (average of chunk embeddings)
    if embeddings:
        dim = len(embeddings[0])
        avg = [0.0] * dim
        for e in embeddings:
            for i, v in enumerate(e):
                avg[i] += v
        avg = [v / len(embeddings) for v in avg]
        avg = l2_normalize(avg)
        # store into a dedicated collection
        create_document("doc_index", {"document_id": document_id, "embedding": avg, "title": doc.title})

    return IngestResponse(document_id=document_id, chunks_created=created)


class QueryRequest(BaseModel):
    query: str
    top_k: int = 5


class QueryResult(BaseModel):
    chunk_index: int
    content: str
    score: float
    document_id: str
    title: Optional[str] = None


class QueryResponse(BaseModel):
    results: List[QueryResult]


def cosine_sim(a: List[float], b: List[float]) -> float:
    import math
    if not a or not b:
        return 0.0
    if len(a) != len(b):
        # pad to shorter
        n = min(len(a), len(b))
        a = a[:n]
        b = b[:n]
    dot = sum(x*y for x,y in zip(a,b))
    na = math.sqrt(sum(x*x for x in a))
    nb = math.sqrt(sum(x*x for x in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na*nb)


@app.post("/query", response_model=QueryResponse)
def query_chunks(payload: QueryRequest):
    if db is None:
        raise HTTPException(status_code=500, detail="Database not configured")

    q_emb = l2_normalize(cheap_embedding(payload.query))

    # Stage 1: shortlist top documents using doc_index
    doc_index = get_documents("doc_index", {}, limit=2000)
    doc_scored = []
    for d in doc_index:
        emb = d.get("embedding")
        score = cosine_sim(q_emb, emb) if emb else 0.0
        doc_scored.append((score, d))
    doc_scored.sort(key=lambda x: x[0], reverse=True)
    top_doc_ids = [d.get("document_id") for _, d in doc_scored[:10]] or []

    # Stage 2: fetch chunks restricted to shortlisted documents
    chunk_filter: Dict[str, Any] = {}
    if top_doc_ids:
        chunk_filter = {"document_id": {"$in": top_doc_ids}}
    chunks = get_documents("chunk", chunk_filter, limit=5000)

    scored = []
    for d in chunks:
        emb = d.get("embedding")
        content_val = d.get("content", "")
        score = cosine_sim(q_emb, emb) if emb else 0.0
        scored.append((score, d, content_val))

    scored.sort(key=lambda x: x[0], reverse=True)
    top = scored[: max(1, min(payload.top_k, 20))]

    # fetch titles for doc ids (fallback to doc_index cache)
    results: List[QueryResult] = []
    id_to_title = {d.get("document_id"): d.get("title") for _, d in doc_scored}
    for score, d, content_val in top:
        doc_id = d.get("document_id")
        title = id_to_title.get(doc_id)
        if title is None:
            try:
                title = next((doc.get("title") for doc in get_documents("document", {"_id": doc_id}, limit=1)), None)
            except Exception:
                pass

        results.append(
            QueryResult(
                chunk_index=int(d.get("chunk_index", 0)),
                content=content_val,
                score=float(score),
                document_id=str(doc_id),
                title=title,
            )
        )

    return QueryResponse(results=results)


# ----------------------------
# Simple streaming agent endpoint
# ----------------------------
class AgentRequest(BaseModel):
    prompt: str
    top_k: int = 5


def agent_streamer(payload: AgentRequest):
    import json
    # Step 1: acknowledge
    yield f"data: {json.dumps({'type': 'status', 'value': 'Agent started'})}\n\n"
    # Step 2: run retrieval
    q = payload.prompt.strip()
    yield f"data: {json.dumps({'type': 'thought', 'value': 'Searching knowledge base'})}\n\n"
    results = query_chunks(QueryRequest(query=q, top_k=payload.top_k)).results
    snippets = [r.content[:500] for r in results]
    yield f"data: {json.dumps({'type': 'retrieved', 'count': len(snippets)})}\n\n"
    for i, s in enumerate(snippets):
        yield f"data: {json.dumps({'type': 'context', 'index': i, 'snippet': s})}\n\n"
    # Step 3: synthesize naive answer (extractive)
    answer = "\n\n".join(snippets[:3]) or "No relevant content found."
    yield f"data: {json.dumps({'type': 'final', 'answer': answer})}\n\n"
    yield "data: [DONE]\n\n"


@app.post("/agent")
def agent_endpoint(payload: AgentRequest):
    return StreamingResponse(agent_streamer(payload), media_type="text/event-stream")


@app.get("/test")
def test_database():
    response = {
        "backend": "✅ Running",
        "database": "❌ Not Available",
        "database_url": None,
        "database_name": None,
        "connection_status": "Not Connected",
        "collections": []
    }

    try:
        if db is not None:
            response["database"] = "✅ Available"
            response["database_url"] = "✅ Configured"
            response["database_name"] = db.name if hasattr(db, 'name') else "✅ Connected"
            response["connection_status"] = "Connected"
            try:
                collections = db.list_collection_names()
                response["collections"] = collections[:10]
                response["database"] = "✅ Connected & Working"
            except Exception as e:
                response["database"] = f"⚠️  Connected but Error: {str(e)[:50]}"
        else:
            response["database"] = "⚠️  Available but not initialized"
    except Exception as e:
        response["database"] = f"❌ Error: {str(e)[:50]}"

    import os as _os
    response["database_url"] = "✅ Set" if _os.getenv("DATABASE_URL") else "❌ Not Set"
    response["database_name"] = "✅ Set" if _os.getenv("DATABASE_NAME") else "❌ Not Set"

    return response


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
