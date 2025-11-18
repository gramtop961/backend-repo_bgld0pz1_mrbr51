import os
from typing import List, Optional
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

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

    # Try to decode as UTF-8; for binary (pdf/docx) a real pipeline would parse
    try:
        text = content_bytes.decode("utf-8")
    except Exception:
        # fallback: treat as binary and create a placeholder; real implementation would parse PDF/DOCX
        text = f"Binary file '{file.filename}' of {len(content_bytes)} bytes could not be text-decoded. Add a parser to extract text."

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
    for idx, c in enumerate(chunks):
        emb = cheap_embedding(c)
        chunk_doc = ChunkSchema(
            document_id=document_id,
            content=c,
            chunk_index=idx,
            tokens=simple_token_count(c),
            embedding=emb,
        )
        create_document("chunk", chunk_doc)
        created += 1

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

    q_emb = cheap_embedding(payload.query)

    # Fetch recent subset to keep simple; a real system would use vector index
    docs = get_documents("chunk", {}, limit=2000)

    scored = []
    for d in docs:
        emb = d.get("embedding")
        content = d.get("content", "")
        score = cosine_sim(q_emb, emb) if emb else 0.0
        scored.append((score, d))

    scored.sort(key=lambda x: x[0], reverse=True)
    top = scored[: max(1, min(payload.top_k, 20))]

    # fetch titles for doc ids
    results: List[QueryResult] = []
    for score, d in top:
        doc_id = d.get("document_id")
        title = None
        try:
            # attempt to find parent document
            title = next((doc.get("title") for doc in get_documents("document", {"_id": d.get("document_id")}, limit=1)), None)  # may not match if string id types differ; kept simple
        except Exception:
            pass

        results.append(
            QueryResult(
                chunk_index=int(d.get("chunk_index", 0)),
                content=content,
                score=float(score),
                document_id=str(doc_id),
                title=title,
            )
        )

    return QueryResponse(results=results)


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
