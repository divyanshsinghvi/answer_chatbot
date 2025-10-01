from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

import pandas as pd
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field
from fastapi import Query
from equichat import router_llm
from openai import OpenAI
from equichat.config import CONFIG, get_settings
from equichat.store import Store
from equichat.ingest import ingest_pdf_with_openai
from equichat import facts_query
from equichat import router as eq_router  # <-- new lightweight router (no class)

# Optional: metrics & rate limits
try:
    from fastapi_instrumentator import Instrumentator  # type: ignore
except Exception:  # pragma: no cover
    Instrumentator = None  # type: ignore
try:
    from slowapi import Limiter  # type: ignore
    from slowapi.util import get_remote_address  # type: ignore
    limiter = Limiter(key_func=get_remote_address)
except Exception:  # pragma: no cover
    limiter = None  # type: ignore

app = FastAPI(title="EquiChat API", version="0.1.0")

# CORS
origins = [o.strip() for o in get_settings().cors_allow_origins.split(",")] if get_settings().cors_allow_origins else []
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins or ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----- Lifespan-managed singletons (avoid DB open at import time) -------------
ENTITIES_CSV = Path(__file__).resolve().parent.parent / "data" / "entities.csv"

@app.on_event("startup")
def _startup():
    # Create one Store per worker (OK). Avoid creating it in the reloader parent.
    store = Store()  # uses CONFIG.db_path (file or :memory:)
    if ENTITIES_CSV.exists():
        try:
            df = pd.read_csv(ENTITIES_CSV)
            store.register_entities_df(df)
        except Exception:
            pass
    app.state.store = store

    if Instrumentator and CONFIG.enable_metrics:
        try:
            Instrumentator().instrument(app).expose(app, include_in_schema=False)
        except Exception:
            pass

@app.on_event("shutdown")
def _shutdown():
    store: Store = getattr(app.state, "store", None)
    if store:
        try:
            store.close()
        except Exception:
            pass

# Dependency to access the store safely
def get_store(request: Request) -> Store:
    return request.app.state.store


# ----- Models -----------------------------------------------------------------
class QueryRequest(BaseModel):
    query: str = Field(..., description="User question")
    company: Optional[str] = Field(default=None)
    industry: Optional[str] = Field(default=None)

class IngestResponse(BaseModel):
    doc_id: str
    company: str
    page_count: int
    source_path: str


# ----- Endpoints ---------------------------------------------------------------
@app.get("/healthz")
def healthz():
    return {"status": "ok", "db": CONFIG.db_path}

@app.post("/ingest", response_model=IngestResponse)
async def ingest_endpoint(
    file: UploadFile = File(...),
    company_hint: Optional[str] = Form(None),
    store: Store = Depends(get_store),
):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
    tmp_dir = Path("./uploads")
    tmp_dir.mkdir(parents=True, exist_ok=True)
    out_path = tmp_dir / file.filename
    data = await file.read()
    out_path.write_bytes(data)

    try:
        result = ingest_pdf_with_openai(
            str(out_path),
            company_hint=company_hint,
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        )
        return JSONResponse(result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {e}")

@app.post("/query")
def query_endpoint(req: QueryRequest, store: Store = Depends(get_store)):
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    result = router_llm.query_with_llm_router(client, req.query, store.conn)
    return result

@app.get("/chat/stream")
def chat_stream(
    query: str,
    company: Optional[str] = None,
    industry: Optional[str] = None,
    store: Store = Depends(get_store),
):
    def _gen():
        yield "event: start\ndata: {}\n\n"
        route = eq_router.route(query)
        if route.tool == "facts_sql":
            rows = facts_query.query_facts_with_store(
                store=store,
                text=query,
                metric_keys=route.metric_keys,
                company_hint=route.company_hint or company,
                period_label=route.period_label,
                limit=5,
            )
            answer = facts_query.answer_text_for_cli(rows, question=query)
            payload = {
                "status": "ok",
                "tool": "facts_sql",
                "answer": answer,
                "rows": rows,
            }
        else:
            payload = {"status": "unknown", "message": "Router could not determine the right tool."}
        yield f"event: message\ndata: {json.dumps(payload)}\n\n"
        yield "event: end\ndata: {}\n\n"

    return StreamingResponse(_gen(), media_type="text/event-stream")


@app.get("/facts")
def list_facts(
    limit: int = Query(50, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    store: Store = Depends(get_store),
):
    df = store.conn.execute(
        """
        SELECT company, metric_key, metric_variant, value, unit, period_type, period_end,
               source_page, substr(source_span,1,160) AS snippet, confidence
        FROM facts
        ORDER BY company, metric_key, COALESCE(period_end, '9999-12-31')
        LIMIT ? OFFSET ?
        """,
        [limit, offset],
    ).df()
    return {"rows": df.to_dict(orient="records")}

# Optional rate limiting
if limiter and CONFIG.enable_rate_limits:
    from slowapi.errors import RateLimitExceeded  # type: ignore
    from fastapi import Request  # type: ignore

    @app.middleware("http")
    async def rate_limit_middleware(request: "Request", call_next):
        try:
            with limiter.limit("30/minute")(lambda: None)():
                return await call_next(request)
        except RateLimitExceeded:  # pragma: no cover
            return JSONResponse(status_code=429, content={"detail": "Rate limit exceeded"})

# ----- Local dev entry point ---------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    reload_flag = os.getenv("EQUICHAT_RELOAD", "false").lower() in {"1", "true", "yes"}
    uvicorn.run(
        "api:app",
        host=CONFIG.api_host,
        port=CONFIG.api_port,
        reload=reload_flag,
        factory=False,
    )
