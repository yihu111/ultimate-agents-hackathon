# main.py
import io, os, time
from typing import Dict, Optional, List, Literal
from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
import httpx
import asyncio
import sys
from pathlib import Path as _Path

# Ensure project root on sys.path so imports like 'langgraph_workflow' work
_PROJECT_ROOT = _Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# LangGraph (Dedalus-backed) workflow
from langgraph_workflow.product_images_graph_dedalus import (
    compile_graph as compile_dedalus_graph,
    ProductImagesState as DedalusState,
)

# =========================================================
# Load environment
# =========================================================
load_dotenv()

## Supabase removed: using direct image URLs now; uploads no longer supported

## Upload helpers removed



# Optional: shared secret for webhooks from agent
WEBHOOK_SECRET = os.environ.get("ADGRID_WEBHOOK_SECRET")  # e.g., "super-secret"

# (Optional/legacy) Agent config to keep other endpoints working
AGENT_BASE_URL = os.environ.get("AGENT_BASE_URL", "http://localhost:9000")
AGENT_API_KEY = os.environ.get("AGENT_API_KEY")

# =========================================================
# FastAPI setup
# =========================================================
app = FastAPI(title="AdGrid + Agent + Supabase API", version="7.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================================================
# In-memory state
# =========================================================
CURRENT_PROMPT: Optional[str] = None
CURRENT_INPUT_URL: Optional[str] = None
CURRENT_INGESTION_ID: Optional[str] = None

GRID_SHAPE = {"rows": 3, "cols": 3}
GRID_ITEMS: Dict[int, "ImageItem"] = {}          # base (variant=0)
VAR_ITEMS: Dict[int, List["ImageItem"]] = {}     # variations (variant>=1)

# =========================================================
# Models
# =========================================================
ImageKind = Literal["base", "variation"]
ImageStatus = Literal["queued", "running", "done", "error"]

class ImageItem(BaseModel):
    slot: int
    variant: int                 # 0 for base, >=1 for variations
    url: Optional[str]           # Supabase (or agent) public URL
    kind: ImageKind              # "base" | "variation"
    status: ImageStatus = "running"
    version: int = 0
    updatedAt: int = 0
    meta: Optional[Dict] = None  # any metadata from the agent

## Upload response model removed; clients should call /generate directly with image_url

class GridResp(BaseModel):
    status: ImageStatus
    rows: int
    cols: int
    progress: Dict[str, int]     # { done, total }
    items: List[ImageItem]       # exactly rows*cols base items

class SlotResp(BaseModel):
    slot: int
    items: List[ImageItem]       # base + variations

class MetaResp(BaseModel):
    prompt: Optional[str] = None

class MetaSetReq(BaseModel):
    prompt: Optional[str] = None

class RegenReq(BaseModel):
    slots: List[int]

class VariationsGenerateReq(BaseModel):
    count: int = 3

class SlotUpdate(BaseModel):
    status: ImageStatus
    url: Optional[str] = None
    meta: Optional[Dict] = None
    version: Optional[int] = None

# =========================================================
# Helpers
# =========================================================
def _now_ms() -> int:
    return int(time.time() * 1000)

def _ensure_base(slot: int) -> ImageItem:
    it = GRID_ITEMS.get(slot)
    if it is None:
        it = ImageItem(
            slot=slot, variant=0, kind="base",
            url=None, status="queued", version=0, updatedAt=_now_ms(),
        )
        GRID_ITEMS[slot] = it
    return it

def _progress(rows: int, cols: int) -> Dict[str, int]:
    total = rows * cols
    done = sum(
        1 for s in range(1, total + 1)
        if (it := GRID_ITEMS.get(s)) and it.status == "done" and it.url
    )
    return {"done": done, "total": total}

def _grid_status(rows: int, cols: int) -> ImageStatus:
    prog = _progress(rows, cols)
    if prog["done"] == 0:
        any_started = any(GRID_ITEMS.get(s) for s in range(1, rows * cols + 1))
        return "running" if any_started else "queued"
    if prog["done"] < (rows * cols):
        return "running"
    return "done"

# =========================================================
# Agent HTTP client
# =========================================================
class AgentClient:
    def __init__(self, base_url: str, api_key: Optional[str] = None):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key

    def _headers(self) -> Dict[str, str]:
        h = {"Content-Type": "application/json"}
        if self.api_key:
            h["Authorization"] = f"Bearer {self.api_key}"
        return h

    async def ingest_url(self, image_url: str, prompt: Optional[str]) -> Optional[str]:
        """
        POST {AGENT}/ingest  { prompt, image_url } → { ingestion_id? }
        If your agent doesn't support this, it's fine to return None.
        """
        payload = {"prompt": prompt, "image_url": image_url}
        async with httpx.AsyncClient(timeout=60) as client:
            r = await client.post(f"{self.base_url}/ingest", headers=self._headers(), json=payload)
            # If your agent has no /ingest, you can ignore errors by returning None.
            if r.status_code >= 400:
                return None
            data = r.json() if "application/json" in r.headers.get("content-type","") else {}
            return data.get("ingestion_id")

    async def generate_grid(self, rows: int, cols: int,
                            prompt: Optional[str],
                            ingestion_id: Optional[str],
                            input_url: Optional[str]) -> List[ImageItem]:
        """
        POST {AGENT}/grid  { rows, cols, prompt, ingestion_id?, image_url? }
        Expect: { items: [ {slot, url, meta?}, ... ] }
        """
        payload = {"rows": rows, "cols": cols, "prompt": prompt}
        if ingestion_id:
            payload["ingestion_id"] = ingestion_id
        elif input_url:
            payload["image_url"] = input_url

        async with httpx.AsyncClient(timeout=None) as client:
            r = await client.post(f"{self.base_url}/grid", headers=self._headers(), json=payload)
            r.raise_for_status()
            data = r.json()

        out: List[ImageItem] = []
        for raw in data.get("items", []):
            slot = int(raw["slot"])
            url = raw.get("url")
            prev = _ensure_base(slot)
            out.append(ImageItem(
                slot=slot, variant=0, kind="base",
                url=url, status="done" if url else "running",
                version=prev.version + 1, updatedAt=_now_ms(),
                meta=raw.get("meta"),
            ))
        return out

    async def generate_variations(self, slot: int, count: int,
                                  prompt: Optional[str],
                                  ingestion_id: Optional[str],
                                  base_item: ImageItem) -> List[ImageItem]:
        """
        POST {AGENT}/variations  { slot, count, prompt, ingestion_id? }
        Expect: { items: [ {slot, variant?, url, meta?}, ... ] }
        """
        payload = {"slot": slot, "count": max(1, count), "prompt": prompt}
        if ingestion_id:
            payload["ingestion_id"] = ingestion_id

        async with httpx.AsyncClient(timeout=None) as client:
            r = await client.post(f"{self.base_url}/variations", headers=self._headers(), json=payload)
            r.raise_for_status()
            data = r.json()

        existing = VAR_ITEMS.get(slot, [])
        used = {v.variant for v in existing}
        next_idx = (max(used) + 1) if used else 1

        out: List[ImageItem] = []
        for raw in data.get("items", []):
            variant = int(raw.get("variant") or next_idx)
            if not raw.get("variant"):
                next_idx += 1
            url = raw.get("url")
            out.append(ImageItem(
                slot=slot, variant=variant, kind="variation",
                url=url, status="done" if url else "running",
                version=base_item.version, updatedAt=_now_ms(),
                meta=raw.get("meta"),
            ))
        return out

AGENT = AgentClient(AGENT_BASE_URL, AGENT_API_KEY)

# =========================================================
# Meta (prompt)
# =========================================================
@app.get("/meta", response_model=MetaResp)
def get_meta():
    return MetaResp(prompt=CURRENT_PROMPT)

@app.post("/meta", response_model=MetaResp)
def set_meta(body: MetaSetReq):
    global CURRENT_PROMPT
    if body.prompt is not None:
        CURRENT_PROMPT = body.prompt
    return MetaResp(prompt=CURRENT_PROMPT)

## Upload endpoint removed; clients should supply image_url to /generate

# =========================================================
# Generate grid via LangGraph (Dedalus workflow)
# This replaces the previous agent-backed /generate implementation.
# The workflow will stream progressive URLs back by POSTing to
# /slot/{slot}/status from inside the graph node.
# =========================================================
@app.post("/generate", response_model=GridResp)
async def generate_grid(rows: int = 3, cols: int = 3,
                        prompt: Optional[str] = None,
                        image_url: Optional[str] = None):
    global GRID_SHAPE, CURRENT_PROMPT, CURRENT_INPUT_URL
    GRID_SHAPE["rows"], GRID_SHAPE["cols"] = rows, cols
    total = rows * cols

    # Mark all base slots as running immediately for UI
    for slot in range(1, total + 1):
        base = _ensure_base(slot)
        base.status = "running"
        base.updatedAt = _now_ms()

    # Remember inputs
    if prompt is not None:
        CURRENT_PROMPT = prompt.strip() or None
    if image_url is not None:
        CURRENT_INPUT_URL = image_url

    # Guard: require a meaningful product description for relevant results
    if not CURRENT_PROMPT or not CURRENT_PROMPT.strip():
        raise HTTPException(400, "Missing prompt. Provide a short product description for relevant results.")
    # Guard: require an image URL for this flow
    if not CURRENT_INPUT_URL or not CURRENT_INPUT_URL.strip():
        raise HTTPException(400, "Missing image_url. Provide a public image URL.")

    # Build initial LangGraph state
    model_name = os.getenv("TEST_OPENAI_MODEL", "gpt-4o")
    image_model = os.getenv("TEST_FLUX_MODEL", "flux-kontext-max")
    use_raw_mode = False

    initial_state: DedalusState = {
        "model_spec": {"name": model_name},
        "product_description": CURRENT_PROMPT,
        # Optional fields will be added conditionally below
        "num_dimensions": 2,
        "num_values_per_dim": 3,
        # Image generation config
        "image_model": image_model,
        "use_raw_mode": use_raw_mode,
        # Results aggregator
        "results": [],
    }

    # Add optional extras only if present to satisfy type checker
    if CURRENT_INPUT_URL:
        initial_state["initial_image_path"] = CURRENT_INPUT_URL

    async def _run_workflow():
        try:
            graph = compile_dedalus_graph()
            await graph.ainvoke(initial_state)
        except Exception as e:
            # Log and continue; UI keeps polling /grid
            print(f"[generate] workflow error: {e}")

    # Fire-and-forget workflow; streaming happens via webhooks
    asyncio.create_task(_run_workflow())

    prog = _progress(rows, cols)
    status = _grid_status(rows, cols)
    return GridResp(status=status, rows=rows, cols=cols, progress=prog,
                    items=[_ensure_base(s) for s in range(1, total + 1)])

# =========================================================
# Poll grid (frontend polls this every ~1–2s)
# =========================================================
@app.get("/grid", response_model=GridResp)
def get_grid():
    rows, cols = GRID_SHAPE["rows"], GRID_SHAPE["cols"]
    total = rows * cols
    items = [ _ensure_base(s) for s in range(1, total + 1) ]
    prog = _progress(rows, cols)
    status = _grid_status(rows, cols)
    return GridResp(status=status, rows=rows, cols=cols, progress=prog, items=items)

# =========================================================
# Regenerate a single base slot (re-call agent for one)
# =========================================================
@app.post("/slot/{slot}/generate", response_model=ImageItem)
async def generate_slot(slot: int):
    rows, cols = GRID_SHAPE["rows"], GRID_SHAPE["cols"]
    total = rows * cols
    if not (1 <= slot <= total):
        raise HTTPException(400, "Invalid slot")

    base = _ensure_base(slot)
    base.status = "running"
    base.updatedAt = _now_ms()

    try:
        # Simple reuse: ask agent for 1x1 and map it to this slot
        items = await AGENT.generate_grid(
            rows=1, cols=1,
            prompt=CURRENT_PROMPT,
            ingestion_id=CURRENT_INGESTION_ID,
            input_url=CURRENT_INPUT_URL,
        )
        if items:
            result = items[0]
            result.slot = slot
            GRID_ITEMS[slot] = result
        else:
            base.status = "error"
    except Exception as e:
        raise HTTPException(502, f"Agent slot error: {e}")

    return GRID_ITEMS[slot]

# =========================================================
# Get a single slot (base + variations)
# =========================================================
@app.get("/slot/{slot}", response_model=SlotResp)
def get_slot(slot: int):
    rows, cols = GRID_SHAPE["rows"], GRID_SHAPE["cols"]
    total = rows * cols
    if not (1 <= slot <= total):
        raise HTTPException(400, "Invalid slot")
    base = _ensure_base(slot)
    variations = VAR_ITEMS.get(slot, [])
    return SlotResp(slot=slot, items=[base] + variations)

# =========================================================
# Generate N variations for a slot (Agent returns URLs)
# =========================================================
@app.post("/slot/{slot}/variations/generate", response_model=SlotResp)
async def generate_variations(slot: int, body: VariationsGenerateReq = VariationsGenerateReq()):
    rows, cols = GRID_SHAPE["rows"], GRID_SHAPE["cols"]
    total = rows * cols
    if not (1 <= slot <= total):
        raise HTTPException(400, "Invalid slot")

    base = GRID_ITEMS.get(slot)
    if base is None or not base.url:
        raise HTTPException(404, "Base slot not generated yet.")

    try:
        new_items = await AGENT.generate_variations(
            slot=slot, count=body.count,
            prompt=CURRENT_PROMPT,
            ingestion_id=CURRENT_INGESTION_ID,
            base_item=base
        )
        VAR_ITEMS[slot] = VAR_ITEMS.get(slot, []) + new_items
    except Exception as e:
        raise HTTPException(502, f"Agent variations error: {e}")

    return SlotResp(slot=slot, items=[base] + VAR_ITEMS[slot])

# =========================================================
# Batch regenerate convenience
# =========================================================
@app.post("/regenerate", response_model=GridResp)
async def regenerate_slots(body: RegenReq):
    rows, cols = GRID_SHAPE["rows"], GRID_SHAPE["cols"]
    total = rows * cols
    invalid = [s for s in body.slots if s < 1 or s > total]
    if invalid:
        raise HTTPException(400, f"Invalid slots: {invalid}")

    for s in body.slots:
        b = _ensure_base(s)
        b.status = "running"
        b.updatedAt = _now_ms()

    try:
        items = await AGENT.generate_grid(
            rows=rows, cols=cols,
            prompt=CURRENT_PROMPT,
            ingestion_id=CURRENT_INGESTION_ID,
            input_url=CURRENT_INPUT_URL,
        )
        for it in items:
            if it.slot in body.slots:
                GRID_ITEMS[it.slot] = it
    except Exception as e:
        raise HTTPException(502, f"Agent regenerate error: {e}")

    prog = _progress(rows, cols)
    status = _grid_status(rows, cols)
    return GridResp(status=status, rows=rows, cols=cols, progress=prog,
                    items=[_ensure_base(s) for s in range(1, total + 1)])

# =========================================================
# Optional webhooks for agent to push updates
# =========================================================
@app.post("/slot/{slot}/status", response_model=ImageItem)
def slot_status_webhook(slot: int, payload: SlotUpdate, x_adgrid_secret: Optional[str] = Header(None)):
    if WEBHOOK_SECRET:
        if not x_adgrid_secret or x_adgrid_secret != WEBHOOK_SECRET:
            raise HTTPException(401, "Unauthorized")

    rows, cols = GRID_SHAPE["rows"], GRID_SHAPE["cols"]
    total = rows * cols
    if not (1 <= slot <= total):
        raise HTTPException(400, "Invalid slot")

    base = _ensure_base(slot)
    if payload.status:
        base.status = payload.status
    if payload.url is not None:
        base.url = payload.url
    if payload.meta is not None:
        base.meta = payload.meta
    base.version = payload.version if payload.version is not None else (base.version + 1)
    base.updatedAt = _now_ms()
    return base

@app.post("/slot/{slot}/variations/{variant}/status", response_model=SlotResp)
def variation_status_webhook(slot: int, variant: int, payload: SlotUpdate, x_adgrid_secret: Optional[str] = Header(None)):
    if WEBHOOK_SECRET:
        if not x_adgrid_secret or x_adgrid_secret != WEBHOOK_SECRET:
            raise HTTPException(401, "Unauthorized")

    base = _ensure_base(slot)
    variations = VAR_ITEMS.get(slot, [])
    v = next((x for x in variations if x.variant == variant), None)
    if v is None:
        v = ImageItem(slot=slot, variant=variant, kind="variation",
                      url=None, status="running", version=base.version, updatedAt=_now_ms())
        variations.append(v)

    if payload.status:
        v.status = payload.status
    if payload.url is not None:
        v.url = payload.url
    if payload.meta is not None:
        v.meta = payload.meta
    v.version = payload.version if payload.version is not None else (v.version + 1)
    v.updatedAt = _now_ms()
    VAR_ITEMS[slot] = variations
    return SlotResp(slot=slot, items=[base] + VAR_ITEMS[slot])

# =========================================================
# Reset all state
# =========================================================
@app.delete("/reset")
def reset_all():
    global CURRENT_PROMPT, CURRENT_INPUT_URL, CURRENT_INGESTION_ID
    CURRENT_PROMPT = None
    CURRENT_INPUT_URL = None
    CURRENT_INGESTION_ID = None
    GRID_ITEMS.clear()
    VAR_ITEMS.clear()
    return {"ok": True}
