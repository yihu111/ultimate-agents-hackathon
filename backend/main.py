import io, os, time, base64
from typing import Dict, Optional, List, Literal
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image
from dotenv import load_dotenv
from supabase import create_client, Client
import httpx

# =========================================================
# Load env
# =========================================================
load_dotenv()

# Supabase config
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_ANON_KEY = os.environ.get("SUPABASE_ANON_KEY")
BUCKET = "images"  # your bucket name

if not SUPABASE_URL or not SUPABASE_ANON_KEY:
    raise RuntimeError("Set SUPABASE_URL and SUPABASE_ANON_KEY in your .env")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)

def public_url(path: str) -> str:
    """Build a public URL for an object (bucket must be public)."""
    base = SUPABASE_URL.rstrip("/")
    return f"{base}/storage/v1/object/public/{BUCKET}/{path.lstrip('/')}"

def supabase_upload_bytes(path: str, data: bytes, content_type: str) -> str:
    """Upload bytes to Supabase Storage (upsert), return public URL."""
    # Upsert so you can overwrite during demo
    supabase.storage.from_(BUCKET).upload(
        path, data, {"content-type": content_type, "upsert": True}
    )
    return public_url(path)

# Agent config
AGENT_BASE_URL = os.environ.get("AGENT_BASE_URL", "http://localhost:9000")
AGENT_API_KEY = os.environ.get("AGENT_API_KEY")

WEBHOOK_SECRET = os.environ.get("ADGRID_WEBHOOK_SECRET")  # optional shared secret for webhooks

# =========================================================
# FastAPI setup
# =========================================================
app = FastAPI(title="AdGrid + Agent + Supabase API", version="7.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten for prod
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

# Base items (variant=0) and variations (variant>=1)
GRID_ITEMS: Dict[int, "ImageItem"] = {}
VAR_ITEMS: Dict[int, List["ImageItem"]] = {}

# =========================================================
# Models
# =========================================================
ImageKind = Literal["base", "variation"]
ImageStatus = Literal["queued", "running", "done", "error"]

class ImageItem(BaseModel):
    slot: int
    variant: int                 # 0 for base, >=1 for variations
    url: Optional[str]           # may be None while queued/running
    kind: ImageKind              # "base" | "variation"
    status: ImageStatus = "running"
    version: int = 0             # bump on each update
    updatedAt: int = 0           # epoch ms
    meta: Optional[Dict] = None  # agent-supplied metadata

class UploadResp(BaseModel):
    prompt: Optional[str]
    input_url: Optional[str]
    ingestion_id: Optional[str] = None

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
        # queued if nothing started, running otherwise
        any_started = any(GRID_ITEMS.get(s) for s in range(1, rows * cols + 1))
        return "running" if any_started else "queued"
    if prog["done"] < (rows * cols):
        return "running"
    return "done"

# =========================================================
# Agent Client (HTTP)
# - Adjust paths/body to match your agent if needed
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
        """
        payload = {"prompt": prompt, "image_url": image_url}
        async with httpx.AsyncClient(timeout=60) as client:
            r = await client.post(f"{self.base_url}/ingest", headers=self._headers(), json=payload)
            r.raise_for_status()
            data = r.json() if r.headers.get("content-type","").startswith("application/json") else {}
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
            base_prev = _ensure_base(slot)
            out.append(ImageItem(
                slot=slot, variant=0, kind="base",
                url=url, status="done" if url else "running",
                version=base_prev.version + 1, updatedAt=_now_ms(),
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

# =========================================================
# Upload: save to Supabase (images/input.png|jpg) and notify Agent
# =========================================================
@app.post("/upload", response_model=UploadResp)
async def upload_reference(file: UploadFile = File(...), prompt: Optional[str] = Form(None)):
    global CURRENT_PROMPT, CURRENT_INPUT_URL, CURRENT_INGESTION_ID, GRID_ITEMS, VAR_ITEMS
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(400, "File must be an image (png/jpeg/webp…).")

    raw = await file.read()
    try:
        img = Image.open(io.BytesIO(raw)); img.verify()
    except Exception as e:
        raise HTTPException(400, f"Invalid image: {e}")

    ext = "png" if file.content_type.endswith("png") else "jpg"
    input_key = f"input.{ext}"

    try:
        CURRENT_INPUT_URL = supabase_upload_bytes(input_key, raw, file.content_type)
    except Exception as e:
        raise HTTPException(502, f"Supabase upload failed: {e}")

    CURRENT_PROMPT = prompt
    CURRENT_INGESTION_ID = None
    GRID_ITEMS.clear()
    VAR_ITEMS.clear()

    # Notify agent (optional but recommended)
    try:
        CURRENT_INGESTION_ID = await AGENT.ingest_url(image_url=CURRENT_INPUT_URL, prompt=CURRENT_PROMPT)
    except Exception:
        CURRENT_INGESTION_ID = None  # fine for hackathon

    return UploadResp(prompt=prompt, input_url= CURRENT_INPUT_URL, ingestion_id=CURRENT_INGESTION_ID)

# =========================================================
# Generate grid (Agent returns URLs in JSON)
# Frontend should poll GET /grid until status === "done"
# =========================================================
@app.post("/generate", response_model=GridResp)
async def generate_grid(rows: int = 3, cols: int = 3):
    global GRID_SHAPE
    GRID_SHAPE["rows"], GRID_SHAPE["cols"] = rows, cols
    total = rows * cols

    # mark all as running immediately so UI shows spinners
    for slot in range(1, total + 1):
        base = _ensure_base(slot)
        base.status = "running"
        base.updatedAt = _now_ms()

    # call agent for actual URLs
    try:
        items = await AGENT.generate_grid(
            rows=rows, cols=cols,
            prompt=CURRENT_PROMPT,
            ingestion_id=CURRENT_INGESTION_ID,
            input_url=CURRENT_INPUT_URL,
        )
        for it in items:
            GRID_ITEMS[it.slot] = it
    except Exception as e:
        raise HTTPException(502, f"Agent grid error: {e}")

    prog = _progress(rows, cols)
    status = _grid_status(rows, cols)
    return GridResp(status=status, rows=rows, cols=cols, progress=prog,
                    items=[GRID_ITEMS[s] for s in range(1, total + 1)])

# =========================================================
# Poll grid
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
# Per-slot regenerate (re-call agent for this slot)
# (We reuse generate_grid(rows=1, cols=1) and map result to this slot)
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
        items = await AGENT.generate_grid(
            rows=1, cols=1,
            prompt=CURRENT_PROMPT,
            ingestion_id=CURRENT_INGESTION_ID,
            input_url=CURRENT_INPUT_URL,
        )
        if items:
            result = items[0]
            result.slot = slot  # remap to target slot if agent returned slot=1
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
# Generate variations (Agent returns 3 URLs, append to state)
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

    # Mark running
    for s in body.slots:
        b = _ensure_base(s)
        b.status = "running"
        b.updatedAt = _now_ms()

    # Simple approach: re-call agent for the full grid and only replace requested slots
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
# Optional webhooks from Agent to update statuses/URLs
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
    lst = VAR_ITEMS.get(slot, [])
    v = next((x for x in lst if x.variant == variant), None)
    if v is None:
        v = ImageItem(slot=slot, variant=variant, kind="variation",
                      url=None, status="running", version=base.version, updatedAt=_now_ms())
        lst.append(v)

    if payload.status:
        v.status = payload.status
    if payload.url is not None:
        v.url = payload.url
    if payload.meta is not None:
        v.meta = payload.meta
    v.version = payload.version if payload.version is not None else (v.version + 1)
    v.updatedAt = _now_ms()
    VAR_ITEMS[slot] = lst
    return SlotResp(slot=slot, items=[base] + VAR_ITEMS[slot])

# =========================================================
# Reset
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
