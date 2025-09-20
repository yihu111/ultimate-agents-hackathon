import io, os, time, base64
from typing import Dict, Optional, List, Literal
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image
import httpx
from dotenv import load_dotenv
load_dotenv()

# =========================================================
# App & CORS
# =========================================================
app = FastAPI(title="AdGrid + Agent API", version="5.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten for prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================================================
# Config (Agent)
# =========================================================
AGENT_BASE_URL = os.environ.get("AGENT_BASE_URL", "http://localhost:9000")
AGENT_API_KEY = os.environ.get("AGENT_API_KEY")  # optional
# If your agent supports pre-ingest, keep True. If not, set to False.
AGENT_SUPPORTS_INGEST = os.environ.get("AGENT_SUPPORTS_INGEST", "true").lower() == "true"

# Optional shared secret for webhook auth (if you use the webhook)
WEBHOOK_SECRET = os.environ.get("ADGRID_WEBHOOK_SECRET")  # set to None/"" to disable

# =========================================================
# In-memory state
# =========================================================
CURRENT_PROMPT: Optional[str] = None
CURRENT_INPUT: Optional[bytes] = None
CURRENT_INGESTION_ID: Optional[str] = None
GRID_SHAPE = {"rows": 3, "cols": 3}

GRID_ITEMS: Dict[int, "ImageItem"] = {}           # base items (variant=0)
VAR_ITEMS: Dict[int, List["ImageItem"]] = {}      # variations (variant>=1)

# =========================================================
# Models
# =========================================================
ImageKind = Literal["base", "variation"]
ImageStatus = Literal["queued", "running", "done", "error"]

class ImageItem(BaseModel):
    slot: int
    variant: int                 # 0 = base, >=1 = variation
    url: Optional[str]           # may be None while queued/running
    kind: ImageKind              # "base" | "variation"
    status: ImageStatus = "done"
    version: int = 0
    updatedAt: int = 0
    meta: Optional[Dict] = None  # any payload from the agent

class UploadResp(BaseModel):
    prompt: Optional[str]
    ingestion_id: Optional[str] = None

class GridResp(BaseModel):
    status: ImageStatus
    rows: int
    cols: int
    progress: Dict[str, int]
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
        return "queued"
    if prog["done"] < (rows * cols):
        return "running"
    return "done"

def _ensure_base(slot: int) -> ImageItem:
    it = GRID_ITEMS.get(slot)
    if it is None:
        it = ImageItem(slot=slot, variant=0, kind="base", url=None,
                       status="queued", version=0, updatedAt=_now_ms())
        GRID_ITEMS[slot] = it
    return it

# =========================================================
# Agent Client
# (Replace endpoint paths to match your Agent)
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

    async def ingest(self, image_bytes: bytes, prompt: Optional[str]) -> Optional[str]:
        """
        Optional pre-ingest, returns ingestion_id.
        Expected Agent endpoint: POST /ingest  { prompt, image_b64 }
        """
        payload = {
            "prompt": prompt,
            "image_b64": base64.b64encode(image_bytes).decode("utf-8"),
        }
        async with httpx.AsyncClient(timeout=60) as client:
            r = await client.post(f"{self.base_url}/ingest", headers=self._headers(), json=payload)
            if r.status_code >= 400:
                # If agent doesn't support ingest or errors out, just skip ingest.
                return None
            data = r.json()
            return data.get("ingestion_id")

    async def generate_grid(self, rows: int, cols: int,
                            prompt: Optional[str],
                            ingestion_id: Optional[str],
                            image_bytes: Optional[bytes]) -> List[ImageItem]:
        """
        Ask the agent for a full grid (base images, 1..rows*cols).
        Expected Agent response JSON:
        {
          "items": [
            {"slot": 1, "url": "...", "meta": {...}},
            ... up to rows*cols (slot numbering 1..N)
          ]
        }
        """
        payload = {"rows": rows, "cols": cols, "prompt": prompt}
        if ingestion_id:
            payload["ingestion_id"] = ingestion_id
        else:
            # fall back to sending bytes if no ingest
            if image_bytes is not None:
                payload["image_b64"] = base64.b64encode(image_bytes).decode("utf-8")

        async with httpx.AsyncClient(timeout=None) as client:
            r = await client.post(f"{self.base_url}/grid", headers=self._headers(), json=payload)
            r.raise_for_status()
            data = r.json()

        # Map agent response to ImageItem list
        out: List[ImageItem] = []
        for raw in data.get("items", []):
            slot = int(raw["slot"])
            url = raw.get("url")
            out.append(ImageItem(
                slot=slot,
                variant=0,
                url=url,
                kind="base",
                status="done" if url else "running",
                version=(_ensure_base(slot).version + 1),
                updatedAt=_now_ms(),
                meta=raw.get("meta"),
            ))
        return out

    async def generate_variations(self, slot: int, count: int,
                                  prompt: Optional[str],
                                  ingestion_id: Optional[str],
                                  base_item: ImageItem) -> List[ImageItem]:
        """
        Ask the agent for N variations for a given slot.
        Expected Agent response JSON:
        {
          "items": [
            {"slot": <same>, "variant": 1, "url": "...", "meta": {...}},
            {"slot": <same>, "variant": 2, "url": "...", "meta": {...}},
            {"slot": <same>, "variant": 3, "url": "...", "meta": {...}}
          ]
        }
        """
        payload = {
            "slot": slot,
            "count": count,
            "prompt": prompt,
            "base_version": base_item.version,
        }
        if ingestion_id:
            payload["ingestion_id"] = ingestion_id

        async with httpx.AsyncClient(timeout=None) as client:
            r = await client.post(f"{self.base_url}/variations", headers=self._headers(), json=payload)
            r.raise_for_status()
            data = r.json()

        out: List[ImageItem] = []
        existing = VAR_ITEMS.get(slot, [])
        used = set(v.variant for v in existing)
        # Assign variants either from agent or auto-increment
        next_idx = (max(used) + 1) if used else 1
        for raw in data.get("items", []):
            variant = int(raw.get("variant") or next_idx)
            if not raw.get("variant"):
                next_idx += 1
            url = raw.get("url")
            out.append(ImageItem(
                slot=slot,
                variant=variant,
                kind="variation",
                url=url,
                status="done" if url else "running",
                version=base_item.version,
                updatedAt=_now_ms(),
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
# Upload image + context (prompt); send to Agent (ingest)
# =========================================================
@app.post("/upload", response_model=UploadResp)
async def upload_reference(file: UploadFile = File(...), prompt: Optional[str] = Form(None)):
    global CURRENT_INPUT, CURRENT_PROMPT, CURRENT_INGESTION_ID, GRID_ITEMS, VAR_ITEMS
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(400, "File must be an image (png/jpeg/webp…).")

    raw = await file.read()
    try:
        img = Image.open(io.BytesIO(raw)); img.verify()
    except Exception as e:
        raise HTTPException(400, f"Invalid image: {e}")

    CURRENT_INPUT = raw
    CURRENT_PROMPT = prompt
    CURRENT_INGESTION_ID = None

    GRID_ITEMS.clear()
    VAR_ITEMS.clear()

    # Optional: ingest with the agent (if supported)
    if AGENT_SUPPORTS_INGEST:
        try:
            ing_id = await AGENT.ingest(image_bytes=raw, prompt=prompt)
            CURRENT_INGESTION_ID = ing_id
        except Exception:
            # swallow for hackathon simplicity; generation can still work by sending bytes later
            CURRENT_INGESTION_ID = None

    return UploadResp(prompt=prompt, ingestion_id=CURRENT_INGESTION_ID)

# =========================================================
# Generate whole grid via Agent; frontend polls GET /grid
# =========================================================
@app.post("/generate", response_model=GridResp)
async def generate_grid(rows: int = 3, cols: int = 3):
    global GRID_SHAPE
    GRID_SHAPE["rows"], GRID_SHAPE["cols"] = rows, cols
    total = rows * cols

    # Pre-fill placeholders (so /grid shows queued/running immediately)
    for slot in range(1, total + 1):
        base = _ensure_base(slot)
        if base.status != "done":
            base.status = "running"
            base.updatedAt = _now_ms()

    # Ask the agent for the final grid items
    try:
        items = await AGENT.generate_grid(
            rows=rows,
            cols=cols,
            prompt=CURRENT_PROMPT,
            ingestion_id=CURRENT_INGESTION_ID,
            image_bytes=CURRENT_INPUT if not AGENT_SUPPORTS_INGEST else None
        )
        # Merge into GRID_ITEMS
        for it in items:
            GRID_ITEMS[it.slot] = it
    except Exception as e:
        raise HTTPException(502, f"Agent grid error: {e}")

    prog = _progress(rows, cols)
    status = _grid_status(rows, cols)
    return GridResp(status=status, rows=rows, cols=cols, progress=prog,
                    items=[GRID_ITEMS[s] for s in range(1, total + 1)])

# =========================================================
# GET overall grid (poll)
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
# Regenerate a single base slot via Agent (fine-grained update)
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
    # You can call a dedicated agent endpoint; for simplicity reusing grid-generation for one slot:
    try:
        items = await AGENT.generate_grid(
            rows=1, cols=1,
            prompt=CURRENT_PROMPT,
            ingestion_id=CURRENT_INGESTION_ID,
            image_bytes=CURRENT_INPUT if not AGENT_SUPPORTS_INGEST else None
        )
        # Expect exactly 1 item (slot=1); remap to our target slot
        if items:
            gen = items[0]
            gen.slot = slot
            GRID_ITEMS[slot] = gen
        else:
            base.status = "error"
    except Exception as e:
        raise HTTPException(502, f"Agent slot error: {e}")
    return GRID_ITEMS[slot]

# =========================================================
# GET a single slot (base + variations)
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
# Generate N variations for a slot via Agent (append)
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
            slot=slot,
            count=max(1, body.count),
            prompt=CURRENT_PROMPT,
            ingestion_id=CURRENT_INGESTION_ID,
            base_item=base
        )
        VAR_ITEMS[slot] = VAR_ITEMS.get(slot, []) + new_items
    except Exception as e:
        raise HTTPException(502, f"Agent variations error: {e}")

    return SlotResp(slot=slot, items=[base] + VAR_ITEMS[slot])

# =========================================================
# Batch regenerate convenience (re-calls agent per slot)
# =========================================================
@app.post("/regenerate", response_model=GridResp)
async def regenerate_slots(body: RegenReq):
    rows, cols = GRID_SHAPE["rows"], GRID_SHAPE["cols"]
    total = rows * cols
    invalid = [s for s in body.slots if s < 1 or s > total]
    if invalid:
        raise HTTPException(400, f"Invalid slots: {invalid}")

    # mark running
    for s in body.slots:
        b = _ensure_base(s)
        b.status = "running"
        b.updatedAt = _now_ms()

    # call agent once for all? (simple approach: call generate_grid for the full grid and merge)
    try:
        items = await AGENT.generate_grid(
            rows=rows, cols=cols,
            prompt=CURRENT_PROMPT,
            ingestion_id=CURRENT_INGESTION_ID,
            image_bytes=CURRENT_INPUT if not AGENT_SUPPORTS_INGEST else None
        )
        for it in items:
            # Only replace requested slots
            if it.slot in body.slots:
                GRID_ITEMS[it.slot] = it
    except Exception as e:
        raise HTTPException(502, f"Agent regenerate error: {e}")

    prog = _progress(rows, cols)
    status = _grid_status(rows, cols)
    return GridResp(status=status, rows=rows, cols=cols, progress=prog,
                    items=[_ensure_base(s) for s in range(1, total + 1)])

# =========================================================
# Optional webhook: agent -> backend updates a slot
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
    base.status = payload.status
    if payload.url is not None:
        base.url = payload.url
    if payload.meta is not None:
        base.meta = payload.meta
    if payload.version is not None:
        base.version = payload.version
    else:
        base.version = (base.version or 0) + 1
    base.updatedAt = _now_ms()
    return base

# =========================================================
# Reset
# =========================================================
@app.delete("/reset")
def reset_all():
    global CURRENT_PROMPT, CURRENT_INPUT, CURRENT_INGESTION_ID
    CURRENT_PROMPT = None
    CURRENT_INPUT = None
    CURRENT_INGESTION_ID = None
    GRID_ITEMS.clear()
    VAR_ITEMS.clear()
    return {"ok": True}
