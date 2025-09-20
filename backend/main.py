import os, io, time, shutil
from typing import Dict, Optional, List
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from PIL import Image, ImageDraw

# ===== Config =====
IMAGES_DIR = "images"
VARIATIONS_SUBDIR = "variations"  # images/variations/<slot>/<idx>.png
os.makedirs(IMAGES_DIR, exist_ok=True)

app = FastAPI(title="AdGrid Single-Grid API", version="1.2.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve /images/* statically
app.mount("/images", StaticFiles(directory=IMAGES_DIR), name="images")

# ===== Models =====
class UploadResp(BaseModel):
    input_url: str
    input_path: str

class GridResp(BaseModel):
    status: str
    rows: int
    cols: int
    progress: Dict[str, int]
    input_url: Optional[str]
    images: Dict[str, Optional[str]]

class RegenReq(BaseModel):
    slots: List[int]

class VariationsGenerateReq(BaseModel):
    count: int = 3  # how many new variations to create for this slot

class VariationsResp(BaseModel):
    slot: int
    urls: List[str]

# ===== Helpers =====
def input_path(ext="png"):
    return os.path.join(IMAGES_DIR, f"input.{ext}")

def slot_path(slot: int):
    return os.path.join(IMAGES_DIR, f"{slot}.png")

def var_dir(slot: int):
    return os.path.join(IMAGES_DIR, VARIATIONS_SUBDIR, str(slot))

def var_path(slot: int, idx: int):
    return os.path.join(var_dir(slot), f"{idx}.png")

def file_url(path: str) -> str:
    ts = int(os.path.getmtime(path)) if os.path.exists(path) else int(time.time())
    rel = os.path.relpath(path, ".").replace("\\", "/")
    return f"/{rel}?ts={ts}"

def placeholder_png(text="Generating…") -> bytes:
    img = Image.new("RGB", (1024, 1024), (240, 240, 240))
    d = ImageDraw.Draw(img)
    d.text((24, 24), text, fill=(90, 90, 90))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()

def write_bytes(path: str, data: bytes):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(data)

def compute_grid(rows=3, cols=3) -> GridResp:
    total = rows * cols
    done = 0
    images: Dict[str, Optional[str]] = {}
    for i in range(1, total + 1):
        p = slot_path(i)
        if os.path.exists(p) and os.path.getsize(p) > 0:
            done += 1
            images[str(i)] = file_url(p)
        else:
            images[str(i)] = None
    status = "done" if done == total else ("running" if done > 0 else "queued")
    inp_png = input_path("png")
    inp_jpg = input_path("jpg")
    inp = inp_png if os.path.exists(inp_png) else inp_jpg if os.path.exists(inp_jpg) else None
    return GridResp(
        status=status,
        rows=rows,
        cols=cols,
        progress={"done": done, "total": total},
        input_url=file_url(inp) if inp else None,
        images=images,
    )

def clear_images_dir() -> dict:
    count = 0
    if os.path.exists(IMAGES_DIR):
        for root, _, files in os.walk(IMAGES_DIR):
            for fn in files:
                try:
                    os.remove(os.path.join(root, fn))
                    count += 1
                except Exception:
                    pass
        shutil.rmtree(IMAGES_DIR, ignore_errors=True)
    os.makedirs(IMAGES_DIR, exist_ok=True)
    return {"deleted_files": count, "folder": IMAGES_DIR}

def list_variations(slot: int) -> List[str]:
    d = var_dir(slot)
    if not os.path.isdir(d):
        return []
    items = sorted(
        [f for f in os.listdir(d) if f.endswith(".png")],
        key=lambda name: int(os.path.splitext(name)[0]) if os.path.splitext(name)[0].isdigit() else 0
    )
    return [file_url(os.path.join(d, f)) for f in items]

# ===== Endpoints =====
@app.post("/upload", response_model=UploadResp)
async def upload_reference(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(400, "File must be an image")
    if file.content_type not in {"image/png", "image/jpeg"}:
        raise HTTPException(400, "Only PNG or JPG allowed")

    raw = await file.read()
    try:
        img = Image.open(io.BytesIO(raw))
        img.verify()
    except Exception as e:
        raise HTTPException(400, f"Invalid image: {e}")

    ext = "png" if file.content_type == "image/png" else "jpg"
    ipath = input_path(ext)
    write_bytes(ipath, raw)
    return UploadResp(input_url=file_url(ipath), input_path=ipath)

@app.post("/generate", response_model=GridResp)
def generate_grid(rows: int = 3, cols: int = 3):
    total = rows * cols
    for i in range(1, total + 1):
        write_bytes(slot_path(i), placeholder_png(text=f"Slot {i}"))
    return compute_grid(rows, cols)

@app.get("/grid", response_model=GridResp)
def get_grid(rows: int = 3, cols: int = 3):
    return compute_grid(rows, cols)

@app.post("/regenerate", response_model=GridResp)
def regenerate_slots(req: RegenReq, rows: int = 3, cols: int = 3):
    total = rows * cols
    invalid = [s for s in req.slots if s < 1 or s > total]
    if invalid:
        raise HTTPException(400, f"Invalid slot numbers: {invalid}")
    for s in req.slots:
        write_bytes(slot_path(s), placeholder_png(text=f"Regen {s}"))
    return compute_grid(rows, cols)

@app.delete("/reset")
def reset_all():
    info = clear_images_dir()
    return {"ok": True, **info}

# ----- NEW: Variations -----

@app.post("/variations/{slot}/generate", response_model=VariationsResp)
def generate_variations(slot: int, body: VariationsGenerateReq = VariationsGenerateReq()):
    """
    Create N new variations derived from images/{slot}.png.
    Saves them under images/variations/{slot}/1.png, 2.png, ... (appends new indices).
    Swap placeholder generation with your real model using the chosen slot as a reference.
    """
    base = slot_path(slot)
    if not os.path.exists(base):
        raise HTTPException(404, f"Base image for slot {slot} not found. Generate the grid first.")

    os.makedirs(var_dir(slot), exist_ok=True)

    # determine next index to append (existing count + 1)
    existing = list_variations(slot)
    start_idx = len(existing) + 1

    # Generate 'count' new variations
    for i in range(start_idx, start_idx + max(1, body.count)):
        # TODO: replace with real variation bytes from your model using `base` as input
        png = placeholder_png(text=f"Var {slot}-{i}")
        write_bytes(var_path(slot, i), png)

    return VariationsResp(slot=slot, urls=list_variations(slot))

@app.get("/variations/{slot}", response_model=VariationsResp)
def get_variations(slot: int):
    return VariationsResp(slot=slot, urls=list_variations(slot))

@app.delete("/variations/{slot}")
def delete_variations(slot: int):
    d = var_dir(slot)
    if os.path.isdir(d):
        shutil.rmtree(d, ignore_errors=True)
        os.makedirs(d, exist_ok=True)
    return {"ok": True, "slot": slot, "cleared": True}
