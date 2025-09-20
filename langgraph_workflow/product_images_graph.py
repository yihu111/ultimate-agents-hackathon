# product_images/product_images_graph.py
from typing import TypedDict, Dict, List, Optional, Annotated, NotRequired
from operator import add

from pydantic import BaseModel, Field, constr
from dotenv import load_dotenv, find_dotenv
from pathlib import Path
import base64
import re
import uuid
import shutil
from urllib.parse import urlparse
import requests
from datetime import datetime
import json

from langgraph.graph import StateGraph, START, END
from langgraph.types import Send
from langchain_core.messages import AIMessage
from langchain_openai import ChatOpenAI

from product_images.adapters.flux_adapter import FluxAdapter
from product_images.models.diffusion_prompt import DiffusionPrompt
from product_images.prompts import (
    DIMENSIONS_SYSTEM,
    build_dimensions_human,
    PROMPT_SYSTEM,
    build_prompt_human,
)

load_dotenv(find_dotenv())


# ---------- Pydantic models (LLM structured outputs / requests) ----------

class VariationPlan(BaseModel):
    dim_a_name: str = Field(
        description="Short, human-readable name for the first variation dimension (e.g., 'palette')."
    )
    dim_b_name: str = Field(
        description="Short, human-readable name for the second variation dimension (e.g., 'tone')."
    )
    dim_a_values: List[str] = Field(
        description="Exactly 3 values for the first dimension.", min_length=3, max_length=3
    )
    dim_b_values: List[str] = Field(
        description="Exactly 3 values for the second dimension.", min_length=3, max_length=3
    )


class GeneratedDiffusionPrompt(BaseModel):
    prompt_text: str = Field(
        description="A complete, standalone diffusion prompt."
    )


class ModelSpec(TypedDict):
    name: str


class PromptAndGenerateRequest(BaseModel):
    model_spec: ModelSpec

    product_description: str
    extra_guidance: Optional[str] = None

    dim_a_name: str
    dim_b_name: str
    dim_a_value: str
    dim_b_value: str

    # Image generation config
    image_model: str
    use_raw_mode: bool
    # Optional reference image (same for all map nodes)
    reference_image_path: Optional[str] = None
    # Output directories (required for saving results)
    images_dir: str
    prompts_dir: str


# ---------- State ----------

class VariationResult(TypedDict):
    dim_a_value: str
    dim_b_value: str
    prompt_text: str
    img_path: str


class ProductImagesState(TypedDict):
    # LLM config
    model_spec: ModelSpec

    # Input (from user/frontend, but file path is fine for now)
    product_description: str
    extra_guidance: NotRequired[str]
    initial_image_path: NotRequired[str]  # reserved for future use; not used right now
    num_dimensions: NotRequired[int]
    num_values_per_dim: NotRequired[int]

    # Proposed variation axes
    dim_a_name: NotRequired[str]
    dim_b_name: NotRequired[str]
    dim_a_values: NotRequired[List[str]]
    dim_b_values: NotRequired[List[str]]

    # Image generation config for Flux
    image_model: str
    use_raw_mode: bool

    # Aggregated results from map phase
    results: Annotated[List[VariationResult], add]
    # Output directories (initialized at run start)
    run_dir: NotRequired[str]
    images_dir: NotRequired[str]
    prompts_dir: NotRequired[str]


# ---------- Nodes ----------

async def init_run_output(state: ProductImagesState) -> Dict:
    root = Path(__file__).parent / "gen_images"
    root.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + f"_{uuid.uuid4().hex[:6]}"
    run_dir = root / f"run_{run_id}"
    images_dir = run_dir / "images"
    prompts_dir = run_dir / "prompts"
    images_dir.mkdir(parents=True, exist_ok=True)
    prompts_dir.mkdir(parents=True, exist_ok=True)
    return {
        "run_dir": str(run_dir),
        "images_dir": str(images_dir),
        "prompts_dir": str(prompts_dir),
    }

async def propose_dimensions(state: ProductImagesState) -> Dict:
    llm = ChatOpenAI(model=state["model_spec"]["name"]).with_structured_output(VariationPlan)
    num_dims = state.get("num_dimensions", 2)
    num_vals = state.get("num_values_per_dim", 3)
    raw = await llm.ainvoke([
        AIMessage(content=DIMENSIONS_SYSTEM),
        build_dimensions_human(num_dims, num_vals, state["product_description"], state.get("extra_guidance"))
    ])
    plan = VariationPlan.model_validate(raw) if isinstance(raw, dict) else raw
    return {
        "dim_a_name": plan.dim_a_name,
        "dim_b_name": plan.dim_b_name,
        "dim_a_values": plan.dim_a_values,
        "dim_b_values": plan.dim_b_values,
    }


async def save_run_config(state: ProductImagesState) -> Dict:
    run_dir = state.get("run_dir")
    if not run_dir:
        raise RuntimeError("run_dir not initialised; did 'init_run_output' run?")

    cfg = {
        "timestamp": datetime.now().isoformat(),
        "model_spec": state.get("model_spec", {}),
        "product_description": state.get("product_description"),
        "extra_guidance": state.get("extra_guidance"),
        "initial_image_path": state.get("initial_image_path"),
        "num_dimensions": state.get("num_dimensions", 2),
        "num_values_per_dim": state.get("num_values_per_dim", 3),
        "dim_a_name": state.get("dim_a_name"),
        "dim_b_name": state.get("dim_b_name"),
        "dim_a_values": state.get("dim_a_values"),
        "dim_b_values": state.get("dim_b_values"),
        "image_model": state.get("image_model"),
        "use_raw_mode": state.get("use_raw_mode"),
        "images_dir": state.get("images_dir"),
        "prompts_dir": state.get("prompts_dir"),
    }

    out_path = Path(run_dir) / "config.jsonl"
    # Append a single-line JSON record (one run per file currently)
    with out_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(cfg, ensure_ascii=False) + "\n")

    return {}


async def start_variation_fanout(state: ProductImagesState) -> Dict:
    # no-op; used to anchor conditional SEND fan-out
    return {}


async def split_to_variations(state: ProductImagesState):
    dim_a_values = state.get("dim_a_values")
    dim_b_values = state.get("dim_b_values")
    if dim_a_values is None or dim_b_values is None:
        raise RuntimeError("Variation values not present in state; did 'propose_dimensions' run?")
    ref_img = state.get("initial_image_path")
    images_dir = state.get("images_dir")
    prompts_dir = state.get("prompts_dir")
    if not images_dir or not prompts_dir:
        raise RuntimeError("Output directories not initialized; did 'init_run_output' run?")
    sends: List[Send] = []
    for va in dim_a_values:
        for vb in dim_b_values:
            req = PromptAndGenerateRequest(
                model_spec=state["model_spec"],
                product_description=state["product_description"],
                extra_guidance=state.get("extra_guidance"),
                dim_a_name=state.get("dim_a_name") or "",
                dim_b_name=state.get("dim_b_name") or "",
                dim_a_value=va,
                dim_b_value=vb,
                image_model=state["image_model"],
                use_raw_mode=state["use_raw_mode"],
                reference_image_path=ref_img,
                images_dir=images_dir,
                prompts_dir=prompts_dir,
            )
            sends.append(Send("prompt_and_generate", req))
    return sends


async def prompt_and_generate(request: PromptAndGenerateRequest) -> Dict:
    # 1) Create prompt (LLM structured)
    llm = ChatOpenAI(model=request.model_spec["name"]).with_structured_output(GeneratedDiffusionPrompt)
    raw = await llm.ainvoke([
        AIMessage(content=PROMPT_SYSTEM),
        build_prompt_human(
            request.product_description,
            request.extra_guidance,
            request.dim_a_name,
            request.dim_b_name,
            request.dim_a_value,
            request.dim_b_value,
        )
    ])
    prompt_out = GeneratedDiffusionPrompt.model_validate(raw) if isinstance(raw, dict) else raw

    # 2) Generate image (optionally with reference image if present in request)
    generator = FluxAdapter(model=request.image_model, use_raw_mode=request.use_raw_mode)
    img_path, meta = await generator.generate(
        prompt_out.prompt_text,
        input_image=request.reference_image_path
    )

    # 3) Persist outputs into run directories
    out_images = Path(request.images_dir)
    out_prompts = Path(request.prompts_dir)

    def _slug(text: str) -> str:
        t = text.lower().strip().replace(" ", "-")
        return re.sub(r"[^a-z0-9_-]+", "", t)

    safe_a = _slug(request.dim_a_value)
    safe_b = _slug(request.dim_b_value)

    def _ext_from_url(u: str) -> str:
        path = urlparse(u).path
        suf = Path(path).suffix
        return suf if suf else ".png"

    saved_path: Path
    if isinstance(img_path, str) and img_path.startswith("http"):
        ext = _ext_from_url(img_path)
        filename = f"{safe_a}__{safe_b}__{uuid.uuid4().hex[:8]}{ext}"
        saved_path = out_images / filename
        resp = requests.get(img_path, timeout=30)
        resp.raise_for_status()
        saved_path.write_bytes(resp.content)
    elif isinstance(img_path, str) and img_path.startswith("data:"):
        # data URL: data:image/png;base64,....
        try:
            header, b64data = img_path.split(",", 1)
        except ValueError:
            header, b64data = "data:image/png;base64", img_path
        mime = header.split(";")[0].split(":")[-1]
        ext_guess = mime.split("/")[-1] if "/" in mime else "png"
        ext = f".{ext_guess}" if not ext_guess.startswith(".") else ext_guess
        filename = f"{safe_a}__{safe_b}__{uuid.uuid4().hex[:8]}{ext}"
        saved_path = out_images / filename
        saved_path.write_bytes(base64.b64decode(b64data))
    else:
        # Assume local filesystem path
        src = Path(img_path)
        ext = src.suffix if src.suffix else ".png"
        filename = f"{safe_a}__{safe_b}__{uuid.uuid4().hex[:8]}{ext}"
        saved_path = out_images / filename
        try:
            shutil.copy(src, saved_path)
        except Exception:
            # If copy fails, just write nothing and fallback to original path
            saved_path = src

    # Save prompt text alongside
    prompt_filename = saved_path.stem + ".txt"
    (out_prompts / prompt_filename).write_text(prompt_out.prompt_text, encoding="utf-8")

    # TODO: streaming: emit/send partial result to a sink or use LangGraph .stream events externally

    return {
        "results": [{
            "dim_a_value": request.dim_a_value,
            "dim_b_value": request.dim_b_value,
            "prompt_text": prompt_out.prompt_text,
            "img_path": str(saved_path),
        }]
    }


async def finalize(state: ProductImagesState) -> Dict:
    # barrier node; could sort or post-process results here
    return {}


# ---------- Graph ----------

def compile_graph():
    builder = StateGraph(ProductImagesState)

    builder.add_node("init_run_output", init_run_output)
    builder.add_node("propose_dimensions", propose_dimensions)
    builder.add_node("save_run_config", save_run_config)
    builder.add_node("start_variation_fanout", start_variation_fanout)
    builder.add_node("prompt_and_generate", prompt_and_generate)  # type: ignore[arg-type]
    builder.add_node("finalize", finalize)

    builder.add_edge(START, "init_run_output")
    builder.add_edge("init_run_output", "propose_dimensions")
    builder.add_edge("propose_dimensions", "save_run_config")
    builder.add_edge("save_run_config", "start_variation_fanout")
    builder.add_conditional_edges("start_variation_fanout", split_to_variations, ["prompt_and_generate"])
    builder.add_edge("prompt_and_generate", "finalize")
    builder.add_edge("finalize", END)

    return builder.compile()