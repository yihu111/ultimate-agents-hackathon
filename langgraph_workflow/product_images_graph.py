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

from langgraph.graph import StateGraph, START, END
from langgraph.types import Send
from langchain_core.messages import AIMessage
from langchain_openai import ChatOpenAI

from new_core.adapters.flux_adapter import FluxAdapter
from new_core.models.diffusion_prompt import DiffusionPrompt
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


# ---------- Nodes ----------

async def propose_dimensions(state: ProductImagesState) -> Dict:
    llm = ChatOpenAI(model=state["model_spec"]["name"]).with_structured_output(VariationPlan)
    raw = await llm.ainvoke([AIMessage(content=DIMENSIONS_SYSTEM),
                             build_dimensions_human(state["product_description"], state.get("extra_guidance"))])
    plan = VariationPlan.model_validate(raw) if isinstance(raw, dict) else raw
    return {
        "dim_a_name": plan.dim_a_name,
        "dim_b_name": plan.dim_b_name,
        "dim_a_values": plan.dim_a_values,
        "dim_b_values": plan.dim_b_values,
    }


async def start_variation_fanout(state: ProductImagesState) -> Dict:
    # no-op; used to anchor conditional SEND fan-out
    return {}


async def split_to_variations(state: ProductImagesState):
    dim_a_values = state.get("dim_a_values")
    dim_b_values = state.get("dim_b_values")
    if dim_a_values is None or dim_b_values is None:
        raise RuntimeError("Variation values not present in state; did 'propose_dimensions' run?")
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

    # 2) Generate image (no reference image for now)
    generator = FluxAdapter(model=request.image_model, use_raw_mode=request.use_raw_mode)
    img_path, meta = await generator.generate(DiffusionPrompt(text=prompt_out.prompt_text))

    # 3) Persist image into ./gen_images/ (relative to this file)
    out_dir = Path(__file__).parent / "gen_images"
    out_dir.mkdir(parents=True, exist_ok=True)

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
        saved_path = out_dir / filename
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
        saved_path = out_dir / filename
        saved_path.write_bytes(base64.b64decode(b64data))
    else:
        # Assume local filesystem path
        src = Path(img_path)
        ext = src.suffix if src.suffix else ".png"
        filename = f"{safe_a}__{safe_b}__{uuid.uuid4().hex[:8]}{ext}"
        saved_path = out_dir / filename
        try:
            shutil.copy(src, saved_path)
        except Exception:
            # If copy fails, just write nothing and fallback to original path
            saved_path = src

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

    builder.add_node("propose_dimensions", propose_dimensions)
    builder.add_node("start_variation_fanout", start_variation_fanout)
    builder.add_node("prompt_and_generate", prompt_and_generate)  # type: ignore[arg-type]
    builder.add_node("finalize", finalize)

    builder.add_edge(START, "propose_dimensions")
    builder.add_edge("propose_dimensions", "start_variation_fanout")
    builder.add_conditional_edges("start_variation_fanout", split_to_variations, ["prompt_and_generate"])
    builder.add_edge("prompt_and_generate", "finalize")
    builder.add_edge("finalize", END)

    return builder.compile()