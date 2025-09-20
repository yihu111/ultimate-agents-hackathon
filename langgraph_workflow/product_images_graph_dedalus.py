# product_images/product_images_graph.py
from typing import TypedDict, Dict, List, Optional, Annotated, NotRequired
from operator import add

from pydantic import BaseModel, Field, constr
from dotenv import load_dotenv, find_dotenv
from pathlib import Path
import asyncio
import os
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
from dedalus_labs import AsyncDedalus, DedalusRunner
from dedalus_labs.utils.streaming import stream_async

from langgraph_workflow.dedalus_prompts import (
    DIMENSIONS_SYSTEM,
    build_dimensions_human,
    PROMPT_SYSTEM,
    build_prompt_human,
)

load_dotenv(find_dotenv())

# ---------- Global async-safe counters for observed image tool calls ----------
from threading import Lock as _Lock
_IMAGE_CALLS_TOTAL: int = 0
_IMAGE_URLS_SEEN: set[str] = set()
_IMAGE_COUNTER_LOCK = _Lock()

def _note_image_tool_call(url: str) -> None:
    global _IMAGE_CALLS_TOTAL
    try:
        with _IMAGE_COUNTER_LOCK:
            if url not in _IMAGE_URLS_SEEN:
                _IMAGE_URLS_SEEN.add(url)
                _IMAGE_CALLS_TOTAL += 1
                print(f"[image-count] {_IMAGE_CALLS_TOTAL}")
    except Exception:
        # best-effort; never block the graph on metrics
        pass


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
    saw_image_successfully: bool = Field(
        description="Whether the model was able to load and view the provided image URL (if any)."
    )
    image_description: Optional[str] = Field(
        default=None,
        description="Short description of the image content if it was seen."
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
    # Debug from propose_dimensions
    saw_image_successfully: NotRequired[bool]
    image_description: NotRequired[str]


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
    image_url = state.get("initial_image_path")  # may now be an HTTP URL
    raw = await llm.ainvoke([
        AIMessage(content=DIMENSIONS_SYSTEM),
        build_dimensions_human(num_dims, num_vals, state["product_description"], state.get("extra_guidance"), image_url)
    ])
    plan = VariationPlan.model_validate(raw) if isinstance(raw, dict) else raw
    
    print(plan)

    return {
        "dim_a_name": plan.dim_a_name,
        "dim_b_name": plan.dim_b_name,
        "dim_a_values": plan.dim_a_values,
        "dim_b_values": plan.dim_b_values,
        # pass through debug flags
        "saw_image_successfully": plan.saw_image_successfully,
        "image_description": plan.image_description or "",
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
    # 1) Create combined prompt (system + human) for Dedalus
    human_msg = build_prompt_human(
        request.product_description,
        request.extra_guidance,
        request.dim_a_name,
        request.dim_b_name,
        request.dim_a_value,
        request.dim_b_value,
    )
    human_text_parts = []
    for part in getattr(human_msg, "content", []) or []:
        if isinstance(part, dict) and part.get("type") == "text":
            human_text_parts.append(part.get("text", ""))
    combined_prompt = f"{PROMPT_SYSTEM}\n\n" + "\n".join(human_text_parts)

    # Helper: extract image_url JSON robustly
    def _extract_image_url(text: str) -> str:
        import re
        # 1) Direct JSON
        try:
            data = json.loads(text)
            if isinstance(data, dict) and isinstance(data.get("image_url"), str):
                return data["image_url"]
        except Exception:
            pass
        # 2) Code-fenced JSON
        try:
            m = re.search(r"```(?:json)?\s*({[\s\S]*?})\s*```", text)
            if m:
                data = json.loads(m.group(1))
                if isinstance(data, dict) and isinstance(data.get("image_url"), str):
                    return data["image_url"]
        except Exception:
            pass
        # 3) Any object containing image_url
        try:
            m = re.search(r"\{[\s\S]*?\"image_url\"\s*:\s*\"([^\"]+)\"[\s\S]*?\}", text)
            if m:
                return m.group(1)
        except Exception:
            pass
        # 4) Fallback: first URL or data URL in text
        try:
            m = re.search(r"(https?://\S+|data:image/[a-zA-Z]+;base64,[A-Za-z0-9+/=]+)", text)
            if m:
                return m.group(1)
        except Exception:
            pass
        # 5) Give up with context for debugging
        preview = (text or "")[:500]
        raise RuntimeError(f"Could not extract image_url JSON from Dedalus output. Preview: {preview}")

    # Helper: persist returned image_url (data URL, http(s), or local path)
    def _save_image(image_url: str) -> Path:
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

        if image_url.startswith("data:"):
            try:
                header, b64data = image_url.split(",", 1)
            except ValueError:
                header, b64data = "data:image/png;base64", image_url
            mime = header.split(";")[0].split(":")[-1]
            ext_guess = mime.split("/")[-1] if "/" in mime else "png"
            ext = f".{ext_guess}" if not ext_guess.startswith(".") else ext_guess
            filename = f"{safe_a}__{safe_b}__{uuid.uuid4().hex[:8]}{ext}"
            saved_path = out_images / filename
            saved_path.write_bytes(base64.b64decode(b64data))
            return saved_path

        if image_url.startswith(("http://", "https://")):
            resp = requests.get(image_url, timeout=30)
            resp.raise_for_status()
            ext = _ext_from_url(image_url)
            filename = f"{safe_a}__{safe_b}__{uuid.uuid4().hex[:8]}{ext}"
            saved_path = out_images / filename
            saved_path.write_bytes(resp.content)
            return saved_path

        # Otherwise, assume local filesystem path
        src = Path(image_url)
        ext = src.suffix if src.suffix else ".png"
        filename = f"{safe_a}__{safe_b}__{uuid.uuid4().hex[:8]}{ext}"
        saved_path = Path(request.images_dir) / filename
        try:
            shutil.copy(src, saved_path)
        except Exception:
            # If copy fails, fallback to original path
            saved_path = src
        return saved_path

    # 2) Call Dedalus with Flux MCP tool and parse {image_url}
    client = AsyncDedalus()
    runner = DedalusRunner(client)
    final_text: str = ""
    # Optional streaming (enable via env var DEDALUS_STREAM=1)
    if os.getenv("DEDALUS_STREAM"):
        events = []
        seen_urls: set[str] = set()
        try:
            res = stream_async(runner.run(
                input=combined_prompt,
                model=request.model_spec["name"],
                mcp_servers=["wfoster/flux-mcp"],
                stream=True,
            ))
            events_gen = res
            if asyncio.iscoroutine(res):
                events_gen = await res
            if events_gen is None:
                raise TypeError("stream_async returned None")
            async for ev in events_gen:
                try:
                    # Best-effort console streaming of token deltas
                    if isinstance(ev, dict) and ev.get("delta"):
                        print(ev["delta"], end="", flush=True)
                    # Surface tool-call events in logs if present
                    if isinstance(ev, dict):
                        label = ev.get("event") or ev.get("type")
                        tool = ev.get("tool") or ev.get("tool_name")
                        if label or tool:
                            print(f"\n[dedalus-event] {label or 'event'} {tool or ''}")
                    # Opportunistically detect image URLs/data in the event and emit/save immediately
                    import re as _re
                    blob = ev if isinstance(ev, str) else json.dumps(ev, ensure_ascii=False)
                    m = _re.search(r"(https?://\S+|data:image/[a-zA-Z]+;base64,[A-Za-z0-9+/=]+)", blob)
                    if m:
                        url = m.group(1)
                        if url not in seen_urls:
                            seen_urls.add(url)
                            print(f"\n[image-url] {url}")
                            try:
                                saved_intermediate = _save_image(url)
                                print(f"[image-saved] {saved_intermediate}")
                            except Exception as _e:
                                print(f"[image-save-error] {type(_e).__name__}: {_e}")
                            _note_image_tool_call(url)
                except Exception:
                    pass
                events.append(ev)
        except Exception as e:
            print(f"[streaming-disabled] {type(e).__name__}: {e}")
        # Try to recover final output from events
        for ev in reversed(events):
            try:
                if isinstance(ev, dict) and isinstance(ev.get("final_output"), str) and ev["final_output"].strip():
                    final_text = ev["final_output"].strip()
                    break
            except Exception:
                continue
        # Fallback to non-streaming one-shot if we didn't capture a final output
        if not final_text:
            resp = await runner.run(
                input=combined_prompt,
                model=request.model_spec["name"],
                mcp_servers=["wfoster/flux-mcp"],
                stream=False,
            )
            final_text = (resp.final_output or "").strip()
    else:
        resp = await runner.run(
            input=combined_prompt,
            model=request.model_spec["name"],
            mcp_servers=["wfoster/flux-mcp"],
            stream=False,
        )
        final_text = (resp.final_output or "").strip()
    image_url = _extract_image_url(final_text)
    print(f"\n[dedalus-final-url] {image_url}")
    _note_image_tool_call(image_url)

    # 3) Persist outputs into run directories
    out_images = Path(request.images_dir)
    out_prompts = Path(request.prompts_dir)

    def _slug(text: str) -> str:
        t = text.lower().strip().replace(" ", "-")
        return re.sub(r"[^a-z0-9_-]+", "", t)

    safe_a = _slug(request.dim_a_value)
    safe_b = _slug(request.dim_b_value)

    # Save image from URL/data/local path coming from Dedalus
    saved_path = _save_image(image_url)
    print(f"[image-saved] {saved_path}")

    # Save prompt text alongside
    prompt_filename = saved_path.stem + ".txt"
    (out_prompts / prompt_filename).write_text(combined_prompt, encoding="utf-8")
    print(f"[prompt-saved] {out_prompts / prompt_filename}")
    try:
        with _IMAGE_COUNTER_LOCK:
            print(f"[image-calls-total] {_IMAGE_CALLS_TOTAL}")
    except Exception:
        pass

    # TODO: streaming: emit/send partial result to a sink or use LangGraph .stream events externally

    return {
        "results": [{
            "dim_a_value": request.dim_a_value,
            "dim_b_value": request.dim_b_value,
            "prompt_text": combined_prompt,
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