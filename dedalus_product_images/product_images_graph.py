# dedalus_product_images/product_images_graph.py
from typing import TypedDict, Dict, List, Optional, Annotated
from operator import add
import asyncio
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
from pydantic import BaseModel, Field

# Import Dedalus Labs components
from dedalus_labs import AsyncDedalus, DedalusRunner

# Import original components for local execution
# Note: FluxAdapter and DiffusionPrompt imports removed since we're using Dedalus agents
# from adapters.flux_adapter import FluxAdapter
# from models.diffusion_prompt import DiffusionPrompt

from dedalus_product_images.prompts import (
    DIMENSIONS_SYSTEM,
    build_dimensions_human,
    PROMPT_SYSTEM,
    build_prompt_human,
)

load_dotenv(find_dotenv())

# Python 3.10 compatibility
try:
    from typing import NotRequired
except ImportError:
    NotRequired = Optional


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
            sends.append(Send("dedalus_prompt_and_generate", req))
    return sends


async def dedalus_prompt_and_generate(request: PromptAndGenerateRequest) -> Dict:
    """
    Dedalus agent that handles both prompt generation and image creation using Dedalus agents.
    Uses Dedalus for both tasks instead of direct LangGraph functions.
    """
    print(f"🎨 Dedalus Agent: Generating image for {request.dim_a_value} x {request.dim_b_value}")
    
    # Initialize Dedalus client
    client = AsyncDedalus()
    runner = DedalusRunner(client)
    
    try:
        # 1) Use Dedalus agent for prompt generation
        prompt_query = f"""
        Generate a professional diffusion prompt for product image generation with the following specifications:
        
        Product Description: {request.product_description}
        {f"Extra Guidance: {request.extra_guidance}" if request.extra_guidance else ""}
        
        Target Variations:
        - {request.dim_a_name}: {request.dim_a_value}
        - {request.dim_b_name}: {request.dim_b_value}
        
        IMPORTANT: Return ONLY the final diffusion prompt text. Do not include any explanations, formatting, or additional text. Just the prompt that can be used directly for image generation.
        
        The prompt should follow this structure:
        Use the uploaded [object/product] image as the product reference.
        Scene: [describe the setting, background, and overall environment reflecting the dimension values]
        Composition: [specify placement of the product, framing, camera angle, and negative space]
        Lighting/Color: [describe light interaction and palette, integrating the dimension values]
        Output: [specify aspect ratio like 4:5, 9:16, etc.]
        """
        
        prompt_response = await runner.run(
            input=prompt_query,
            model="openai/gpt-4o",
            mcp_servers=[],  # No MCP needed for prompt generation
            stream=False
        )
        
        # Extract prompt text and clean it up
        prompt_text = prompt_response.final_output.strip()
        
        # Remove any potential formatting or extra text
        if "Use the uploaded" in prompt_text:
            # Extract just the prompt part
            lines = prompt_text.split('\n')
            prompt_lines = []
            for line in lines:
                if line.strip() and not line.strip().startswith(('Generate', 'IMPORTANT', 'The prompt', 'Return')):
                    prompt_lines.append(line.strip())
            prompt_text = '\n'.join(prompt_lines)
        
        # 2) Use Dedalus agent for image generation with Flux MCP
        image_query = f"""
        Use the Flux MCP server to generate an image with this prompt:
        
        {prompt_text}
        
        Image Generation Parameters:
        - Model: {request.image_model}
        - Raw Mode: {request.use_raw_mode}
        {f"- Reference Image: {request.reference_image_path}" if request.reference_image_path else ""}
        
        IMPORTANT: Use the Flux MCP tools to generate the image and return the result in one of these formats:
        1. Base64 encoded image data (data:image/png;base64,...)
        2. URL to the generated image
        3. File path to the generated image
        
        Do not include any explanations or additional text. Just return the image data/URL/path.
        """
        
        image_response = await runner.run(
            input=image_query,
            model="openai/gpt-4o-mini",
            mcp_servers=["yihu/flux-mcp"],  # Using the actual Flux MCP server
            stream=False
        )
        
        # 3) Parse the image response and save
        img_path = await parse_and_save_image_response(image_response.final_output, prompt_text, request)

        return {
            "results": [{
                "dim_a_value": request.dim_a_value,
                "dim_b_value": request.dim_b_value,
                "prompt_text": prompt_text,
                "img_path": str(img_path),
            }]
        }
        
    except Exception as e:
        print(f"❌ Error in Dedalus agent: {e}")
        # Fallback: create a placeholder result
        return {
            "results": [{
                "dim_a_value": request.dim_a_value,
                "dim_b_value": request.dim_b_value,
                "prompt_text": f"Error generating prompt: {str(e)}",
                "img_path": "error_placeholder.png",
            }]
        }


async def parse_and_save_image_response(image_response: str, prompt_text: str, request: PromptAndGenerateRequest) -> str:
    """
    Parse the Dedalus agent's image response and save the image and prompt.
    Handles unstructured text responses from Dedalus agents.
    """
    import base64
    import requests
    from urllib.parse import urlparse
    
    # Create output directories
    out_images = Path(request.images_dir)
    out_prompts = Path(request.prompts_dir)
    
    def _slug(text: str) -> str:
        t = text.lower().strip().replace(" ", "-")
        return re.sub(r"[^a-z0-9_-]+", "", t)
    
    safe_a = _slug(request.dim_a_value)
    safe_b = _slug(request.dim_b_value)
    
    filename = f"{safe_a}__{safe_b}__{uuid.uuid4().hex[:8]}.png"
    saved_path = out_images / filename
    
    # Parse the unstructured response from Dedalus agent
    response_text = image_response.strip()
    
    try:
        # Check for different response formats
        if response_text.startswith("data:image/"):
            # Base64 encoded image
            try:
                header, b64data = response_text.split(",", 1)
                mime = header.split(";")[0].split(":")[-1]
                ext_guess = mime.split("/")[-1] if "/" in mime else "png"
                ext = f".{ext_guess}" if not ext_guess.startswith(".") else ext_guess
                filename = f"{safe_a}__{safe_b}__{uuid.uuid4().hex[:8]}{ext}"
                saved_path = out_images / filename
                saved_path.write_bytes(base64.b64decode(b64data))
            except Exception as e:
                print(f"❌ Error parsing base64 image: {e}")
                raise
                
        elif response_text.startswith("http"):
            # URL to image
            try:
                resp = requests.get(response_text, timeout=30)
                resp.raise_for_status()
                ext = Path(urlparse(response_text).path).suffix or ".png"
                filename = f"{safe_a}__{safe_b}__{uuid.uuid4().hex[:8]}{ext}"
                saved_path = out_images / filename
                saved_path.write_bytes(resp.content)
            except Exception as e:
                print(f"❌ Error downloading image from URL: {e}")
                raise
                
        elif Path(response_text).exists():
            # Local file path
            try:
                src = Path(response_text)
                ext = src.suffix if src.suffix else ".png"
                filename = f"{safe_a}__{safe_b}__{uuid.uuid4().hex[:8]}{ext}"
                saved_path = out_images / filename
                shutil.copy(src, saved_path)
            except Exception as e:
                print(f"❌ Error copying local file: {e}")
                raise
                
        else:
            # Fallback: treat as error or create placeholder
            print(f"⚠️ Unrecognized response format: {response_text[:100]}...")
            print("Creating placeholder file. Check your MCP tool's response format.")
            saved_path.write_bytes(b"placeholder_image_data_from_dedalus_mcp")
            
    except Exception as e:
        print(f"❌ Error parsing image response: {e}")
        # Create error placeholder
        saved_path.write_bytes(b"error_parsing_image_response")
    
    # Save prompt text alongside
    prompt_filename = saved_path.stem + ".txt"
    (out_prompts / prompt_filename).write_text(prompt_text, encoding="utf-8")
    
    return str(saved_path)


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
    builder.add_node("dedalus_prompt_and_generate", dedalus_prompt_and_generate)  # type: ignore[arg-type]
    builder.add_node("finalize", finalize)

    builder.add_edge(START, "init_run_output")
    builder.add_edge("init_run_output", "propose_dimensions")
    builder.add_edge("propose_dimensions", "save_run_config")
    builder.add_edge("save_run_config", "start_variation_fanout")
    builder.add_conditional_edges("start_variation_fanout", split_to_variations, ["dedalus_prompt_and_generate"])
    builder.add_edge("dedalus_prompt_and_generate", "finalize")
    builder.add_edge("finalize", END)

    return builder.compile()
