import os
import asyncio
from pathlib import Path
from typing import Dict, Any

from dotenv import load_dotenv, find_dotenv

from langgraph_workflow.product_images_graph import ProductImagesState, compile_graph

load_dotenv(find_dotenv())


async def main() -> None:
    # Check optional reference image path (not used by graph yet, but we log it)
    candidate_img = Path("/Users/william/Downloads/plushie.jpeg")
    if candidate_img.is_file():
        print(f"Found reference image: {candidate_img}")
        initial_image_path = str(candidate_img)
    else:
        print("Reference image not found at /Users/william/Downloads/plushie.jpeg. Proceeding without it.")
        initial_image_path = None

    # Minimal local model spec; ensure your OPENAI_API_KEY is set and model exists
    model_name = os.getenv("TEST_OPENAI_MODEL", "gpt-4o")

    # Image model for FluxAdapter; ensure BFL_API_KEY is set in env
    image_model = os.getenv("TEST_FLUX_MODEL", "flux-kontext-max")
    use_raw_mode = False

    # Build initial state
    state: ProductImagesState = {
        "model_spec": {"name": model_name},
        "product_description": (
            "A soft plush toy product suitable for infants, featuring friendly animal shapes."
        ),
        "extra_guidance": "Target audience is mothers with babies.",
        "num_dimensions": 2,
        "num_values_per_dim": 3,
        "image_model": image_model,
        "use_raw_mode": use_raw_mode,
        "results": [],
    }
    if initial_image_path:
        state["initial_image_path"] = initial_image_path

    print("Compiling product images graph...")
    graph = compile_graph()

    print("Invoking graph (this will call LLM and Flux if keys are set)...")
    try:
        final_state = await graph.ainvoke(state)
    except Exception as e:
        print("Graph invocation failed. Common causes: missing OPENAI_API_KEY or BFL_API_KEY.")
        print(f"Error: {e}")
        return

    results = final_state.get("results", [])
    print(f"Generated {len(results)} results.")
    for i, r in enumerate(results, start=1):
        print(f"[{i}] {r['dim_a_value']} x {r['dim_b_value']} -> {r['img_path']}")


if __name__ == "__main__":
    asyncio.run(main())


