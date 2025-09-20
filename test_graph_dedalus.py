import os
import asyncio
from pathlib import Path

from dotenv import load_dotenv, find_dotenv

from langgraph_workflow.product_images_graph_dedalus import ProductImagesState, compile_graph


load_dotenv(find_dotenv())


async def main() -> None:
    # Optional reference image (not required by the dedalus node; kept for parity)
    # Prefer HTTP URL if provided; otherwise allow a local path fallback
    default_url = "https://images.unsplash.com/photo-1567169866456-a0759b6bb0c8?fm=jpg&q=60&w=3000&ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxzZWFyY2h8M3x8cGx1c2h8ZW58MHx8MHx8fDA%3D"
    initial_image_path = os.getenv("TEST_REFERENCE_IMAGE_URL", default_url)
    print(f"Using image URL for proposal step: {initial_image_path}")

    # Model used by the graph for dimension proposal and the Dedalus call
    model_name = os.getenv("TEST_OPENAI_MODEL", "gpt-4o")

    # Image model name is forwarded in the prompt to the Flux MCP tool
    image_model = os.getenv("TEST_FLUX_MODEL", "flux-kontext-max")
    use_raw_mode = False

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

    # Keys required:
    # - OPENAI_API_KEY (for dimension proposal and model routing)
    # - DEDALUS_API_KEY (for Dedalus client)
    missing = [k for k in ("OPENAI_API_KEY", "DEDALUS_API_KEY") if not os.getenv(k)]
    if missing:
        print(f"Warning: missing environment variables: {', '.join(missing)}")

    print("Compiling Dedalus product images graph...")
    graph = compile_graph()

    print("Invoking graph (this will call OpenAI and Flux MCP via Dedalus if keys are set)...")
    try:
        final_state = await graph.ainvoke(state)
    except Exception as e:
        print("Graph invocation failed. Common causes: missing OPENAI_API_KEY, DEDALUS_API_KEY, or MCP access.")
        print(f"Error: {e}")
        return

    results = final_state.get("results", [])
    print(f"Generated {len(results)} results.")
    for i, r in enumerate(results, start=1):
        print(f"[{i}] {r['dim_a_value']} x {r['dim_b_value']} -> {r['img_path']}")

    # Debug flags from dimension proposal
    saw = final_state.get("saw_image_successfully")
    desc = final_state.get("image_description")
    if saw is not None:
        print(f"saw_image_successfully={saw}")
    if desc:
        print(f"image_description={desc}")


if __name__ == "__main__":
    asyncio.run(main())


