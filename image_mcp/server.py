from mcp.server.fastmcp import FastMCP
from dotenv import load_dotenv
from .flux_adapter import FluxAdapter, DiffusionPrompt
import os
from typing import Optional

# Load environment variables from .env file
load_dotenv()

# Create an MCP server
mcp = FastMCP("ImageEditor")


@mcp.tool()
async def flux_generate(prompt: str,
                        model: str = "flux-pro-1.1",
                        aspect_ratio: str | None = "16:9",
                        width: int = 1024,
                        height: int = 1024,
                        raw: bool = False,
                        safety_tolerance: int = 6,
                        prompt_upsampling: bool = False) -> dict:
    api_key = os.getenv("BFL_API_KEY")
    if not api_key:
        return {"status": "error", "message": "BFL_API_KEY not set"}
    try:
        adapter = FluxAdapter(
            model=model,
            use_raw_mode=raw,
            api_key=api_key,
            aspect_ratio=aspect_ratio,
            width=width,
            height=height,
            safety_tolerance=safety_tolerance,
            prompt_upsampling=prompt_upsampling,
        )
        image_url, meta = await adapter.generate(DiffusionPrompt(text=prompt))
        return {"status": "success", "image": image_url, "meta": meta}
    except Exception as e:
        return {"status": "error", "message": str(e)}
    

if __name__ == "__main__":
    mcp.run()