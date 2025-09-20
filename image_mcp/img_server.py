# image_mcp/server.py
from fastmcp import FastMCP
from fastmcp.utilities.types import Image
from mcp.types import ImageContent
from dotenv import load_dotenv
from .flux_adapter import FluxAdapter, DiffusionPrompt
import os, io, requests

load_dotenv()
mcp = FastMCP("ImageEditor")

@mcp.tool()
async def flux_generate(prompt: str,
                        model: str = "flux-pro-1.1",
                        aspect_ratio: str | None = "16:9",
                        width: int = 1024,
                        height: int = 1024,
                        raw: bool = False,
                        safety_tolerance: int = 6,
                        prompt_upsampling: bool = False) -> ImageContent:
    api_key = os.getenv("BFL_API_KEY")
    if not api_key:
        # Return a short error string; FastMCP will surface this cleanly.
        return {"type": "text", "text": "BFL_API_KEY not set"}  # fallback

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

    # Generate → returns a URL (or data URL) today
    image_url, meta = await adapter.generate(DiffusionPrompt(text=prompt))

    # If data URL, decode; if http(s), download; otherwise assume local file.
    img_bytes: bytes
    fmt = "png"

    if isinstance(image_url, str) and image_url.startswith("data:"):
        header, b64data = image_url.split(",", 1)
        if "image/" in header:
            fmt = header.split(";")[0].split("/")[-1]
        import base64
        img_bytes = base64.b64decode(b64data)
    elif image_url.startswith(("http://", "https://")):
        r = requests.get(image_url, timeout=30)
        r.raise_for_status()
        img_bytes = r.content
        ct = r.headers.get("content-type", "")
        if "image/" in ct:
            fmt = ct.split("/")[-1].split(";")[0] or "png"
    else:
        with open(image_url, "rb") as f:
            img_bytes = f.read()
        ext = os.path.splitext(image_url)[1].lstrip(".").lower()
        if ext:
            fmt = ext

    # Wrap as FastMCP Image → ImageContent
    return Image(data=img_bytes, format=fmt).to_image_content()

if __name__ == "__main__":
    mcp.run()
