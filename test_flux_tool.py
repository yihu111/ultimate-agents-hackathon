# test_flux_tool.py
import asyncio, base64, json
from image_mcp.img_server import flux_generate  # <- use img_server

async def main():
    res = await flux_generate("a photo of a red apple on a wooden table")

    # Handle non-image responses (e.g., missing BFL_API_KEY)
    if not isinstance(res, dict) or res.get("type") != "image":
        print("Non-image response:\n", json.dumps(res, indent=2, default=str))
        return

    mime = res.get("mimeType", "image/png")
    data = res.get("data")
    if isinstance(data, str):
        raw = base64.b64decode(data)
    elif isinstance(data, (bytes, bytearray)):
        raw = data
    else:
        raise RuntimeError(f"Unexpected image data type: {type(data)}")

    ext = mime.split("/")[-1] if "/" in mime else "png"
    out = f"out.{ext}"
    with open(out, "wb") as f:
        f.write(raw)
    print(f"Saved {out} ({mime}, {len(raw)} bytes)")

if __name__ == "__main__":
    asyncio.run(main())