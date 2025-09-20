"""
Simple Dedalus agent for image generation with structured output.
"""

import asyncio
import json
from dedalus_labs import AsyncDedalus, DedalusRunner
from dotenv import load_dotenv

load_dotenv()


async def simple_image_agent():
    """Simple agent that generates an image and returns structured output"""
    
    print("🔧 Initializing Dedalus client...")
    client = AsyncDedalus()
    runner = DedalusRunner(client)
    
    # First, let's test without MCP to see if basic Dedalus works
    print("🧪 Testing basic Dedalus functionality...")
    try:
        basic_response = await runner.run(
            input="Say hello and return this JSON: {\"message\": \"hello\", \"status\": \"success\"}",
            model="openai/gpt-4o",
            mcp_servers=[],
            stream=False
        )
        print("✅ Basic Dedalus works!")
        print(f"Basic response: {basic_response.final_output}")
    except Exception as e:
        print(f"❌ Basic Dedalus failed: {e}")
        return
    
    # Now test with MCP
    print("\n🧪 Testing with Flux MCP...")
    prompt = """
    Generate an image using the Flux MCP server with this prompt: "A beautiful sunset over mountains"
    
    Return ONLY this JSON (no other text):
    {
        "prompt_used": "string",
        "image_data": "string (base64, URL, or file path)",
        "generation_success": true/false
    }
    """
    
    try:
        response = await runner.run(
            input=prompt,
            model="openai/gpt-4o",
            mcp_servers=["yihu/flux-mcp"],
            stream=False
        )
        
        print("✅ MCP response received!")
        print("Raw response:")
        print(response.final_output)
        
        # Parse JSON
        response_text = response.final_output.strip()
        if response_text.startswith("{"):
            json_text = response_text
        else:
            start = response_text.find("{")
            end = response_text.rfind("}") + 1
            json_text = response_text[start:end]
        
        data = json.loads(json_text)
        
        print("\n✅ Structured output:")
        print(f"Prompt: {data['prompt_used']}")
        print(f"Success: {data['generation_success']}")
        print(f"Image: {data['image_data'][:50]}...")
        
    except Exception as e:
        print(f"❌ MCP Error: {e}")
        print("This might be an issue with the Flux MCP server or your API keys")


if __name__ == "__main__":
    asyncio.run(simple_image_agent())