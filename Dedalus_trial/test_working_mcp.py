"""
Test with a known working MCP server to verify structured output works.
"""

import asyncio
import json
from dedalus_labs import AsyncDedalus, DedalusRunner
from dotenv import load_dotenv

load_dotenv()


async def test_working_mcp():
    """Test with AWS Documentation MCP (known to work)"""
    
    client = AsyncDedalus()
    runner = DedalusRunner(client)
    
    prompt = """
    Use the AWS Documentation MCP to search for information about S3 bucket naming rules.
    
    Return ONLY this JSON (no other text):
    {
        "search_query": "string",
        "results_found": number,
        "summary": "string",
        "success": true/false
    }
    """
    
    try:
        response = await runner.run(
            input=prompt,
            model="openai/gpt-4o",
            mcp_servers=["awslabs.aws-documentation-mcp-server"],
            stream=False
        )
        
        print("✅ AWS MCP response received!")
        print("Raw response:")
        print(response.final_output)
        
        # Try to parse JSON
        response_text = response.final_output.strip()
        if response_text.startswith("{"):
            json_text = response_text
        else:
            start = response_text.find("{")
            end = response_text.rfind("}") + 1
            json_text = response_text[start:end]
        
        data = json.loads(json_text)
        
        print("\n✅ Structured output:")
        print(f"Search Query: {data['search_query']}")
        print(f"Results Found: {data['results_found']}")
        print(f"Summary: {data['summary']}")
        print(f"Success: {data['success']}")
        
    except Exception as e:
        print(f"❌ AWS MCP Error: {e}")


if __name__ == "__main__":
    asyncio.run(test_working_mcp())
