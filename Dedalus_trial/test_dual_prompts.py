"""
Test if Dedalus can handle system + human prompts combined.
"""

import asyncio
from dedalus_labs import AsyncDedalus, DedalusRunner
from dotenv import load_dotenv

load_dotenv()


async def test_dual_prompts():
    """Test if Dedalus can handle system + human prompts"""
    
    client = AsyncDedalus()
    runner = DedalusRunner(client)
    
    # Test 1: Simple combined prompt
    print("🧪 Test 1: Simple combined prompt")
    simple_prompt = """
    You are a helpful assistant that writes product descriptions.
    
    Product: Wireless headphones
    Style: Modern
    Color: Blue
    
    Write a short product description.
    """
    
    try:
        response = await runner.run(
            input=simple_prompt,
            model="openai/gpt-4o-mini",
            mcp_servers=[],
            stream=False
        )
        print("✅ Simple combined prompt works!")
        print(f"Response: {response.final_output[:100]}...")
    except Exception as e:
        print(f"❌ Simple combined prompt failed: {e}")
    
    # Test 2: Complex system + human prompt (like our use case)
    print("\n🧪 Test 2: Complex system + human prompt")
    complex_prompt = """
    You are a prompt-writer. Your task is to generate image generation prompts in a consistent, clean format.

    Inputs You Will Receive:
    1. Reference image - the uploaded product image that must always be kept accurate and central.
    2. Dimension values - stylistic or semantic controls such as tone: calm, palette: pastel, mood: energetic, etc.

    Goals:
    - Come up with interesting and creative prompt ideas for this type of advert spread.
    - Keep the ideas fresh and varied, but not over the top or cluttered.
    - Ensure the prompt could realistically guide the creation of a professional, polished ad image.

    Output You Must Produce:
    Write a single prompt in the following format:
    - Start with: "Use the uploaded [object/product] image as the product reference."
    - Follow with Scene: describe the setting, background, and overall environment.
    - Follow with Composition: specify placement of the product, how it is framed, any camera angle.
    - Follow with Lighting/Color: describe how light interacts with the object and the palette.
    - End with Output format: specify aspect ratio like 4:5, 9:16, etc.

    Product description:
    A premium wireless headphone with sleek design

    Target variation:
    - Style: Modern
    - Color: Blue

    Write one final diffusion prompt, incorporating these target variations.
    """
    
    try:
        response = await runner.run(
            input=complex_prompt,
            model="openai/gpt-4o-mini",
            mcp_servers=[],
            stream=False
        )
        print("✅ Complex system + human prompt works!")
        print(f"Response: {response.final_output[:200]}...")
    except Exception as e:
        print(f"❌ Complex system + human prompt failed: {e}")
    
    # Test 3: Very long prompt (like our full PROMPT_SYSTEM)
    print("\n🧪 Test 3: Very long prompt")
    long_prompt = """
    You are a prompt-writer. Your task is to generate image generation prompts in a consistent, clean format.

    Inputs You Will Receive:
    1. Reference image - the uploaded product image that must always be kept accurate and central.
    2. Dimension values - stylistic or semantic controls such as tone: calm, palette: pastel, mood: energetic, etc.

    Goals:
    - Come up with interesting and creative prompt ideas for this type of advert spread.
    - Keep the ideas fresh and varied, but not over the top or cluttered.
    - Ensure the prompt could realistically guide the creation of a professional, polished ad image.

    Output You Must Produce:
    Write a single prompt in the following format:
    - Start with: "Use the uploaded [object/product] image as the product reference."
    - Follow with Scene: describe the setting, background, and overall environment.
    - Follow with Composition: specify placement of the product, how it is framed, any camera angle.
    - Follow with Lighting/Color: describe how light interacts with the object and the palette.
    - End with Output format: specify aspect ratio like 4:5, 9:16, etc.

    Style Rules:
    - Always keep the reference image/object crisp, sharp, and unaltered.
    - Avoid exaggeration, clutter, or unrelated props.
    - Write in concise, technical description style, not in marketing voice.
    - Ensure the dimension values are reflected clearly in the scene, composition, or palette.
    - Do not use emojis, slang, or promotional filler.

    Product description:
    A premium wireless headphone with sleek design

    Target variation:
    - Style: Modern
    - Color: Blue

    Write one final diffusion prompt, incorporating these target variations.
    """
    
    try:
        response = await runner.run(
            input=long_prompt,
            model="openai/gpt-4o-mini",
            mcp_servers=[],
            stream=False
        )
        print("✅ Very long prompt works!")
        print(f"Response: {response.final_output[:200]}...")
    except Exception as e:
        print(f"❌ Very long prompt failed: {e}")


if __name__ == "__main__":
    asyncio.run(test_dual_prompts())
