import asyncio
from dedalus_labs import AsyncDedalus, DedalusRunner
from dotenv import load_dotenv
from dedalus_labs.utils.streaming import stream_async

load_dotenv()

async def main():
    # Initialize Dedalus client
    client = AsyncDedalus()
    
    # Create a runner - MCP servers are configured through the client or environment
    runner = DedalusRunner(client)

    # Example 1: Search for events in a specific city
    print("=== Searching for events in New York ===")
    response1 = await runner.run(
        input="Find me some upcoming concerts or events in New York City this weekend",
        model="openai/gpt-4o-mini",
        mcp_servers=["windsor/ticketmaster-mcp"],
        stream=False
    )
    print("Response:", response1.final_output)
    print("\n" + "="*50 + "\n")

    # Example 2: Search for specific artist events
    print("=== Searching for Taylor Swift events ===")
    response2 = await runner.run(
        input="Are there any Taylor Swift concerts coming up? Show me dates, venues, and ticket availability",
        model="openai/gpt-4o-mini",
        mcp_servers=["windsor/ticketmaster-mcp"],
        stream=False
    )
    print("Response:", response2.final_output)
    print("\n" + "="*50 + "\n")

    # Example 3: Search for sports events
    print("=== Searching for sports events ===")
    response3 = await runner.run(
        input="Find me upcoming NBA games or major sports events in Los Angeles",
        model="openai/gpt-4o-mini",
        mcp_servers=["windsor/ticketmaster-mcp"],
        stream=False
    )
    print("Response:", response3.final_output)
    print("\n" + "="*50 + "\n")

    # Example 4: Interactive event discovery
    print("=== Interactive Event Discovery ===")
    response4 = await runner.run(
        input="I'm looking for family-friendly events in Chicago next month. What options do I have?",
        model="openai/gpt-4o-mini",
        mcp_servers=["windsor/ticketmaster-mcp"],
        stream=False
    )
    print("Response:", response4.final_output)

async def interactive_ticketmaster_agent():
    """Interactive agent for ongoing Ticketmaster queries"""
    client = AsyncDedalus()
    runner = DedalusRunner(client)
    
    print("🎫 Welcome to the Ticketmaster Event Discovery Agent!")
    print("Ask me about events, concerts, sports games, or any live entertainment!")
    print("Type 'quit' to exit.\n")
    
    while True:
        try:
            user_input = input("🎤 What events are you looking for? ")
            
            if user_input.lower() in ['quit', 'exit', 'bye']:
                print("👋 Thanks for using the Ticketmaster Agent!")
                break
                
            if not user_input.strip():
                continue
                
            print("🔍 Searching for events...")
            response = await runner.run(
                input=user_input,
                model="openai/gpt-4o-mini",
                mcp_servers=["windsor/ticketmaster-mcp"],
                stream=False
            )
            
            print(f"🎭 {response.final_output}\n")
            
        except KeyboardInterrupt:
            print("\n👋 Thanks for using the Ticketmaster Agent!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")
            print("Please try again.\n")

if __name__ == "__main__":
    print("Choose an option:")
    print("1. Run example queries")
    print("2. Interactive mode")
    
    choice = input("Enter choice (1 or 2): ").strip()
    
    if choice == "2":
        asyncio.run(interactive_ticketmaster_agent())
    else:
        asyncio.run(main())