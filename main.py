import json
from textwrap import dedent
from typing import Dict, AsyncIterator, Optional, List, Any
from agno.agent import Agent
from agno.models.google import Gemini
from datetime import datetime, timedelta  # Added this import
# from agno.storage.sqlite import SqliteStorage  # Storage not available in this version
from agno.utils.log import logger
import os
from agno.utils.pprint import pprint_run_response
from dotenv import load_dotenv
import asyncio
from agno.tools.firecrawl import FirecrawlTools


# Load environment variables
load_dotenv()

# Get API keys from environment variables
FIRECRAWL_API_KEY = os.getenv("FIRECRAWL_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Validate API keys
if not FIRECRAWL_API_KEY:
    raise ValueError("FIRECRAWL_API_KEY environment variable is not set. Please set it in your .env file or environment.")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY environment variable is not set. Please set it in your .env file or environment.")

# Add this line to create the 'tmp' directory if it doesn't exist
os.makedirs("tmp", exist_ok=True)

# Newsletter Research Agent: Handles web searching and content extraction using Firecrawl
try:
    newsletter_agent = Agent(
        model=Gemini(id="gemini-1.5-flash", api_key=GOOGLE_API_KEY),
        tools=[FirecrawlTools(api_key=FIRECRAWL_API_KEY)],
        description="You are NewsletterResearch-X, an elite research assistant specializing in discovering and extracting high-quality, recent content for compelling newsletters.",
        instructions=dedent("""\
        You are an elite newsletter research assistant. Your primary job is to find the most recent articles about given topics.
        
        CRITICAL: You MUST use the firecrawl_search function with specific time parameters to get recent articles.
        
        SEARCH INSTRUCTIONS:
        1. Use firecrawl_search with these parameters:
           - query: include the topic + current date terms
           - tbs: "qdr:d" for past day, "qdr:w" for past week, etc.
           - limit: number of articles to find
        
        2. ALWAYS search with time constraints:
           - For breaking news: tbs="qdr:h" (past hour)
           - For daily news: tbs="qdr:d" (past 24 hours)  
           - For weekly roundups: tbs="qdr:w" (past week)
        
        3. MULTIPLE SEARCHES: Perform several searches with different keywords:
           - "{topic} breaking news"
           - "{topic} latest news 2024" 
           - "{topic} recent developments"
           - "{topic} today news"
        
        4. LINK FORMAT: Every article MUST include [Article Title](full-url)
        
        5. DATE VERIFICATION: Only include articles that show recent publication dates
        
        Example search calls you should make:
        - firecrawl_search(query="AI news breaking", tbs="qdr:d", limit=5)
        - firecrawl_search(query="artificial intelligence latest 2024", tbs="qdr:w", limit=5)
        
        If you don't find recent articles, explicitly state the search timeframe used and results found.
        """),
        markdown=True,
        show_tool_calls=True,
        add_datetime_to_instructions=True
    )
    print("Agent with FirecrawlTools created successfully!")
except Exception as e:
    print(f"Error creating agent with tools: {e}")
    try:
        # Fallback: create agent without tools
        newsletter_agent = Agent(
            model=Gemini(id="gemini-1.5-flash", api_key=GOOGLE_API_KEY),
            description="Newsletter research assistant"
        )
        print("Basic agent created successfully!")
    except Exception as e2:
        print(f"Error creating basic agent: {e2}")
        newsletter_agent = None

def NewsletterGenerator(topic: str, search_limit: int = 5, time_range: str = "qdr:w") -> Dict[str, Any]:
    """
    Generate a newsletter based on the given topic and search parameters.
    
    Args:
        topic (str): The topic to generate the newsletter about
        search_limit (int): Maximum number of articles to search and analyze
        time_range (str): Time range for article search (e.g., "qdr:w" for past week)
    
    Returns:
        Dict[str, Any]: Processed newsletter content with structured metadata
    
    Raises:
        ValueError: If configuration validation fails
        RuntimeError: If newsletter generation fails
    """
    try:
        if newsletter_agent is None:
            # Fallback response when agent creation fails
            return {
                "content": f"# Newsletter Generator Error\n\nUnable to initialize the newsletter agent. Please check your API keys and try again.\n\nRequested topic: {topic}",
                "messages": []
            }
        
        # Create a detailed prompt that emphasizes recent content and source links
        time_context = {
            "qdr:h": "past hour",
            "qdr:d": "past 24 hours", 
            "qdr:w": "past week",
            "qdr:m": "past month",
            "qdr:y": "past year"
        }.get(time_range, "recent")
        
        # Create a more specific and urgent prompt for recent content
        from datetime import datetime, timedelta
        
        # Get current date for context
        today = datetime.now()
        yesterday = today - timedelta(days=1)
        week_ago = today - timedelta(days=7)
        
        time_context = {
            "qdr:h": f"past hour (since {(today - timedelta(hours=1)).strftime('%Y-%m-%d %H:00')})",
            "qdr:d": f"past 24 hours (since {yesterday.strftime('%Y-%m-%d')})", 
            "qdr:w": f"past week (since {week_ago.strftime('%Y-%m-%d')})",
            "qdr:m": f"past month (since {(today - timedelta(days=30)).strftime('%Y-%m-%d')})",
            "qdr:y": f"past year (since {(today - timedelta(days=365)).strftime('%Y-%m-%d')})"
        }.get(time_range, "recent")
        
        enhanced_topic = f"""
        SEARCH TASK: Find recent articles about {topic}
        
        You MUST use the firecrawl_search function with these exact parameters:
        
        1. FIRST SEARCH (Breaking news):
           firecrawl_search(
               query="{topic} breaking news",
               tbs="{time_range}",
               limit={search_limit},
               formats=["markdown", "links"]
           )
        
        2. SECOND SEARCH (Latest developments):
           firecrawl_search(
               query="{topic} latest news 2024",
               tbs="{time_range}",
               limit={search_limit},
               formats=["markdown", "links"]
           )
        
        3. THIRD SEARCH (Recent announcements):
           firecrawl_search(
               query="{topic} recent developments",
               tbs="{time_range}",
               limit={search_limit},
               formats=["markdown", "links"]
           )
        
        TIME FILTER: {time_range} = {time_context}
        TODAY'S DATE: {today.strftime('%Y-%m-%d')}
        
        REQUIREMENTS:
        - Only include articles from the specified timeframe: {time_context}
        - Each article MUST include: [Article Title](full-url)
        - Include publication date when available
        - If no recent articles found, explicitly state the search timeframe used
        
        Create a newsletter with the search results, ensuring all articles are from {time_context}.
        """
        
        response = newsletter_agent.run(enhanced_topic)
        logger.info('Newsletter generated successfully')
        return response
    except ValueError as ve:
        logger.error('Configuration error: %s', ve)
        raise
    except Exception as e:
        logger.error('Unexpected error in newsletter generation: %s', e, exc_info=True)
        raise RuntimeError('Newsletter generation failed: %s' % e) from e

if __name__ == "__main__":
    NewsletterGenerator("Latest developments in AI")