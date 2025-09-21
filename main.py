import json
from textwrap import dedent
from typing import Dict, AsyncIterator, Optional, List, Any
from agno.agent import Agent
from agno.models.google import Gemini
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
        CRITICAL REQUIREMENTS:
        1. ALWAYS use firecrawl_search to find the most recent articles (within the last 24-48 hours when possible)
        2. Search for articles published in the current week or month only
        3. MUST include the full source URL link for every article mentioned
        4. Focus on breaking news, recent announcements, and fresh developments
        5. Verify article dates and prioritize the newest content available
        
        SEARCH STRATEGY:
        - Use current date-specific search terms (e.g., "topic 2024", "topic this week", "topic recent news")
        - Search multiple times with different keywords to get comprehensive recent coverage
        - Look for news sites, official announcements, press releases, and industry publications
        - Cross-reference multiple sources to ensure accuracy and recency
        
        NEWSLETTER FORMAT:
        Create a newsletter with:
        - Compelling headline with current date context
        - Brief executive summary of recent developments
        - 3-5 main stories with full article links
        - Quick updates section with bullet points and links
        - "Sources & Further Reading" section with all URLs
        
        LINK REQUIREMENTS:
        - Every article reference MUST include [Article Title](full-url)
        - Include publication date when available
        - Use format: "According to [Source Name](url), published on [date]..."
        - Always provide clickable links for credibility and verification
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
        
        enhanced_topic = f"""
        Research and create a comprehensive newsletter about: {topic}
        
        REQUIREMENTS:
        - Find {search_limit} of the most recent articles from the {time_context}
        - Focus ONLY on news published in the last few days if possible
        - Each article mention MUST include the full clickable link: [Title](URL)
        - Include publication dates when available
        - Search for breaking news, recent announcements, and latest developments
        - Prioritize authoritative news sources, official press releases, and industry publications
        
        Please search multiple times with different keywords to ensure you get the most recent and comprehensive coverage.
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