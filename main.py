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
        model=Gemini(id="gemini-1.5-flash", api_key=GOOGLE_API_KEY)
    )
    print("Agent created successfully!")
except Exception as e:
    print(f"Error creating agent: {e}")
    # Fallback: create a simple function-based approach
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
        
        # Include search parameters in the topic query since we can't modify FirecrawlTools parameters directly
        enhanced_topic = f"{topic}. Please search for recent articles (limit: {search_limit}, time range: {time_range}) and create a comprehensive newsletter."
        
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