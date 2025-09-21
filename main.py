import os
from dotenv import load_dotenv
from agno.agent import Agent
from agno.models.google import Gemini
from agno.tools.firecrawl import FirecrawlTools
from agno.utils.log import logger
from typing import Dict, Any

# Load environment variables
load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
FIRECRAWL_API_KEY = os.getenv("FIRECRAWL_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY is not set.")
if not FIRECRAWL_API_KEY:
    raise ValueError("FIRECRAWL_API_KEY is not set.")

# Ensure tmp directory exists
os.makedirs("tmp", exist_ok=True)

# Default and fallback models
DEFAULT_MODEL = "gemini-1.5-pro"
FALLBACK_MODEL = "gemini-1.5-flash"

def get_newsletter_agent(model_id: str = DEFAULT_MODEL) -> Agent:
    """Initialize a newsletter research agent (no search_params dependency)."""
    return Agent(
        model=Gemini(id=model_id, api_key=GOOGLE_API_KEY),
        tools=[FirecrawlTools(api_key=FIRECRAWL_API_KEY)],
        description="You are NewsletterResearch-X, an assistant for discovering and extracting content for newsletters.",
        markdown=True,
    )

def NewsletterGenerator(topic: str, search_limit: int = 5, time_range: str = "qdr:w") -> Dict[str, Any]:
    """Generate a newsletter based on topic and search parameters."""
    try:
        # Try pro model first, fallback to flash
        try:
            agent = get_newsletter_agent(DEFAULT_MODEL)
        except Exception as e:
            logger.warning(f"Falling back to {FALLBACK_MODEL} due to: {e}")
            agent = get_newsletter_agent(FALLBACK_MODEL)

        # 🚨 Removed `.search_params.update()` since it's None in your version
        # Instead, pass context into the agent run call
        query = f"{topic} (limit={search_limit}, time_range={time_range})"

        response = agent.run(query)
        logger.info("Newsletter generated successfully")
        return response
    except Exception as e:
        logger.error("Unexpected error in newsletter generation: %s", e, exc_info=True)
        raise RuntimeError(f"Newsletter generation failed: {e}") from e

if __name__ == "__main__":
    print(NewsletterGenerator("Latest developments in AI"))