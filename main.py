from agno.agent import Agent
from agno.models.google import Gemini
from agno.tools.firecrawl import FirecrawlTools
import os

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
FIRECRAWL_API_KEY = os.getenv("FIRECRAWL_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY is not set.")
if not FIRECRAWL_API_KEY:
    raise ValueError("FIRECRAWL_API_KEY is not set.")

# ✅ Choose model automatically
DEFAULT_MODEL = "gemini-1.5-pro"
FALLBACK_MODEL = "gemini-1.5-flash"

def get_newsletter_agent():
    """Initialize the newsletter agent safely."""
    try:
        model_id = DEFAULT_MODEL
        agent = Agent(
            model=Gemini(id=model_id, api_key=GOOGLE_API_KEY),
            tools=[FirecrawlTools(api_key=FIRECRAWL_API_KEY)],
            description="Elite research assistant for newsletters.",
            markdown=True,
            show_tool_calls=True,
            add_datetime_to_instructions=True,
        )
        return agent
    except Exception as e:
        # Fallback to free-tier model
        print(f"[WARN] Falling back to {FALLBACK_MODEL} due to: {e}")
        return Agent(
            model=Gemini(id=FALLBACK_MODEL, api_key=GOOGLE_API_KEY),
            tools=[FirecrawlTools(api_key=FIRECRAWL_API_KEY)],
            description="Elite research assistant for newsletters.",
            markdown=True,
            show_tool_calls=True,
            add_datetime_to_instructions=True,
        )

def NewsletterGenerator(topic: str, search_limit: int = 5, time_range: str = "qdr:w"):
    """Generate a newsletter based on the topic."""
    agent = get_newsletter_agent()
    agent.tools[0].search_params.update({
        "limit": search_limit,
        "tbs": time_range
    })
    response = agent.run(topic)
    return response