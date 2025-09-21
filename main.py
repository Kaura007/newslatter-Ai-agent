import os
from textwrap import dedent
from typing import Dict, Any
from dotenv import load_dotenv
from agno.agent import Agent
from agno.models.google import Gemini
from agno.tools.firecrawl import FirecrawlTools
from agno.utils.log import logger

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
    """Safely initialize a newsletter research agent."""
    return Agent(
        model=Gemini(id=model_id, api_key=GOOGLE_API_KEY),
        tools=[FirecrawlTools(api_key=FIRECRAWL_API_KEY)],
        description=dedent("""\
            You are NewsletterResearch-X, an elite research assistant specializing in discovering
            and extracting high-quality content for compelling newsletters. Your expertise includes:

            - Finding authoritative and trending sources across multiple domains
            - Extracting and synthesizing content efficiently while maintaining accuracy
            - Evaluating content credibility, relevance, and potential impact
            - Identifying diverse perspectives, expert opinions, and emerging trends
            - Ensuring comprehensive topic coverage with balanced viewpoints
            - Maintaining journalistic integrity and ethical reporting standards
            - Creating engaging narratives that resonate with target audiences
            - Adapting content style and depth based on audience expertise level
        """),
        instructions=dedent("""\
            1. Initial Research & Discovery:
               - Use firecrawl_search to find recent articles about the topic
               - Search for authoritative sources, expert opinions, and industry leaders
               - Focus on the most recent and relevant content (last 7 days preferred)
               - Identify key stakeholders and their perspectives
               - Look for contrasting viewpoints to ensure balanced coverage

            2. Content Analysis & Processing:
               - Extract key insights, trends, and statistics from each article
               - Evaluate source credibility and potential biases
               - Identify gaps in coverage needing further research
               - Connect insights across sources for deeper analysis

            3. Content Organization & Structure:
               - Group related information by theme and significance
               - Identify main story angles and supporting narratives
               - Ensure balanced coverage of different perspectives
               - Structure content for reader engagement

            4. Newsletter Creation:
               - Follow the exact template structure below
               - Write clear, concise, and engaging sections
               - Use markdown formatting
               - Include actionable insights and citations
        """),
        expected_output=dedent("""\
            # ${Compelling Subject Line}

            ## Welcome
            {Engaging hook and context}

            ## ${Main Story}
            {Key insights and analysis}
            {Expert quotes and statistics}

            ## Featured Content
            {Deeper exploration}
            {Real-world examples}

            ## Quick Updates
            {Actionable insights}
            {Expert recommendations}

            ## This Week's Highlights
            - {Notable update 1}
            - {Important news 2}
            - {Key development 3}

            ## Sources & Further Reading
            {Properly attributed sources with links}
        """),
        markdown=True,
        add_datetime_to_instructions=True,
    )

def NewsletterGenerator(topic: str, search_limit: int = 5, time_range: str = "qdr:w") -> Dict[str, Any]:
    """
    Generate a newsletter based on the given topic and search parameters.
    """
    try:
        # Try with pro model first, then fallback to flash
        try:
            agent = get_newsletter_agent(DEFAULT_MODEL)
        except Exception as e:
            logger.warning(f"Falling back to {FALLBACK_MODEL} due to: {e}")
            agent = get_newsletter_agent(FALLBACK_MODEL)

        # Update Firecrawl search params
        agent.tools[0].search_params.update({
            "limit": search_limit,
            "tbs": time_range
        })

        response = agent.run(topic)
        logger.info("Newsletter generated successfully")
        return response
    except Exception as e:
        logger.error("Unexpected error in newsletter generation: %s", e, exc_info=True)
        raise RuntimeError(f"Newsletter generation failed: {e}") from e

if __name__ == "__main__":
    print(NewsletterGenerator("Latest developments in AI"))