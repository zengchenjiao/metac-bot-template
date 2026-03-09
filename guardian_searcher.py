"""
Guardian API Searcher for news research
"""
import os
import requests
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class GuardianSearcher:
    """Searcher that uses The Guardian API to fetch news articles"""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("GUARDIAN_API_KEY")
        if not self.api_key:
            raise ValueError("GUARDIAN_API_KEY not found in environment variables")
        self.base_url = "https://content.guardianapis.com/search"

    async def search_news(self, query: str, page_size: int = 10) -> str:
        """
        Search The Guardian for news articles related to the query

        Args:
            query: Search query string
            page_size: Number of results to return (max 50)

        Returns:
            Formatted string with news articles
        """
        params = {
            "q": query,
            "api-key": self.api_key,
            "page-size": min(page_size, 50),
            "show-fields": "headline,trailText,bodyText,byline,firstPublicationDate",
            "order-by": "relevance"
        }

        try:
            response = requests.get(self.base_url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()

            if data.get("response", {}).get("status") != "ok":
                logger.error(f"Guardian API error: {data}")
                return "No news articles found."

            results = data.get("response", {}).get("results", [])

            if not results:
                return "No news articles found for this query."

            # Format results
            formatted_articles = []
            for idx, article in enumerate(results, 1):
                fields = article.get("fields", {})
                title = fields.get("headline", article.get("webTitle", "No title"))
                trail_text = fields.get("trailText", "")
                body_text = fields.get("bodyText", "")
                byline = fields.get("byline", "")
                pub_date = fields.get("firstPublicationDate", article.get("webPublicationDate", ""))
                url = article.get("webUrl", "")

                # Truncate body text to first 500 chars
                if body_text:
                    body_text = body_text[:500] + "..." if len(body_text) > 500 else body_text

                article_text = f"""
Article {idx}: {title}
Published: {pub_date}
{f"By: {byline}" if byline else ""}
{trail_text}

{body_text}

URL: {url}
---
"""
                formatted_articles.append(article_text.strip())

            return "\n\n".join(formatted_articles)

        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching from Guardian API: {e}")
            return f"Error fetching news: {str(e)}"

    async def call_preconfigured_version(self, config: str, prompt: str) -> str:
        """
        Call with a preconfigured search strategy (for compatibility with AskNews interface)

        Args:
            config: Configuration string (e.g., "guardian/news-search")
            prompt: The search prompt/query

        Returns:
            Formatted news articles
        """
        # Extract the actual question from the prompt
        # The prompt format is usually: "...Question:\n{question_text}\n..."
        import re

        # Try to extract the question text
        question_match = re.search(r'Question:\s*\n(.+?)(?:\n\n|This question)', prompt, re.DOTALL)
        if question_match:
            query = question_match.group(1).strip()
            # Limit query length to avoid API errors
            query = query[:200] if len(query) > 200 else query
        else:
            # Fallback: use first 200 chars of prompt
            query = prompt[:200]

        logger.info(f"Guardian API search query: {query}")
        return await self.search_news(query, page_size=10)
