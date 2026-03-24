"""
Tavily Search for news research
"""
import os
import re
import logging
from typing import Optional

from tavily import TavilyClient

import config.settings as cfg

logger = logging.getLogger(__name__)


class TavilySearcher:
    """Searcher using Tavily API to fetch relevant news articles.
    Class name kept for backward compatibility with main.py imports.
    """

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("TAVILY_API_KEY")
        if not self.api_key:
            raise ValueError("TAVILY_API_KEY not found in environment variables")
        self.client = TavilyClient(api_key=self.api_key)

    async def search_news(self, query: str, max_results: int = 10) -> str:
        """
        Search for news articles using Tavily.

        Args:
            query: Search query string
            max_results: Number of results to return (max 10)

        Returns:
            Formatted string with news articles
        """
        try:
            response = self.client.search(
                query=query,
                search_depth=cfg.TAVILY_SEARCH_DEPTH,
                topic="news",
                max_results=min(max_results, 10),
                include_answer=cfg.TAVILY_INCLUDE_ANSWER,
            )

            results = response.get("results", [])
            if not results:
                return "No news articles found for this query."

            formatted_articles = []
            for idx, article in enumerate(results, 1):
                title = article.get("title", "No title")
                content = article.get("content", "")
                url = article.get("url", "")
                published_date = article.get("published_date", "")

                if content and len(content) > cfg.TAVILY_CONTENT_TRUNCATE_LENGTH:
                    content = content[:cfg.TAVILY_CONTENT_TRUNCATE_LENGTH] + "..."

                article_text = f"""Article {idx}: {title}
{f"Published: {published_date}" if published_date else ""}

{content}

URL: {url}
---"""
                formatted_articles.append(article_text.strip())

            return "\n\n".join(formatted_articles)

        except Exception as e:
            logger.error(f"Error fetching from Tavily API: {e}")
            return f"Error fetching news: {str(e)}"

    async def call_preconfigured_version(self, config: str, prompt: str) -> str:
        """Call with a preconfigured search strategy (tavily/news-search)."""
        # Extract question text from prompt
        question_match = re.search(
            r'Question:\s*\n(.+?)(?:\n\n|This question)', prompt, re.DOTALL
        )
        if question_match:
            query = question_match.group(1).strip()
            query = query[:200] if len(query) > 200 else query
        else:
            query = prompt[:200]

        logger.info(f"Tavily search query: {query}")
        return await self.search_news(query, max_results=10)
