"""
Centralized configuration for the Metaculus forecasting bot.
All tunable parameters are defined here. Modify this file to customize behavior.
"""
import os

# ─────────────────────────── LLM Configuration ───────────────────────────

# API
OPENAI_API_BASE = os.getenv("OPENAI_API_BASE", "https://api.wlai.vip/v1")

# Model
LLM_MODEL = "gpt-4o"
LLM_TEMPERATURE = 0.3
LLM_TIMEOUT = 180  # seconds
LLM_ALLOWED_TRIES = 2

# DSPy LM (used by role agents for ChainOfThought)
DSPY_MODEL = "gpt-4o"
DSPY_TEMPERATURE = 0.3
DSPY_CACHE = False  # Must be False: multiple predictions per question need diversity

# DSPy Hub (used by main.py _get_dspy_hub)
DSPY_HUB_MODEL = "gpt-4o"
DSPY_HUB_TEMPERATURE = 0.5  # slightly higher for diversity

# Meta-predictor
META_PREDICTOR_MODEL = "gpt-4o"
META_PREDICTOR_TEMPERATURE = 0.2  # lower for more deterministic synthesis
META_PREDICTOR_TIMEOUT = 120
META_PREDICTOR_ALLOWED_TRIES = 2

# ─────────────────────────── Bot Configuration ───────────────────────────

# ForecastBot parameters
RESEARCH_REPORTS_PER_QUESTION = 1
PREDICTIONS_PER_RESEARCH_REPORT = 1
USE_RESEARCH_SUMMARY_TO_FORECAST = False
PUBLISH_REPORTS_TO_METACULUS = True
FOLDER_TO_SAVE_REPORTS_TO = None
SKIP_PREVIOUSLY_FORECASTED_QUESTIONS = True
EXTRA_METADATA_IN_EXPLANATION = True

# Concurrency
MAX_CONCURRENT_QUESTIONS = 1

# structure_output validation
STRUCTURE_OUTPUT_VALIDATION_SAMPLES = 2

# ─────────────────────────── Search Configuration ───────────────────────────

# Tavily
TAVILY_SEARCH_DEPTH = "advanced"
TAVILY_MAX_RESULTS_PER_ROLE = 5       # per role agent search
TAVILY_MAX_RESULTS_GENERAL = 10       # for general search_news()
TAVILY_CONTENT_TRUNCATE_LENGTH = 500  # chars per article content
TAVILY_INCLUDE_ANSWER = False

# ─────────────────────────── Role Agent Configuration ───────────────────────────

# Which roles to enable (order matters for display)
ENABLED_ROLES = [
    "base_rate_analyst",
    "news_analyst",
    "contrarian_analyst",
    "community_anchor_analyst",  # auto-skipped when no community prediction
    "domain_expert",
]

# Role search prefixes (customize search strategy per role)
ROLE_SEARCH_PREFIXES = {
    "base_rate_analyst": "historical statistics frequency rate",
    "news_analyst": "latest news update 2026",
    "contrarian_analyst": "criticism risk unlikely scenario counterargument",
    "community_anchor_analyst": "expert forecast prediction consensus",
    "domain_expert": "research study report data analysis",
}

# Role search topics (Tavily topic: "news" or "general")
ROLE_SEARCH_TOPICS = {
    "base_rate_analyst": "general",
    "news_analyst": "news",
    "contrarian_analyst": "general",
    "community_anchor_analyst": "general",
    "domain_expert": "general",
}

# ─────────────────────────── Optimized Model Paths ───────────────────────────

OPTIMIZED_BINARY_PATH = "json/optimized_binary_forecaster.json"
OPTIMIZED_MC_PATH = "json/optimized_mc_forecaster.json"
OPTIMIZED_NUMERIC_PATH = "json/optimized_numeric_forecaster.json"

# ─────────────────────────── Tournament IDs ───────────────────────────

DEFAULT_TOURNAMENT_ID = 32916

# ─────────────────────────── Test Questions ───────────────────────────

EXAMPLE_QUESTIONS = [
    "https://www.metaculus.com/questions/578/human-extinction-by-2100/",
    "https://www.metaculus.com/questions/22427/number-of-new-leading-ai-labs/",
]

# ─────────────────────────── Logging ───────────────────────────

LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
