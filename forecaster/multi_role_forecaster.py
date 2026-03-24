"""
Multi-role forecasting system with 5 specialized agents and meta-predictor aggregation.

Each agent has a unique perspective, search strategy, and DSPy Signature.
Final prediction is produced by a meta-predictor that synthesizes all 5 agents' reasoning.
"""
import asyncio
import json
import logging
import os
from datetime import datetime
from typing import TypedDict

import dspy
from langgraph.graph import StateGraph, END

from forecaster.dspy_forecaster import DSPyForecasterHub, configure_dspy_lm
from forecaster.tavily_searcher import TavilySearcher
import config.settings as cfg

logger = logging.getLogger(__name__)


# ─────────────────────────── Role-Specific Signatures ───────────────────────────

# Shared input fields for all roles (defined as a helper)
_COMMON_INPUT_FIELDS = {
    "question_text": dspy.InputField(desc="The forecasting question"),
    "background_info": dspy.InputField(desc="Background context for the question"),
    "resolution_criteria": dspy.InputField(desc="Exact criteria for how the question resolves"),
    "fine_print": dspy.InputField(desc="Additional fine print and edge cases"),
    "research": dspy.InputField(desc="Recent news and research findings"),
    "today_date": dspy.InputField(desc="Today's date in YYYY-MM-DD format"),
    "conditional_disclaimer": dspy.InputField(desc="Conditional question disclaimer, empty string if not applicable"),
}

_BINARY_PROB_OUTPUT = dspy.OutputField(
    desc=(
        "Final answer as exactly 'Probability: ZZ%' where ZZ is a SINGLE integer 0-100. "
        "Never output a range. Pick one number."
    )
)

# ── Base Rate Analyst ──

class BaseRateBinarySignature(dspy.Signature):
    """You are a base rate analyst specializing in reference class forecasting.
    Always identify the most relevant reference class first, then adjust minimally."""

    question_text: str = dspy.InputField(desc="The forecasting question")
    background_info: str = dspy.InputField(desc="Background context for the question")
    resolution_criteria: str = dspy.InputField(desc="Exact criteria for how the question resolves")
    fine_print: str = dspy.InputField(desc="Additional fine print and edge cases")
    research: str = dspy.InputField(desc="Historical data and reference class statistics")
    today_date: str = dspy.InputField(desc="Today's date in YYYY-MM-DD format")
    conditional_disclaimer: str = dspy.InputField(desc="Conditional question disclaimer, empty string if not applicable")

    reasoning: str = dspy.OutputField(
        desc=(
            "Step-by-step reasoning: "
            "(a) Identify the most relevant reference class for this question, "
            "(b) What is the historical base rate for this reference class? Give a concrete percentage, "
            "(c) List specific factors that justify deviating from the base rate, "
            "(d) For each factor, estimate the magnitude and direction of adjustment, "
            "(e) Final probability = base rate + sum of adjustments. Show the math."
        )
    )
    probability: str = dspy.OutputField(
        desc="Final answer as exactly 'Probability: ZZ%' where ZZ is a SINGLE integer 0-100."
    )


class BaseRateMCSignature(dspy.Signature):
    """You are a base rate analyst. Start from equal base rates (1/N) and adjust
    only when historical data strongly supports a different distribution."""

    question_text: str = dspy.InputField(desc="The forecasting question")
    options: str = dspy.InputField(desc="The list of options to assign probabilities to")
    background_info: str = dspy.InputField(desc="Background context for the question")
    resolution_criteria: str = dspy.InputField(desc="Exact criteria for how the question resolves")
    fine_print: str = dspy.InputField(desc="Additional fine print and edge cases")
    research: str = dspy.InputField(desc="Historical data and reference class statistics")
    today_date: str = dspy.InputField(desc="Today's date in YYYY-MM-DD format")
    conditional_disclaimer: str = dspy.InputField(desc="Conditional question disclaimer, empty string if not applicable")

    reasoning: str = dspy.OutputField(
        desc=(
            "Step-by-step reasoning: "
            "(a) Equal base rate = 1/N for each option, "
            "(b) Historical frequency of each option in the reference class, "
            "(c) Adjust each option's probability based on historical data, "
            "(d) Verify probabilities sum to 100%."
        )
    )
    probabilities: str = dspy.OutputField(
        desc="Probabilities for each option summing to 100%. Format: OptionName: XX\n for each."
    )


class BaseRateNumericSignature(dspy.Signature):
    """You are a base rate analyst. Anchor on historical median values and
    set wide confidence intervals based on historical variance."""

    question_text: str = dspy.InputField(desc="The forecasting question")
    background_info: str = dspy.InputField(desc="Background context for the question")
    resolution_criteria: str = dspy.InputField(desc="Exact criteria for how the question resolves")
    fine_print: str = dspy.InputField(desc="Additional fine print and edge cases")
    unit_of_measure: str = dspy.InputField(desc="Units for the numeric answer")
    research: str = dspy.InputField(desc="Historical data and reference class statistics")
    today_date: str = dspy.InputField(desc="Today's date in YYYY-MM-DD format")
    lower_bound_message: str = dspy.InputField(desc="Lower bound constraint")
    upper_bound_message: str = dspy.InputField(desc="Upper bound constraint")
    conditional_disclaimer: str = dspy.InputField(desc="Conditional question disclaimer, empty string if not applicable")

    reasoning: str = dspy.OutputField(
        desc=(
            "Step-by-step reasoning: "
            "(a) Historical median value for this type of outcome, "
            "(b) Historical variance and range, "
            "(c) Adjustments based on current conditions, "
            "(d) Set P10/P90 wide enough to capture historical variance."
        )
    )
    percentiles: str = dspy.OutputField(
        desc="Percentile 10: XX\nPercentile 20: XX\nPercentile 40: XX\nPercentile 60: XX\nPercentile 80: XX\nPercentile 90: XX"
    )


# ── News Analyst ──

class NewsAnalystBinarySignature(dspy.Signature):
    """You are a news analyst. Focus on what has changed recently that shifts
    the probability. Weight recent credible reporting heavily."""

    question_text: str = dspy.InputField(desc="The forecasting question")
    background_info: str = dspy.InputField(desc="Background context for the question")
    resolution_criteria: str = dspy.InputField(desc="Exact criteria for how the question resolves")
    fine_print: str = dspy.InputField(desc="Additional fine print and edge cases")
    research: str = dspy.InputField(desc="Latest news articles and recent developments")
    today_date: str = dspy.InputField(desc="Today's date in YYYY-MM-DD format")
    conditional_disclaimer: str = dspy.InputField(desc="Conditional question disclaimer, empty string if not applicable")

    reasoning: str = dspy.OutputField(
        desc=(
            "Step-by-step reasoning: "
            "(a) What is the default/prior expectation for this question? "
            "(b) What recent news (last 7-30 days) is directly relevant? Cite specific articles, "
            "(c) How does each piece of news shift the probability and in which direction? "
            "(d) Are the news sources credible? Discount speculation and opinion, "
            "(e) Net probability shift from recent developments."
        )
    )
    probability: str = dspy.OutputField(
        desc="Final answer as exactly 'Probability: ZZ%' where ZZ is a SINGLE integer 0-100."
    )


class NewsAnalystMCSignature(dspy.Signature):
    """You are a news analyst. Identify recent developments that favor specific options."""

    question_text: str = dspy.InputField(desc="The forecasting question")
    options: str = dspy.InputField(desc="The list of options to assign probabilities to")
    background_info: str = dspy.InputField(desc="Background context for the question")
    resolution_criteria: str = dspy.InputField(desc="Exact criteria for how the question resolves")
    fine_print: str = dspy.InputField(desc="Additional fine print and edge cases")
    research: str = dspy.InputField(desc="Latest news articles and recent developments")
    today_date: str = dspy.InputField(desc="Today's date in YYYY-MM-DD format")
    conditional_disclaimer: str = dspy.InputField(desc="Conditional question disclaimer, empty string if not applicable")

    reasoning: str = dspy.OutputField(
        desc=(
            "Step-by-step reasoning: "
            "(a) Default expectation for each option, "
            "(b) Recent news that favors or disfavors specific options, "
            "(c) Credibility assessment of each news source, "
            "(d) Adjusted probabilities based on news impact."
        )
    )
    probabilities: str = dspy.OutputField(
        desc="Probabilities for each option summing to 100%. Format: OptionName: XX\n for each."
    )


class NewsAnalystNumericSignature(dspy.Signature):
    """You are a news analyst. Use recent data points and trends to estimate numeric outcomes."""

    question_text: str = dspy.InputField(desc="The forecasting question")
    background_info: str = dspy.InputField(desc="Background context for the question")
    resolution_criteria: str = dspy.InputField(desc="Exact criteria for how the question resolves")
    fine_print: str = dspy.InputField(desc="Additional fine print and edge cases")
    unit_of_measure: str = dspy.InputField(desc="Units for the numeric answer")
    research: str = dspy.InputField(desc="Latest news articles and recent data points")
    today_date: str = dspy.InputField(desc="Today's date in YYYY-MM-DD format")
    lower_bound_message: str = dspy.InputField(desc="Lower bound constraint")
    upper_bound_message: str = dspy.InputField(desc="Upper bound constraint")
    conditional_disclaimer: str = dspy.InputField(desc="Conditional question disclaimer, empty string if not applicable")

    reasoning: str = dspy.OutputField(
        desc=(
            "Step-by-step reasoning: "
            "(a) Most recent data point or measurement, "
            "(b) Recent trend direction and magnitude, "
            "(c) Any announced changes or upcoming events that affect the number, "
            "(d) Extrapolate from recent trend to estimate percentiles."
        )
    )
    percentiles: str = dspy.OutputField(
        desc="Percentile 10: XX\nPercentile 20: XX\nPercentile 40: XX\nPercentile 60: XX\nPercentile 80: XX\nPercentile 90: XX"
    )


# ── Contrarian Analyst ──

class ContrarianBinarySignature(dspy.Signature):
    """You are a contrarian analyst. Challenge the obvious answer. Identify overlooked
    risks and tail events. If the consensus seems too confident, push back."""

    question_text: str = dspy.InputField(desc="The forecasting question")
    background_info: str = dspy.InputField(desc="Background context for the question")
    resolution_criteria: str = dspy.InputField(desc="Exact criteria for how the question resolves")
    fine_print: str = dspy.InputField(desc="Additional fine print and edge cases")
    research: str = dspy.InputField(desc="Research focused on counterarguments and risks")
    today_date: str = dspy.InputField(desc="Today's date in YYYY-MM-DD format")
    conditional_disclaimer: str = dspy.InputField(desc="Conditional question disclaimer, empty string if not applicable")

    reasoning: str = dspy.OutputField(
        desc=(
            "Step-by-step reasoning: "
            "(a) What is the obvious/consensus prediction? "
            "(b) What are the strongest arguments AGAINST the consensus? "
            "(c) What tail risks or black swan scenarios could change the outcome? "
            "(d) Are there historical examples where the consensus was wrong in similar situations? "
            "(e) How much should the probability shift away from consensus based on these concerns?"
        )
    )
    probability: str = dspy.OutputField(
        desc="Final answer as exactly 'Probability: ZZ%' where ZZ is a SINGLE integer 0-100."
    )


class ContrarianMCSignature(dspy.Signature):
    """You are a contrarian analyst. Look for undervalued options that others dismiss."""

    question_text: str = dspy.InputField(desc="The forecasting question")
    options: str = dspy.InputField(desc="The list of options to assign probabilities to")
    background_info: str = dspy.InputField(desc="Background context for the question")
    resolution_criteria: str = dspy.InputField(desc="Exact criteria for how the question resolves")
    fine_print: str = dspy.InputField(desc="Additional fine print and edge cases")
    research: str = dspy.InputField(desc="Research focused on counterarguments and risks")
    today_date: str = dspy.InputField(desc="Today's date in YYYY-MM-DD format")
    conditional_disclaimer: str = dspy.InputField(desc="Conditional question disclaimer, empty string if not applicable")

    reasoning: str = dspy.OutputField(
        desc=(
            "Step-by-step reasoning: "
            "(a) Which option does the consensus favor? "
            "(b) Why might the consensus be wrong? "
            "(c) Which undervalued options deserve more probability? "
            "(d) Redistribute probability giving more weight to overlooked options."
        )
    )
    probabilities: str = dspy.OutputField(
        desc="Probabilities for each option summing to 100%. Format: OptionName: XX\n for each."
    )


class ContrarianNumericSignature(dspy.Signature):
    """You are a contrarian analyst. Consider extreme scenarios and widen confidence intervals."""

    question_text: str = dspy.InputField(desc="The forecasting question")
    background_info: str = dspy.InputField(desc="Background context for the question")
    resolution_criteria: str = dspy.InputField(desc="Exact criteria for how the question resolves")
    fine_print: str = dspy.InputField(desc="Additional fine print and edge cases")
    unit_of_measure: str = dspy.InputField(desc="Units for the numeric answer")
    research: str = dspy.InputField(desc="Research focused on extreme scenarios and risks")
    today_date: str = dspy.InputField(desc="Today's date in YYYY-MM-DD format")
    lower_bound_message: str = dspy.InputField(desc="Lower bound constraint")
    upper_bound_message: str = dspy.InputField(desc="Upper bound constraint")
    conditional_disclaimer: str = dspy.InputField(desc="Conditional question disclaimer, empty string if not applicable")

    reasoning: str = dspy.OutputField(
        desc=(
            "Step-by-step reasoning: "
            "(a) What is the consensus estimate? "
            "(b) What extreme low scenario could occur? Be specific, "
            "(c) What extreme high scenario could occur? Be specific, "
            "(d) Set wider P10/P90 than the consensus would suggest."
        )
    )
    percentiles: str = dspy.OutputField(
        desc="Percentile 10: XX\nPercentile 20: XX\nPercentile 40: XX\nPercentile 60: XX\nPercentile 80: XX\nPercentile 90: XX"
    )


# ── Community Anchor Analyst ──

class CommunityAnchorBinarySignature(dspy.Signature):
    """You are a community anchor analyst. Crowd wisdom is your strongest signal.
    Start from expert/community consensus and deviate only with strong evidence."""

    question_text: str = dspy.InputField(desc="The forecasting question")
    background_info: str = dspy.InputField(desc="Background context for the question")
    resolution_criteria: str = dspy.InputField(desc="Exact criteria for how the question resolves")
    fine_print: str = dspy.InputField(desc="Additional fine print and edge cases")
    research: str = dspy.InputField(desc="Expert forecasts, prediction market data, and community consensus")
    today_date: str = dspy.InputField(desc="Today's date in YYYY-MM-DD format")
    conditional_disclaimer: str = dspy.InputField(desc="Conditional question disclaimer, empty string if not applicable")

    reasoning: str = dspy.OutputField(
        desc=(
            "Step-by-step reasoning: "
            "(a) What do expert forecasters and prediction markets say? Give specific numbers if available, "
            "(b) What is the Metaculus community prediction if known? "
            "(c) Is there any strong evidence to deviate from the community consensus? "
            "(d) If deviating, explain why and by how much (typically 5-10% max), "
            "(e) Final probability anchored on community consensus with minimal adjustment."
        )
    )
    probability: str = dspy.OutputField(
        desc="Final answer as exactly 'Probability: ZZ%' where ZZ is a SINGLE integer 0-100."
    )


class CommunityAnchorMCSignature(dspy.Signature):
    """You are a community anchor analyst. Use expert consensus to weight options."""

    question_text: str = dspy.InputField(desc="The forecasting question")
    options: str = dspy.InputField(desc="The list of options to assign probabilities to")
    background_info: str = dspy.InputField(desc="Background context for the question")
    resolution_criteria: str = dspy.InputField(desc="Exact criteria for how the question resolves")
    fine_print: str = dspy.InputField(desc="Additional fine print and edge cases")
    research: str = dspy.InputField(desc="Expert forecasts, prediction market data, and community consensus")
    today_date: str = dspy.InputField(desc="Today's date in YYYY-MM-DD format")
    conditional_disclaimer: str = dspy.InputField(desc="Conditional question disclaimer, empty string if not applicable")

    reasoning: str = dspy.OutputField(
        desc=(
            "Step-by-step reasoning: "
            "(a) What do experts and prediction markets favor? "
            "(b) Assign probabilities close to expert consensus, "
            "(c) Only deviate if you have specific evidence experts may have missed, "
            "(d) Keep deviations small (5-10% per option max)."
        )
    )
    probabilities: str = dspy.OutputField(
        desc="Probabilities for each option summing to 100%. Format: OptionName: XX\n for each."
    )


class CommunityAnchorNumericSignature(dspy.Signature):
    """You are a community anchor analyst. Anchor on expert estimates for numeric values."""

    question_text: str = dspy.InputField(desc="The forecasting question")
    background_info: str = dspy.InputField(desc="Background context for the question")
    resolution_criteria: str = dspy.InputField(desc="Exact criteria for how the question resolves")
    fine_print: str = dspy.InputField(desc="Additional fine print and edge cases")
    unit_of_measure: str = dspy.InputField(desc="Units for the numeric answer")
    research: str = dspy.InputField(desc="Expert estimates, official forecasts, and market expectations")
    today_date: str = dspy.InputField(desc="Today's date in YYYY-MM-DD format")
    lower_bound_message: str = dspy.InputField(desc="Lower bound constraint")
    upper_bound_message: str = dspy.InputField(desc="Upper bound constraint")
    conditional_disclaimer: str = dspy.InputField(desc="Conditional question disclaimer, empty string if not applicable")

    reasoning: str = dspy.OutputField(
        desc=(
            "Step-by-step reasoning: "
            "(a) What do official forecasts and expert estimates say? Give specific numbers, "
            "(b) What do prediction markets imply? "
            "(c) Anchor percentiles on expert consensus, "
            "(d) Only widen intervals if experts disagree significantly."
        )
    )
    percentiles: str = dspy.OutputField(
        desc="Percentile 10: XX\nPercentile 20: XX\nPercentile 40: XX\nPercentile 60: XX\nPercentile 80: XX\nPercentile 90: XX"
    )


# ── Domain Expert Analyst ──

class DomainExpertBinarySignature(dspy.Signature):
    """You are a domain expert analyst. Use deep technical knowledge and causal
    reasoning to make predictions grounded in domain-specific models."""

    question_text: str = dspy.InputField(desc="The forecasting question")
    background_info: str = dspy.InputField(desc="Background context for the question")
    resolution_criteria: str = dspy.InputField(desc="Exact criteria for how the question resolves")
    fine_print: str = dspy.InputField(desc="Additional fine print and edge cases")
    research: str = dspy.InputField(desc="Academic papers, official reports, and domain-specific data")
    today_date: str = dspy.InputField(desc="Today's date in YYYY-MM-DD format")
    conditional_disclaimer: str = dspy.InputField(desc="Conditional question disclaimer, empty string if not applicable")

    reasoning: str = dspy.OutputField(
        desc=(
            "Step-by-step reasoning: "
            "(a) What domain does this question belong to? What are the key causal factors? "
            "(b) What do domain-specific models or frameworks predict? "
            "(c) What relevant data from official sources or academic research is available? "
            "(d) What are the key uncertainties in the domain model? "
            "(e) Final probability based on domain expertise and causal analysis."
        )
    )
    probability: str = dspy.OutputField(
        desc="Final answer as exactly 'Probability: ZZ%' where ZZ is a SINGLE integer 0-100."
    )


class DomainExpertMCSignature(dspy.Signature):
    """You are a domain expert. Use technical knowledge to evaluate each option."""

    question_text: str = dspy.InputField(desc="The forecasting question")
    options: str = dspy.InputField(desc="The list of options to assign probabilities to")
    background_info: str = dspy.InputField(desc="Background context for the question")
    resolution_criteria: str = dspy.InputField(desc="Exact criteria for how the question resolves")
    fine_print: str = dspy.InputField(desc="Additional fine print and edge cases")
    research: str = dspy.InputField(desc="Academic papers, official reports, and domain-specific data")
    today_date: str = dspy.InputField(desc="Today's date in YYYY-MM-DD format")
    conditional_disclaimer: str = dspy.InputField(desc="Conditional question disclaimer, empty string if not applicable")

    reasoning: str = dspy.OutputField(
        desc=(
            "Step-by-step reasoning: "
            "(a) What domain knowledge is relevant to each option? "
            "(b) What do domain-specific models predict for each option? "
            "(c) Which options are technically feasible vs unlikely? "
            "(d) Assign probabilities based on domain analysis."
        )
    )
    probabilities: str = dspy.OutputField(
        desc="Probabilities for each option summing to 100%. Format: OptionName: XX\n for each."
    )


class DomainExpertNumericSignature(dspy.Signature):
    """You are a domain expert. Use technical models and data to estimate numeric outcomes."""

    question_text: str = dspy.InputField(desc="The forecasting question")
    background_info: str = dspy.InputField(desc="Background context for the question")
    resolution_criteria: str = dspy.InputField(desc="Exact criteria for how the question resolves")
    fine_print: str = dspy.InputField(desc="Additional fine print and edge cases")
    unit_of_measure: str = dspy.InputField(desc="Units for the numeric answer")
    research: str = dspy.InputField(desc="Academic papers, official reports, and domain-specific data")
    today_date: str = dspy.InputField(desc="Today's date in YYYY-MM-DD format")
    lower_bound_message: str = dspy.InputField(desc="Lower bound constraint")
    upper_bound_message: str = dspy.InputField(desc="Upper bound constraint")
    conditional_disclaimer: str = dspy.InputField(desc="Conditional question disclaimer, empty string if not applicable")

    reasoning: str = dspy.OutputField(
        desc=(
            "Step-by-step reasoning: "
            "(a) What domain-specific models or formulas apply? "
            "(b) What are the key input parameters and their current values? "
            "(c) Model output: what does the technical analysis predict? "
            "(d) Uncertainty analysis: what are the main sources of error?"
        )
    )
    percentiles: str = dspy.OutputField(
        desc="Percentile 10: XX\nPercentile 20: XX\nPercentile 40: XX\nPercentile 60: XX\nPercentile 80: XX\nPercentile 90: XX"
    )


# ─────────────────────────── Role → Signature Mapping ───────────────────────────

ROLE_SIGNATURES = {
    "base_rate_analyst": {
        "binary": BaseRateBinarySignature,
        "mc": BaseRateMCSignature,
        "numeric": BaseRateNumericSignature,
    },
    "news_analyst": {
        "binary": NewsAnalystBinarySignature,
        "mc": NewsAnalystMCSignature,
        "numeric": NewsAnalystNumericSignature,
    },
    "contrarian_analyst": {
        "binary": ContrarianBinarySignature,
        "mc": ContrarianMCSignature,
        "numeric": ContrarianNumericSignature,
    },
    "community_anchor_analyst": {
        "binary": CommunityAnchorBinarySignature,
        "mc": CommunityAnchorMCSignature,
        "numeric": CommunityAnchorNumericSignature,
    },
    "domain_expert": {
        "binary": DomainExpertBinarySignature,
        "mc": DomainExpertMCSignature,
        "numeric": DomainExpertNumericSignature,
    },
}

# ─────────────────────────── Role Definitions ───────────────────────────

ROLES = {
    "base_rate_analyst": {
        "name": "Base Rate Analyst",
        "system_prompt": (
            "You are a base rate analyst. Your approach is grounded in historical statistics and reference class forecasting. "
            "Always start by identifying the most relevant reference class and its historical frequency. "
            "Anchor your prediction on base rates before adjusting for specific evidence. "
            "Be skeptical of narratives that deviate far from historical patterns without strong evidence."
        ),
        "search_prefix": cfg.ROLE_SEARCH_PREFIXES.get("base_rate_analyst", "historical statistics frequency rate"),
        "search_topic": cfg.ROLE_SEARCH_TOPICS.get("base_rate_analyst", "general"),
    },
    "news_analyst": {
        "name": "News Analyst",
        "system_prompt": (
            "You are a news analyst focused on the latest developments and breaking information. "
            "Your strength is identifying recent events that shift probabilities away from base rates. "
            "Focus on what has changed in the last 7-30 days that is directly relevant to the question. "
            "Weight recent credible reporting heavily, but discount speculation and opinion pieces."
        ),
        "search_prefix": cfg.ROLE_SEARCH_PREFIXES.get("news_analyst", "latest news update 2026"),
        "search_topic": cfg.ROLE_SEARCH_TOPICS.get("news_analyst", "news"),
    },
    "contrarian_analyst": {
        "name": "Contrarian Analyst",
        "system_prompt": (
            "You are a contrarian analyst. Your job is to challenge the consensus and identify overlooked risks or opportunities. "
            "Actively seek evidence that contradicts the most popular prediction. "
            "Consider tail risks, black swan events, and scenarios others might dismiss. "
            "If the obvious answer seems too easy, dig deeper for reasons it might be wrong."
        ),
        "search_prefix": cfg.ROLE_SEARCH_PREFIXES.get("contrarian_analyst", "criticism risk unlikely scenario counterargument"),
        "search_topic": cfg.ROLE_SEARCH_TOPICS.get("contrarian_analyst", "general"),
    },
    "community_anchor_analyst": {
        "name": "Community Anchor Analyst",
        "system_prompt": (
            "You are a community anchor analyst. You use crowd wisdom as your starting point. "
            "The Metaculus community prediction is a strong baseline — historically it outperforms most individual forecasters. "
            "Your job is to start from the community prediction and only deviate when you have specific, strong evidence. "
            "Small deviations (5-10%) from the community are more common than large ones."
        ),
        "search_prefix": cfg.ROLE_SEARCH_PREFIXES.get("community_anchor_analyst", "expert forecast prediction consensus"),
        "search_topic": cfg.ROLE_SEARCH_TOPICS.get("community_anchor_analyst", "general"),
    },
    "domain_expert": {
        "name": "Domain Expert Analyst",
        "system_prompt": (
            "You are a domain expert analyst. You focus on deep technical and domain-specific knowledge. "
            "Seek out academic papers, official reports, government data, and expert analyses. "
            "Understand the underlying mechanisms and causal factors driving the outcome. "
            "Your predictions should be grounded in domain-specific models and data, not just surface-level news."
        ),
        "search_prefix": cfg.ROLE_SEARCH_PREFIXES.get("domain_expert", "research study report data analysis"),
        "search_topic": cfg.ROLE_SEARCH_TOPICS.get("domain_expert", "general"),
    },
}


# ─────────────────────────── Agent State ───────────────────────────

class RoleAgentState(TypedDict):
    # Question fields
    question_text: str
    background_info: str
    resolution_criteria: str
    fine_print: str
    question_type: str
    options: str
    unit_of_measure: str
    lower_bound_message: str
    upper_bound_message: str
    conditional_disclaimer: str
    today_date: str
    # Role config
    role_id: str
    role_system_prompt: str
    search_prefix: str
    search_topic: str
    # Research
    research_results: list[str]
    # Output
    prediction_text: str
    reasoning_text: str


# ─────────────────────────── Role Agent Nodes ───────────────────────────

async def role_research_node(state: RoleAgentState) -> dict:
    """Single research pass tailored to the role's search strategy."""
    role_id = state["role_id"]
    prefix = state["search_prefix"]
    topic = state["search_topic"]
    q = state["question_text"][:120]
    query = f"{prefix} {q}"

    logger.info(f"[{role_id}] Research query: {query[:100]!r}")
    try:
        searcher = TavilySearcher()
        # Use topic-specific search
        response = searcher.client.search(
            query=query,
            search_depth=cfg.TAVILY_SEARCH_DEPTH,
            topic=topic,
            max_results=cfg.TAVILY_MAX_RESULTS_PER_ROLE,
            include_answer=cfg.TAVILY_INCLUDE_ANSWER,
        )
        results = response.get("results", [])
        formatted = []
        for article in results:
            title = article.get("title", "")
            content = article.get("content", "")[:400]
            formatted.append(f"- {title}: {content}")
        research = "\n".join(formatted) if formatted else ""
    except Exception as e:
        logger.warning(f"[{role_id}] Search failed: {e}")
        research = ""

    return {"research_results": [research]}


async def role_forecast_node(state: RoleAgentState) -> dict:
    """Forecast using role-specific DSPy Signature for differentiated reasoning."""
    # Ensure DSPy LM is configured (reuses singleton)
    DSPyForecasterHub.get_instance()

    role_id = state["role_id"]
    qtype = state["question_type"]

    # Get role-specific signature
    sig_class = ROLE_SIGNATURES[role_id][qtype]
    predictor = dspy.ChainOfThought(sig_class)

    # Build research context
    combined_research = "\n---\n".join(r for r in state["research_results"] if r)

    logger.info(f"[{role_id}] Forecasting type={qtype}")

    # Build kwargs based on question type
    kwargs = {
        "question_text": state["question_text"],
        "background_info": state["background_info"],
        "resolution_criteria": state["resolution_criteria"],
        "fine_print": state["fine_print"],
        "research": combined_research,
        "today_date": state["today_date"],
        "conditional_disclaimer": state["conditional_disclaimer"],
    }

    if qtype == "mc":
        kwargs["options"] = state["options"]
    elif qtype == "numeric":
        kwargs["unit_of_measure"] = state["unit_of_measure"]
        kwargs["lower_bound_message"] = state["lower_bound_message"]
        kwargs["upper_bound_message"] = state["upper_bound_message"]

    # Run DSPy in thread pool (sync → async)
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, lambda: predictor(**kwargs))

    # Build output text
    if qtype == "binary":
        result_text = f"{result.reasoning}\n\n{result.probability}"
    elif qtype == "mc":
        result_text = f"{result.reasoning}\n\n{result.probabilities}"
    else:
        result_text = f"{result.reasoning}\n\n{result.percentiles}"

    return {
        "prediction_text": result_text,
        "reasoning_text": result_text,
    }


# ─────────────────────────── Role Agent Graph ───────────────────────────

def build_role_agent():
    """Build a simple research → forecast graph (no reflect loop, 1 iteration)."""
    graph = StateGraph(RoleAgentState)
    graph.add_node("research", role_research_node)
    graph.add_node("forecast", role_forecast_node)
    graph.set_entry_point("research")
    graph.add_edge("research", "forecast")
    graph.add_edge("forecast", END)
    return graph.compile()


def build_role_initial_state(
    role_id: str,
    question_text: str,
    question_type: str,
    background_info: str = "",
    resolution_criteria: str = "",
    fine_print: str = "",
    conditional_disclaimer: str = "",
    options: str = "",
    unit_of_measure: str = "",
    lower_bound_message: str = "",
    upper_bound_message: str = "",
) -> RoleAgentState:
    role_config = ROLES[role_id]
    return RoleAgentState(
        question_text=question_text,
        background_info=background_info,
        resolution_criteria=resolution_criteria,
        fine_print=fine_print,
        question_type=question_type,
        options=options,
        unit_of_measure=unit_of_measure,
        lower_bound_message=lower_bound_message,
        upper_bound_message=upper_bound_message,
        conditional_disclaimer=conditional_disclaimer,
        today_date=datetime.now().strftime("%Y-%m-%d"),
        role_id=role_id,
        role_system_prompt=role_config["system_prompt"],
        search_prefix=role_config["search_prefix"],
        search_topic=role_config["search_topic"],
        research_results=[],
        prediction_text="",
        reasoning_text="",
    )


# ─────────────────────────── Multi-Role Orchestrator ───────────────────────────

async def run_all_role_agents(
    question_text: str,
    question_type: str,
    background_info: str = "",
    resolution_criteria: str = "",
    fine_print: str = "",
    conditional_disclaimer: str = "",
    options: str = "",
    unit_of_measure: str = "",
    lower_bound_message: str = "",
    upper_bound_message: str = "",
    has_community_prediction: bool = True,
) -> dict[str, dict]:
    """Run role agents in parallel and return their results.
    When has_community_prediction is False, the Community Anchor Analyst is skipped."""
    agent = build_role_agent()

    # Determine which roles to run
    active_roles = {k: v for k, v in ROLES.items() if k in cfg.ENABLED_ROLES}
    if not has_community_prediction:
        active_roles = {k: v for k, v in ROLES.items() if k != "community_anchor_analyst"}
        logger.info("[MultiRole] No community prediction available — skipping Community Anchor Analyst")

    async def run_single_role(role_id: str) -> tuple[str, dict]:
        state = build_role_initial_state(
            role_id=role_id,
            question_text=question_text,
            question_type=question_type,
            background_info=background_info,
            resolution_criteria=resolution_criteria,
            fine_print=fine_print,
            conditional_disclaimer=conditional_disclaimer,
            options=options,
            unit_of_measure=unit_of_measure,
            lower_bound_message=lower_bound_message,
            upper_bound_message=upper_bound_message,
        )
        result = await agent.ainvoke(state)
        return role_id, result

    tasks = [run_single_role(role_id) for role_id in active_roles]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    role_results = {}
    for item in results:
        if isinstance(item, Exception):
            logger.error(f"[MultiRole] Agent failed: {item}")
            continue
        role_id, result = item
        role_results[role_id] = {
            "name": ROLES[role_id]["name"],
            "prediction_text": result.get("prediction_text", ""),
            "reasoning_text": result.get("reasoning_text", ""),
        }
        logger.info(f"[MultiRole] {ROLES[role_id]['name']} completed")

    return role_results


# ─────────────────────────── Meta-Predictor (方案C) ───────────────────────────

async def meta_predict(
    question_text: str,
    question_type: str,
    role_results: dict[str, dict],
    options: str = "",
    has_community_prediction: bool = True,
) -> str:
    """
    Meta-predictor: an LLM sees all 5 agents' reasoning and predictions,
    then produces a single final prediction.
    """
    from forecasting_tools import GeneralLlm

    llm = GeneralLlm(
        model=cfg.META_PREDICTOR_MODEL,
        temperature=cfg.META_PREDICTOR_TEMPERATURE,
        timeout=cfg.META_PREDICTOR_TIMEOUT,
        allowed_tries=cfg.META_PREDICTOR_ALLOWED_TRIES,
        api_base=cfg.OPENAI_API_BASE,
    )

    # Build summary of all agents
    agent_summaries = []
    for role_id, data in role_results.items():
        name = data["name"]
        pred = data["prediction_text"]
        # Truncate to keep prompt manageable
        pred_short = pred[:2000] if len(pred) > 2000 else pred
        agent_summaries.append(f"### {name}\n{pred_short}")

    agents_text = "\n\n".join(agent_summaries)

    if question_type == "binary":
        format_instruction = (
            "Give your final prediction as exactly 'Probability: ZZ%' where ZZ is a single integer 0-100. "
            "Do not give a range."
        )
    elif question_type == "mc":
        format_instruction = (
            f"The options are: {options}\n"
            "Give your final prediction as probabilities for each option that sum to 100%. "
            "Format: OptionName: XX%\n for each option."
        )
    else:  # numeric
        format_instruction = (
            "Give your final prediction as percentiles:\n"
            "Percentile 10: XX\nPercentile 20: XX\nPercentile 40: XX\n"
            "Percentile 60: XX\nPercentile 80: XX\nPercentile 90: XX"
        )

    prompt = f"""You are a meta-forecaster. You have received predictions from 5 specialized analysts, each with a different perspective. Your job is to synthesize their reasoning into a single, well-calibrated final prediction.

## Question
{question_text}

## Analyst Predictions

{agents_text}

## Your Task

1. Identify where the analysts agree — this is likely close to the truth.
2. Identify where they disagree — weigh the quality of evidence on each side.
3. {"Give extra weight to the Community Anchor Analyst (crowd wisdom is a strong baseline)." if has_community_prediction else "No community prediction is available. Give extra weight to the Base Rate Analyst and Domain Expert Analyst as primary anchors."}
4. Give extra weight to the Base Rate Analyst when other evidence is weak.
5. Be cautious about the Contrarian Analyst's view — only incorporate it if the evidence is compelling.
6. Produce a final, well-calibrated prediction with brief reasoning.

## Format
First give 2-3 sentences of synthesis reasoning, then:
{format_instruction}"""

    response = await llm.invoke(prompt)
    logger.info(f"[MetaPredictor] Synthesized prediction from {len(role_results)} agents")
    return response
