"""
LangGraph-based forecasting agent.

Flow:
  START → [research] → [forecast] → [reflect] → (loop or END)

- research: calls TavilySearcher, appends results to state
- forecast: calls DSPyForecasterHub based on question_type
- reflect: LLM evaluates confidence; if < 0.65 and iterations < 3, loops back to research
"""
import json
import logging
import os
from datetime import datetime
from typing import TypedDict

from langgraph.graph import StateGraph, END

from dspy_forecaster import DSPyForecasterHub
from tavily_searcher import TavilySearcher

logger = logging.getLogger(__name__)

MAX_ITERATIONS = 3
CONFIDENCE_THRESHOLD = 0.65


# ─────────────────────────── State ───────────────────────────

class ForecastState(TypedDict):
    # Question fields (input)
    question_text: str
    background_info: str
    resolution_criteria: str
    fine_print: str
    question_type: str          # "binary" | "mc" | "numeric"
    options: str                # MC only
    unit_of_measure: str        # numeric only
    lower_bound_message: str    # numeric only
    upper_bound_message: str    # numeric only
    conditional_disclaimer: str
    today_date: str

    # Research accumulation (appended each loop)
    research_results: list[str]

    # Forecast output
    prediction_text: str        # raw DSPy output: "{reasoning}\n\n{answer}"

    # Control
    iterations: int
    confidence: float
    reflection_notes: str       # next search query hint from reflect node


# ─────────────────────────── Helpers ───────────────────────────

def _build_search_query(question_text: str, reflection_notes: str) -> str:
    """Build a focused search query from question text and reflection hints."""
    if reflection_notes:
        return reflection_notes[:200]
    # Extract core topic from question (first 150 chars is usually enough)
    return question_text[:150]


# ─────────────────────────── Nodes ───────────────────────────

async def research_node(state: ForecastState) -> dict:
    query = _build_search_query(state["question_text"], state["reflection_notes"])
    logger.info(f"[Agent] Research query: {query!r}")
    try:
        result = await TavilySearcher().search_news(query, max_results=10)
    except Exception as e:
        logger.warning(f"[Agent] Tavily search failed: {e}")
        result = ""
    return {
        "research_results": state["research_results"] + [result],
        "reflection_notes": "",  # clear after use
    }


async def forecast_node(state: ForecastState) -> dict:
    hub = DSPyForecasterHub.get_instance()
    combined_research = "\n\n---\n\n".join(
        r for r in state["research_results"] if r
    )
    qtype = state["question_type"]
    logger.info(f"[Agent] Forecasting type={qtype}, iteration={state['iterations']}")

    if qtype == "binary":
        result = await hub.forecast_binary(
            question_text=state["question_text"],
            background_info=state["background_info"],
            resolution_criteria=state["resolution_criteria"],
            fine_print=state["fine_print"],
            research=combined_research,
            today_date=state["today_date"],
            conditional_disclaimer=state["conditional_disclaimer"],
        )
    elif qtype == "mc":
        result = await hub.forecast_multiple_choice(
            question_text=state["question_text"],
            options=state["options"],
            background_info=state["background_info"],
            resolution_criteria=state["resolution_criteria"],
            fine_print=state["fine_print"],
            research=combined_research,
            today_date=state["today_date"],
            conditional_disclaimer=state["conditional_disclaimer"],
        )
    else:  # numeric
        result = await hub.forecast_numeric(
            question_text=state["question_text"],
            background_info=state["background_info"],
            resolution_criteria=state["resolution_criteria"],
            fine_print=state["fine_print"],
            unit_of_measure=state["unit_of_measure"],
            research=combined_research,
            today_date=state["today_date"],
            lower_bound_message=state["lower_bound_message"],
            upper_bound_message=state["upper_bound_message"],
            conditional_disclaimer=state["conditional_disclaimer"],
        )

    return {"prediction_text": result}


async def reflect_node(state: ForecastState) -> dict:
    """Ask LLM to evaluate prediction quality and suggest next search if needed."""
    from forecasting_tools import GeneralLlm

    llm = GeneralLlm(
        model="gpt-4o",
        temperature=0.3,
        timeout=60,
        allowed_tries=2,
        api_base=os.getenv("OPENAI_API_BASE", "https://api.wlai.vip/v1"),
    )

    prompt = f"""You are evaluating a forecasting prediction for quality.

Question: {state["question_text"]}

Research conducted ({len(state["research_results"])} searches):
{chr(10).join(f"Search {i+1}: {r[:300]}..." for i, r in enumerate(state["research_results"]) if r)}

Current prediction:
{state["prediction_text"][:800]}

Evaluate the prediction and respond ONLY with valid JSON (no markdown, no extra text):
{{
  "confidence": <float 0.0-1.0, how confident you are in this prediction>,
  "missing_info": "<what key information is missing, if any>",
  "next_query": "<specific search query that would help most, empty string if not needed>"
}}

Guidelines:
- confidence >= 0.65 means the prediction is well-supported and no more research is needed
- confidence < 0.65 means more research would meaningfully improve the prediction
- next_query should be specific and different from previous searches"""

    try:
        response = await llm.invoke(prompt)
        # Strip markdown code fences if present
        cleaned = response.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("```")[1]
            if cleaned.startswith("json"):
                cleaned = cleaned[4:]
        parsed = json.loads(cleaned.strip())
        confidence = float(parsed.get("confidence", 0.7))
        next_query = str(parsed.get("next_query", ""))
    except Exception as e:
        logger.warning(f"[Agent] Reflect parse failed: {e}, defaulting confidence=0.7")
        confidence = 0.7
        next_query = ""

    logger.info(f"[Agent] Reflect: confidence={confidence:.2f}, next_query={next_query!r}")
    return {
        "confidence": confidence,
        "reflection_notes": next_query,
        "iterations": state["iterations"] + 1,
    }


# ─────────────────────────── Routing ───────────────────────────

def should_continue(state: ForecastState) -> str:
    if state["iterations"] >= MAX_ITERATIONS:
        logger.info(f"[Agent] Max iterations reached, stopping.")
        return END
    if state["confidence"] < CONFIDENCE_THRESHOLD:
        logger.info(f"[Agent] Low confidence ({state['confidence']:.2f}), doing more research.")
        return "research"
    logger.info(f"[Agent] Confidence sufficient ({state['confidence']:.2f}), stopping.")
    return END


# ─────────────────────────── Graph ───────────────────────────

def build_forecast_agent():
    graph = StateGraph(ForecastState)
    graph.add_node("research", research_node)
    graph.add_node("forecast", forecast_node)
    graph.add_node("reflect", reflect_node)

    graph.set_entry_point("research")
    graph.add_edge("research", "forecast")
    graph.add_edge("forecast", "reflect")
    graph.add_conditional_edges("reflect", should_continue, {
        "research": "research",
        END: END,
    })

    return graph.compile()


# ─────────────────────────── State builder ───────────────────────────

def build_initial_state(
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
) -> ForecastState:
    return ForecastState(
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
        research_results=[],
        prediction_text="",
        iterations=0,
        confidence=0.0,
        reflection_notes="",
    )
