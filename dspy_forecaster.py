"""
DSPy-based forecaster modules for Metaculus prediction bot.
Replaces hardcoded prompts with ChainOfThought for better reasoning quality.
"""
import asyncio
import os
import logging
from typing import Optional

import dspy
from dspy.teleprompt import BootstrapFewShot

logger = logging.getLogger(__name__)


def configure_dspy_lm(model: str = "gpt-4o-mini", temperature: float = 0.3) -> dspy.LM:
    """Configure DSPy LM with 云雾 API."""
    lm = dspy.LM(
        model=f"openai/{model}",
        temperature=temperature,
        api_base="https://api.wlai.vip/v1",
        api_key=os.getenv("OPENAI_API_KEY"),
        cache=False,  # Must be False: 5 predictions per question need diversity
    )
    dspy.configure(lm=lm)
    return lm


# ─────────────────────────── Signatures ───────────────────────────

class BinaryForecastSignature(dspy.Signature):
    """You are a professional forecaster. Analyze the question and research,
    reason step by step, then give a calibrated probability.
    Good forecasters put extra weight on the status quo since the world changes slowly."""

    question_text: str = dspy.InputField(desc="The forecasting question")
    background_info: str = dspy.InputField(desc="Background context for the question")
    resolution_criteria: str = dspy.InputField(desc="Exact criteria for how the question resolves")
    fine_print: str = dspy.InputField(desc="Additional fine print and edge cases")
    research: str = dspy.InputField(desc="Recent news and research findings")
    today_date: str = dspy.InputField(desc="Today's date in YYYY-MM-DD format")
    conditional_disclaimer: str = dspy.InputField(
        desc="Conditional question disclaimer, empty string if not applicable"
    )

    reasoning: str = dspy.OutputField(
        desc=(
            "Step-by-step reasoning covering: "
            "(a) time left until outcome is known, "
            "(b) historical base rate: how often do similar events occur historically? Give a concrete percentage estimate, "
            "(c) status quo outcome if nothing changed, "
            "(d) strongest evidence supporting Yes, "
            "(e) strongest evidence supporting No, "
            "(f) final adjustment relative to base rate and direction of change. "
            "Be willing to give probabilities above 80% or below 20% when evidence is strong. "
            "Avoid anchoring near 50% unless genuinely uncertain."
        )
    )
    probability: str = dspy.OutputField(
        desc=(
            "Final answer as exactly 'Probability: ZZ%' where ZZ is a SINGLE integer 0-100. "
            "Never output a range like '5-10%'. Pick one number. "
            "Do not anchor near 50% unless genuinely uncertain. "
            "If evidence strongly favors one side, go above 80% or below 20%."
        )
    )


class MultipleChoiceForecastSignature(dspy.Signature):
    """You are a professional forecaster. Analyze the question and assign
    probabilities to each option. Leave moderate probability on most options
    to account for unexpected outcomes."""

    question_text: str = dspy.InputField(desc="The forecasting question")
    options: str = dspy.InputField(desc="The list of options to assign probabilities to")
    background_info: str = dspy.InputField(desc="Background context for the question")
    resolution_criteria: str = dspy.InputField(desc="Exact criteria for how the question resolves")
    fine_print: str = dspy.InputField(desc="Additional fine print and edge cases")
    research: str = dspy.InputField(desc="Recent news and research findings")
    today_date: str = dspy.InputField(desc="Today's date in YYYY-MM-DD format")
    conditional_disclaimer: str = dspy.InputField(
        desc="Conditional question disclaimer, empty string if not applicable"
    )

    reasoning: str = dspy.OutputField(
        desc=(
            "Step-by-step reasoning covering: "
            "(a) time left until outcome is known, "
            "(b) equal base rate for each option (1/N where N is number of options), "
            "(c) status quo most likely outcome, "
            "(d) evidence-based adjustment for each option relative to base rate, "
            "(e) scenario leading to an unexpected outcome. "
            "Start from equal base rates then adjust based on evidence."
        )
    )
    probabilities: str = dspy.OutputField(
        desc=(
            "Final probabilities starting from equal base rates (1/N each), "
            "adjusted based on evidence. Probabilities MUST sum to exactly 100%. "
            "Use whole numbers only (e.g. 40, not 0.4). "
            "Format: Option_A: P_A\nOption_B: P_B\n..."
        )
    )


class NumericForecastSignature(dspy.Signature):
    """You are a professional forecaster. Produce a probability distribution
    over a numeric outcome expressed as percentiles. Be humble and set wide
    90/10 confidence intervals to account for unknown unknowns."""

    question_text: str = dspy.InputField(desc="The forecasting question")
    background_info: str = dspy.InputField(desc="Background context for the question")
    resolution_criteria: str = dspy.InputField(desc="Exact criteria for how the question resolves")
    fine_print: str = dspy.InputField(desc="Additional fine print and edge cases")
    unit_of_measure: str = dspy.InputField(desc="Units for the numeric answer")
    research: str = dspy.InputField(desc="Recent news and research findings")
    today_date: str = dspy.InputField(desc="Today's date in YYYY-MM-DD format")
    lower_bound_message: str = dspy.InputField(desc="Lower bound constraint for the answer")
    upper_bound_message: str = dspy.InputField(desc="Upper bound constraint for the answer")
    conditional_disclaimer: str = dspy.InputField(
        desc="Conditional question disclaimer, empty string if not applicable"
    )

    reasoning: str = dspy.OutputField(
        desc=(
            "Step-by-step reasoning covering: "
            "(a) time left until outcome is known, "
            "(b) outcome if nothing changed (anchor value with specific number), "
            "(c) outcome if current trend continued, "
            "(d) expert and market expectations with specific numbers if available, "
            "(e) P10 extreme low scenario: what specific conditions would cause this low value, "
            "(f) P90 extreme high scenario: what specific conditions would cause this high value. "
            "P10 and P90 should be wide enough to capture genuine uncertainty. "
            "Good forecasters are humble and avoid overconfident narrow intervals."
        )
    )
    percentiles: str = dspy.OutputField(
        desc=(
            "Final distribution in this exact format (values must be ascending, "
            "never use scientific notation, give answer in the requested units):\n"
            "Percentile 10: XX\n"
            "Percentile 20: XX\n"
            "Percentile 40: XX\n"
            "Percentile 60: XX\n"
            "Percentile 80: XX\n"
            "Percentile 90: XX"
        )
    )


# ─────────────────────────── Modules ───────────────────────────

class BinaryForecaster(dspy.Module):
    def __init__(self):
        super().__init__()
        self.predict = dspy.ChainOfThought(BinaryForecastSignature)

    def forward(self, **kwargs) -> str:
        result = self.predict(**kwargs)
        return f"{result.reasoning}\n\n{result.probability}"


class MultipleChoiceForecaster(dspy.Module):
    def __init__(self):
        super().__init__()
        self.predict = dspy.ChainOfThought(MultipleChoiceForecastSignature)

    def forward(self, **kwargs) -> str:
        result = self.predict(**kwargs)
        return f"{result.reasoning}\n\n{result.probabilities}"


class NumericForecaster(dspy.Module):
    def __init__(self):
        super().__init__()
        self.predict = dspy.ChainOfThought(NumericForecastSignature)

    def forward(self, **kwargs) -> str:
        result = self.predict(**kwargs)
        return f"{result.reasoning}\n\n{result.percentiles}"


# ─────────────────────────── Hub ───────────────────────────

class DSPyForecasterHub:
    """
    Singleton hub holding all DSPy forecaster modules.
    Use DSPyForecasterHub.get_instance() to avoid re-configuring the LM.
    """
    _instance: Optional["DSPyForecasterHub"] = None

    def __init__(self, model: str = "gpt-4o-mini", temperature: float = 0.3):
        configure_dspy_lm(model, temperature)
        self.binary = BinaryForecaster()
        self.multiple_choice = MultipleChoiceForecaster()
        self.numeric = NumericForecaster()
        logger.info(f"DSPyForecasterHub initialized with model={model}, temperature={temperature}")

    @classmethod
    def get_instance(cls, model: str = "gpt-4o-mini", temperature: float = 0.3) -> "DSPyForecasterHub":
        if cls._instance is None:
            cls._instance = cls(model=model, temperature=temperature)
        return cls._instance

    async def forecast_binary(self, **kwargs) -> str:
        """Async wrapper: runs DSPy (sync) in a thread pool executor."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: self.binary(**kwargs))

    async def forecast_multiple_choice(self, **kwargs) -> str:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: self.multiple_choice(**kwargs))

    async def forecast_numeric(self, **kwargs) -> str:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: self.numeric(**kwargs))


# ─────────────────────────── Optimizer (stub) ───────────────────────────

def binary_metric(example: dspy.Example, prediction, trace=None) -> float:
    """
    Brier score metric for binary forecasts.
    Requires example.resolved_value (0.0 or 1.0) and prediction.probability string.
    Returns score in [0, 1] where higher is better.
    """
    import re
    try:
        resolved = float(example.resolved_value)
        match = re.search(r"(\d+(?:\.\d+)?)\s*%", str(prediction.probability))
        if not match:
            return 0.0
        predicted_prob = float(match.group(1)) / 100.0
        brier = (predicted_prob - resolved) ** 2
        return 1.0 - brier  # higher = better
    except Exception:
        return 0.0


def optimize_forecaster(
    module: dspy.Module,
    trainset: list,
    metric_fn=binary_metric,
    max_bootstrapped_demos: int = 4,
) -> dspy.Module:
    """
    Optimize a forecaster module using BootstrapFewShot.

    Args:
        module: A BinaryForecaster / MultipleChoiceForecaster / NumericForecaster instance
        trainset: List of dspy.Example with input fields + resolved_value
        metric_fn: callable(example, prediction) -> float, higher is better
        max_bootstrapped_demos: Number of few-shot examples to bootstrap

    Returns:
        Compiled (optimized) module with injected few-shot examples

    Example trainset entry:
        dspy.Example(
            question_text="Will X happen by Y?",
            background_info="...",
            resolution_criteria="...",
            fine_print="",
            research="...",
            today_date="2025-01-15",
            conditional_disclaimer="",
            resolved_value=1.0,
        ).with_inputs(
            "question_text", "background_info", "resolution_criteria",
            "fine_print", "research", "today_date", "conditional_disclaimer"
        )
    """
    teleprompter = BootstrapFewShot(
        metric=metric_fn,
        max_bootstrapped_demos=max_bootstrapped_demos,
        max_labeled_demos=4,
    )
    return teleprompter.compile(module, trainset=trainset)
