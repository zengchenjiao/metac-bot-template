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


def configure_dspy_lm(model: str = "gpt-4o", temperature: float = 0.3) -> dspy.LM:
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

    def forward(self, **kwargs):
        return self.predict(**kwargs)


class MultipleChoiceForecaster(dspy.Module):
    def __init__(self):
        super().__init__()
        self.predict = dspy.ChainOfThought(MultipleChoiceForecastSignature)

    def forward(self, **kwargs):
        return self.predict(**kwargs)


class NumericForecaster(dspy.Module):
    def __init__(self):
        super().__init__()
        self.predict = dspy.ChainOfThought(NumericForecastSignature)

    def forward(self, **kwargs):
        return self.predict(**kwargs)


# ─────────────────────────── Hub ───────────────────────────

OPTIMIZED_BINARY_PATH = "json/optimized_binary_forecaster.json"
OPTIMIZED_MC_PATH = "json/optimized_mc_forecaster.json"
OPTIMIZED_NUMERIC_PATH = "json/optimized_numeric_forecaster.json"


class DSPyForecasterHub:
    """
    Singleton hub holding all DSPy forecaster modules.
    Use DSPyForecasterHub.get_instance() to avoid re-configuring the LM.
    """
    _instance: Optional["DSPyForecasterHub"] = None

    def __init__(self, model: str = "gpt-4o", temperature: float = 0.3):
        configure_dspy_lm(model, temperature)
        self.binary = BinaryForecaster()
        self.multiple_choice = MultipleChoiceForecaster()
        self.numeric = NumericForecaster()
        for attr, path in [
            ("binary", OPTIMIZED_BINARY_PATH),
            ("multiple_choice", OPTIMIZED_MC_PATH),
            ("numeric", OPTIMIZED_NUMERIC_PATH),
        ]:
            if os.path.exists(path):
                getattr(self, attr).load(path)
                logger.info(f"Loaded optimized {attr} forecaster from {path}")
            else:
                logger.info(f"No optimized {attr} forecaster found, using default")
        logger.info(f"DSPyForecasterHub initialized with model={model}, temperature={temperature}")

    @classmethod
    def get_instance(cls, model: str = "gpt-4o", temperature: float = 0.3) -> "DSPyForecasterHub":
        if cls._instance is None:
            cls._instance = cls(model=model, temperature=temperature)
        return cls._instance

    @classmethod
    def reload(cls, model: str = "gpt-4o", temperature: float = 0.3) -> "DSPyForecasterHub":
        """Force reload optimized models from disk without restarting the process."""
        cls._instance = None
        return cls.get_instance(model, temperature)

    async def forecast_binary(self, **kwargs) -> str:
        """Async wrapper: runs DSPy (sync) in a thread pool executor."""
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, lambda: self.binary(**kwargs))
        return f"{result.reasoning}\n\n{result.probability}"

    async def forecast_multiple_choice(self, **kwargs) -> str:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, lambda: self.multiple_choice(**kwargs))
        return f"{result.reasoning}\n\n{result.probabilities}"

    async def forecast_numeric(self, **kwargs) -> str:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, lambda: self.numeric(**kwargs))
        return f"{result.reasoning}\n\n{result.percentiles}"


# ─────────────────────────── Metrics ───────────────────────────

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


def mc_metric(example: dspy.Example, prediction, trace=None) -> float:
    """
    Brier score metric for multiple choice forecasts.
    Requires example.resolved_index (int) and example.options (list[str]).
    Parses prediction.probabilities text like "Option A: 40\nOption B: 60\n..."
    Returns score in [0, 1] where higher is better.
    """
    import re
    try:
        options: list = example.options
        resolved_idx: int = int(example.resolved_index)
        n = len(options)

        # Parse "OptionName: XX" lines from probabilities output
        prob_text = str(prediction.probabilities)
        probs = [0.0] * n
        total = 0.0
        for i, opt in enumerate(options):
            # Match the option name followed by a colon and number
            pattern = re.escape(str(opt)) + r"[:\s]+(\d+(?:\.\d+)?)"
            match = re.search(pattern, prob_text, re.IGNORECASE)
            if match:
                p = float(match.group(1))
                # Values may be 0-100 or 0-1
                if p > 1.0:
                    p = p / 100.0
                probs[i] = p
                total += p

        # Normalize if needed
        if total > 0:
            probs = [p / total for p in probs]
        else:
            probs = [1.0 / n] * n  # fallback: uniform

        # Brier score: sum((p_i - indicator_i)^2) / 2
        brier = sum((probs[i] - (1.0 if i == resolved_idx else 0.0)) ** 2 for i in range(n)) / 2.0
        return 1.0 - brier
    except Exception:
        return 0.0


def numeric_metric(example: dspy.Example, prediction, trace=None) -> float:
    """
    MAE-based metric for numeric forecasts (normalized to [0,1]).
    Requires example.resolved_normalized (float in [0,1]) — the answer
    normalized via Autocast's log-scale formula.
    Parses the median (P50 or average of P40/P60) from prediction.percentiles.
    Returns score in [0, 1] where higher is better.
    """
    import re
    from build_trainset import autocast_normalize
    try:
        resolved = float(example.resolved_normalized)
        percentile_text = str(prediction.percentiles)

        # Extract all "Percentile XX: VALUE" pairs
        matches = re.findall(r"Percentile\s+(\d+)\s*:\s*([\d.eE+\-]+)", percentile_text)
        if not matches:
            return 0.0

        percentile_map = {int(p): float(v) for p, v in matches}

        # Use P50 if available, else average P40 and P60
        if 50 in percentile_map:
            predicted = percentile_map[50]
        elif 40 in percentile_map and 60 in percentile_map:
            predicted = (percentile_map[40] + percentile_map[60]) / 2.0
        else:
            # Fallback: median of all parsed values
            vals = sorted(percentile_map.values())
            predicted = vals[len(vals) // 2]

        # Normalize model output (real units) to [0,1] before comparing
        choices = example.choices_meta if hasattr(example, "choices_meta") else {}
        if choices:
            predicted = autocast_normalize(predicted, choices)
        else:
            predicted = max(0.0, min(1.0, predicted))

        mae = abs(predicted - resolved)
        return 1.0 - mae  # higher is better
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

    从 trainset 中选出得分最高的 max_bootstrapped_demos 条作为 few-shot examples。
    """
    teleprompter = BootstrapFewShot(
        metric=metric_fn,
        max_bootstrapped_demos=max_bootstrapped_demos,
        max_labeled_demos=max_bootstrapped_demos,
        max_rounds=1,
    )
    return teleprompter.compile(module, trainset=trainset)
