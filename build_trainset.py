"""
从 Autocast 数据集构建 DSPy 训练集（支持 binary / mc / numeric 三种题型）。
用法: poetry run python build_trainset.py
"""
import json
import logging
import math
from pathlib import Path

import dspy

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

BINARY_OUTPUT = Path("autocast_binary_trainset.json")
MC_OUTPUT = Path("autocast_mc_trainset.json")
NUMERIC_OUTPUT = Path("autocast_numeric_trainset.json")

MAX_BINARY = 300
MAX_MC = 200
MAX_NUMERIC = 200

BINARY_INPUTS = (
    "question_text", "background_info", "resolution_criteria",
    "fine_print", "research", "today_date", "conditional_disclaimer",
)
MC_INPUTS = (
    "question_text", "options", "background_info", "resolution_criteria",
    "fine_print", "research", "today_date", "conditional_disclaimer",
)
NUMERIC_INPUTS = (
    "question_text", "background_info", "resolution_criteria", "fine_print",
    "unit_of_measure", "research", "today_date",
    "lower_bound_message", "upper_bound_message", "conditional_disclaimer",
)


# ─────────────────────────── Autocast numeric helpers ───────────────────────────

def autocast_denormalize(normalized: float, choices: dict) -> float:
    """
    Autocast 数值题的反归一化公式（log-scale）。
    choices 包含 min, max, deriv_ratio。
    """
    min_val = float(choices["min"])
    max_val = float(choices["max"])
    deriv_ratio = float(choices.get("deriv_ratio", 1.0))
    if abs(deriv_ratio - 1.0) < 1e-9:
        return min_val + (max_val - min_val) * normalized
    deriv_term = (deriv_ratio ** normalized - 1) / (deriv_ratio - 1)
    return min_val + (max_val - min_val) * deriv_term


def autocast_normalize(value: float, choices: dict) -> float:
    """将真实值归一化到 [0,1]（用于 numeric_metric）。"""
    min_val = float(choices["min"])
    max_val = float(choices["max"])
    deriv_ratio = float(choices.get("deriv_ratio", 1.0))
    if max_val <= min_val:
        return 0.5
    if abs(deriv_ratio - 1.0) < 1e-9:
        return (value - min_val) / (max_val - min_val)
    # Inverse of log-scale: solve for x in value = min + (max-min)*(r^x-1)/(r-1)
    frac = (value - min_val) / (max_val - min_val)
    frac = max(0.0, min(1.0, frac))
    inner = frac * (deriv_ratio - 1) + 1
    if inner <= 0:
        return 0.0
    return math.log(inner) / math.log(deriv_ratio)


# ─────────────────────────── Builders ───────────────────────────

RAW_TRAIN_URL = "https://huggingface.co/datasets/AlgoveraAI/autocast/resolve/main/autocast_questions.json"
LOCAL_RAW_PATH = Path("autocast_raw.json")


def _load_dataset() -> list:
    # 优先从本地缓存加载
    if LOCAL_RAW_PATH.exists():
        logger.info(f"Loading from local cache: {LOCAL_RAW_PATH}")
        data = json.loads(LOCAL_RAW_PATH.read_text())
        logger.info(f"Total examples: {len(data)}")
        return data

    # 用 requests 下载（自动读取 ALL_PROXY / HTTPS_PROXY 环境变量）
    import requests
    logger.info(f"Downloading raw dataset from GitHub: {RAW_TRAIN_URL}")
    resp = requests.get(RAW_TRAIN_URL, timeout=120)
    resp.raise_for_status()
    data = resp.json()

    # 缓存到本地
    LOCAL_RAW_PATH.write_text(json.dumps(data, ensure_ascii=False))
    logger.info(f"Cached {len(data)} examples to {LOCAL_RAW_PATH}")
    return data


def build_binary_trainset(
    raw: list | None = None, max_examples: int = MAX_BINARY
) -> list[dspy.Example]:
    if raw is None:
        raw = _load_dataset()

    trainset, skipped = [], 0
    for item in raw:
        if item["qtype"] != "t/f" or item["status"] != "Resolved":
            skipped += 1
            continue
        if item["answer"] not in ("yes", "no"):
            skipped += 1
            continue

        resolved_value = 1.0 if item["answer"] == "yes" else 0.0
        crowd_final = item["crowd"][-1] if item["crowd"] else None

        trainset.append(dspy.Example(
            question_text=item["question"] or "",
            background_info=item["background"] or "",
            resolution_criteria="",
            fine_print="",
            research="",
            today_date=(item["close_time"] or "2021-01-01")[:10],
            conditional_disclaimer="",
            resolved_value=resolved_value,
            crowd_final=crowd_final,
            question_id=item["id"],
        ).with_inputs(*BINARY_INPUTS))

        if len(trainset) >= max_examples:
            break

    yes = sum(1 for e in trainset if e.resolved_value == 1.0)
    logger.info(f"Binary trainset: {len(trainset)} (yes={yes}, no={len(trainset)-yes}, skipped={skipped})")
    return trainset


def build_mc_trainset(
    raw: list | None = None, max_examples: int = MAX_MC
) -> list[dspy.Example]:
    if raw is None:
        raw = _load_dataset()

    trainset, skipped = [], 0
    for item in raw:
        if item["qtype"] != "mc" or item["status"] != "Resolved":
            skipped += 1
            continue

        choices = item.get("choices") or []
        answer = item.get("answer", "")
        if not choices or not answer:
            skipped += 1
            continue

        # answer is a letter like "A", "B", "C"...
        answer_idx = ord(answer.upper()) - ord("A")
        if answer_idx < 0 or answer_idx >= len(choices):
            skipped += 1
            continue

        # crowd[-1] is a list of probabilities over choices
        crowd_final = item["crowd"][-1] if item["crowd"] else None

        trainset.append(dspy.Example(
            question_text=item["question"] or "",
            options=str(choices),
            background_info=item["background"] or "",
            resolution_criteria="",
            fine_print="",
            research="",
            today_date=(item["close_time"] or "2021-01-01")[:10],
            conditional_disclaimer="",
            # metric fields
            resolved_index=answer_idx,
            options_list=choices,
            crowd_final=crowd_final,
            question_id=item["id"],
        ).with_inputs(*MC_INPUTS))

        if len(trainset) >= max_examples:
            break

    logger.info(f"MC trainset: {len(trainset)} (skipped={skipped})")
    return trainset


def build_numeric_trainset(
    raw: list | None = None, max_examples: int = MAX_NUMERIC
) -> list[dspy.Example]:
    if raw is None:
        raw = _load_dataset()

    trainset, skipped = [], 0
    for item in raw:
        if item["qtype"] != "num" or item["status"] != "Resolved":
            skipped += 1
            continue

        choices = item.get("choices")
        answer = item.get("answer")
        if not choices or answer is None:
            skipped += 1
            continue

        # choices is a dict with min/max/deriv_ratio; answer is normalized [0,1]
        if not isinstance(choices, dict) or "min" not in choices or "max" not in choices:
            skipped += 1
            continue

        try:
            resolved_normalized = float(answer)
            real_value = autocast_denormalize(resolved_normalized, choices)
            min_val = float(choices["min"])
            max_val = float(choices["max"])
        except Exception:
            skipped += 1
            continue

        lower_bound_message = f"The outcome cannot be lower than {min_val}."
        upper_bound_message = f"The outcome cannot be higher than {max_val}."

        crowd_final = item["crowd"][-1] if item["crowd"] else None

        trainset.append(dspy.Example(
            question_text=item["question"] or "",
            background_info=item["background"] or "",
            resolution_criteria="",
            fine_print="",
            unit_of_measure="",
            research="",
            today_date=(item["close_time"] or "2021-01-01")[:10],
            lower_bound_message=lower_bound_message,
            upper_bound_message=upper_bound_message,
            conditional_disclaimer="",
            # metric fields
            resolved_normalized=resolved_normalized,
            real_value=real_value,
            choices_meta=choices,
            crowd_final=crowd_final,
            question_id=item["id"],
        ).with_inputs(*NUMERIC_INPUTS))

        if len(trainset) >= max_examples:
            break

    logger.info(f"Numeric trainset: {len(trainset)} (skipped={skipped})")
    return trainset


# ─────────────────────────── Serialize / Deserialize ───────────────────────────

def _save(trainset: list[dspy.Example], path: Path, extra_fields: list[str]):
    base_fields = list(BINARY_INPUTS)
    records = []
    for ex in trainset:
        rec = {f: getattr(ex, f, "") for f in base_fields}
        for f in extra_fields:
            rec[f] = getattr(ex, f, None)
        records.append(rec)
    path.write_text(json.dumps(records, ensure_ascii=False, indent=2))
    logger.info(f"Saved {len(records)} examples to {path}")


def _load(path: Path, input_fields: tuple, extra_fields: list[str]) -> list[dspy.Example]:
    records = json.loads(path.read_text())
    trainset = []
    for r in records:
        trainset.append(dspy.Example(**r).with_inputs(*input_fields))
    logger.info(f"Loaded {len(trainset)} examples from {path}")
    return trainset


def save_binary(trainset, path=BINARY_OUTPUT):
    _save(trainset, path, ["resolved_value", "crowd_final", "question_id"])

def save_mc(trainset, path=MC_OUTPUT):
    _save(trainset, path, ["options", "resolved_index", "options_list", "crowd_final", "question_id"])

def save_numeric(trainset, path=NUMERIC_OUTPUT):
    _save(trainset, path, [
        "unit_of_measure", "lower_bound_message", "upper_bound_message",
        "resolved_normalized", "real_value", "choices_meta", "crowd_final", "question_id",
    ])

def load_binary(path=BINARY_OUTPUT):
    return _load(path, BINARY_INPUTS, ["resolved_value", "crowd_final", "question_id"])

def load_mc(path=MC_OUTPUT):
    return _load(path, MC_INPUTS, ["options", "resolved_index", "options_list", "crowd_final", "question_id"])

def load_numeric(path=NUMERIC_OUTPUT):
    return _load(path, NUMERIC_INPUTS, [
        "unit_of_measure", "lower_bound_message", "upper_bound_message",
        "resolved_normalized", "real_value", "choices_meta", "crowd_final", "question_id",
    ])


# ─────────────────────────── Main ───────────────────────────

if __name__ == "__main__":
    raw = _load_dataset()

    binary = build_binary_trainset(raw, MAX_BINARY)
    save_binary(binary)

    mc = build_mc_trainset(raw, MAX_MC)
    save_mc(mc)

    numeric = build_numeric_trainset(raw, MAX_NUMERIC)
    save_numeric(numeric)

    print(f"\n✅ 训练集构建完成:")
    print(f"   Binary:   {len(binary)} 条 → {BINARY_OUTPUT}")
    print(f"   MC:       {len(mc)} 条 → {MC_OUTPUT}")
    print(f"   Numeric:  {len(numeric)} 条 → {NUMERIC_OUTPUT}")
    print("\n下一步: poetry run python optimize_forecaster.py")
