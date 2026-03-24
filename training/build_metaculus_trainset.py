"""
从 Metaculus API 拉取 resolved 问题，用 community prediction 作为 proxy label 构建 DSPy 训练集。

为什么用 resolved 问题而不是 resolved 问题:
  Metaculus API 对 resolved 问题不返回 resolution、description、resolution_criteria 等字段。
  resolved 问题有完整字段，且 community prediction（几百人聚合预测）是很强的 proxy label。
  这也是 forecasting-tools 官方 benchmark 系统的做法。

用法: poetry run python build_metaculus_trainset.py [--max-binary 300] [--max-mc 200] [--max-numeric 200]
"""
import argparse
import asyncio
import json
import logging
from pathlib import Path
from typing import Literal

import dotenv
import dspy
import pendulum

dotenv.load_dotenv()

from forecasting_tools import MetaculusClient
from forecasting_tools.data_models.questions import (
    BinaryQuestion,
    MultipleChoiceQuestion,
    NumericQuestion,
    MetaculusQuestion,
)
from forecasting_tools.helpers.metaculus_client import ApiFilter

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# 输出路径
METACULUS_BINARY_OUTPUT = Path("json/metaculus_binary_trainset.json")
METACULUS_MC_OUTPUT = Path("json/metaculus_mc_trainset.json")
METACULUS_NUMERIC_OUTPUT = Path("json/metaculus_numeric_trainset.json")

# DSPy 输入字段定义（与 dspy_forecaster.py 的 Signature 一致）
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


# ─────────────────────────── 数据拉取 ───────────────────────────

async def fetch_resolved_questions(
    question_type: Literal["binary", "multiple_choice", "numeric"],
    max_questions: int,
    min_forecasters: int = 30,
) -> list[MetaculusQuestion]:
    """
    从 Metaculus API 拉取 resolved 问题（有完整字段 + community prediction）。
    """
    client = MetaculusClient()

    api_filter = ApiFilter(
        allowed_types=[question_type],
        allowed_statuses=["resolved"],
        num_forecasters_gte=min_forecasters,
        community_prediction_exists=True if question_type == "binary" else None,
        includes_bots_in_aggregates=False,
        group_question_mode="exclude",
        order_by="-nr_forecasters",
    )

    logger.info(f"Fetching resolved {question_type} questions (min_forecasters={min_forecasters})...")
    questions = await client.get_questions_matching_filter(
        api_filter,
        num_questions=max_questions * 2,  # 多拉一些，后面会过滤
        randomly_sample=False,
        error_if_question_target_missed=False,
    )

    logger.info(f"Fetched {len(questions)} resolved {question_type} questions")
    return questions


# ─────────────────────────── Binary 训练集构建 ───────────────────────────

def build_binary_trainset(
    questions: list[BinaryQuestion], max_examples: int
) -> list[dspy.Example]:
    """
    从 resolved BinaryQuestion 构建 DSPy 训练集。
    用 community_prediction 作为 proxy label（resolved_value）。
    """
    trainset, skipped = [], 0

    for q in questions:
        cp = q.community_prediction_at_access_time
        if cp is None:
            skipped += 1
            continue
        if not q.background_info and not q.resolution_criteria:
            skipped += 1
            continue

        # 用 community prediction 作为 proxy: >0.5 视为 Yes(1.0), <=0.5 视为 No(0.0)
        resolved_value = 1.0 if cp > 0.5 else 0.0

        today_date = pendulum.now(tz="UTC").strftime("%Y-%m-%d")

        trainset.append(dspy.Example(
            question_text=q.question_text or "",
            background_info=q.background_info or "",
            resolution_criteria=q.resolution_criteria or "",
            fine_print=q.fine_print or "",
            research="",
            today_date=today_date,
            conditional_disclaimer="",
            # Metric 字段
            resolved_value=resolved_value,
            community_prediction=cp,
            question_id=q.id_of_question,
            page_url=q.page_url,
        ).with_inputs(*BINARY_INPUTS))

        if len(trainset) >= max_examples:
            break

    yes_count = sum(1 for e in trainset if e.resolved_value == 1.0)
    logger.info(
        f"Binary trainset: {len(trainset)} examples "
        f"(yes={yes_count}, no={len(trainset)-yes_count}, skipped={skipped})"
    )
    return trainset


# ─────────────────────────── Multiple Choice 训练集构建 ───────────────────────────

def build_mc_trainset(
    questions: list[MultipleChoiceQuestion], max_examples: int
) -> list[dspy.Example]:
    """
    从 resolved MultipleChoiceQuestion 构建 DSPy 训练集。
    用 community prediction 中概率最高的选项作为 proxy label。
    """
    trainset, skipped = [], 0

    for q in questions:
        if not q.options:
            skipped += 1
            continue
        if not q.background_info and not q.resolution_criteria:
            skipped += 1
            continue

        # 从 aggregations 中提取各选项的 community prediction
        qj = (q.api_json or {}).get("question", {})
        agg = qj.get("aggregations", {})
        rw = agg.get("recency_weighted", {})
        latest = rw.get("latest")
        if not latest:
            skipped += 1
            continue

        # MC 问题的 community prediction 在 forecast_values 中
        forecast_values = latest.get("forecast_values")
        if not forecast_values or len(forecast_values) != len(q.options):
            skipped += 1
            continue

        # 找概率最高的选项作为 proxy resolved_index
        resolved_index = max(range(len(forecast_values)), key=lambda i: forecast_values[i])

        today_date = pendulum.now(tz="UTC").strftime("%Y-%m-%d")

        trainset.append(dspy.Example(
            question_text=q.question_text or "",
            options=str(q.options),
            background_info=q.background_info or "",
            resolution_criteria=q.resolution_criteria or "",
            fine_print=q.fine_print or "",
            research="",
            today_date=today_date,
            conditional_disclaimer="",
            # Metric 字段
            resolved_index=resolved_index,
            options_list=q.options,
            question_id=q.id_of_question,
            page_url=q.page_url,
        ).with_inputs(*MC_INPUTS))

        if len(trainset) >= max_examples:
            break

    logger.info(f"MC trainset: {len(trainset)} examples (skipped={skipped})")
    return trainset


# ─────────────────────────── Numeric 训练集构建 ───────────────────────────

def build_numeric_trainset(
    questions: list[NumericQuestion], max_examples: int
) -> list[dspy.Example]:
    """
    从 resolved NumericQuestion 构建 DSPy 训练集。
    用 community prediction 的 median 作为 proxy label。
    """
    trainset, skipped = [], 0

    for q in questions:
        if not q.background_info and not q.resolution_criteria:
            skipped += 1
            continue

        # 从 aggregations 中提取 community prediction 的 centers (median)
        qj = (q.api_json or {}).get("question", {})
        agg = qj.get("aggregations", {})
        rw = agg.get("recency_weighted", {})
        latest = rw.get("latest")
        if not latest:
            skipped += 1
            continue

        centers = latest.get("centers")
        if not centers or len(centers) == 0:
            skipped += 1
            continue

        # centers[0] 是 community prediction 的 median（归一化到 [0,1]）
        cp_normalized = centers[0]

        # 反归一化到实际值
        cp_real = q.lower_bound + cp_normalized * (q.upper_bound - q.lower_bound)

        # 归一化值用于 metric
        resolved_normalized = max(0.0, min(1.0, cp_normalized))

        today_date = pendulum.now(tz="UTC").strftime("%Y-%m-%d")

        lower_bound_message = (
            f"The outcome cannot be lower than {q.lower_bound} {q.unit_of_measure or ''}."
            if not q.resolved_lower_bound
            else f"The question creator thinks the number is likely not lower than {q.lower_bound} {q.unit_of_measure or ''}."
        )
        upper_bound_message = (
            f"The outcome cannot be higher than {q.upper_bound} {q.unit_of_measure or ''}."
            if not q.resolved_upper_bound
            else f"The question creator thinks the number is likely not higher than {q.upper_bound} {q.unit_of_measure or ''}."
        )

        trainset.append(dspy.Example(
            question_text=q.question_text or "",
            background_info=q.background_info or "",
            resolution_criteria=q.resolution_criteria or "",
            fine_print=q.fine_print or "",
            unit_of_measure=q.unit_of_measure or "",
            research="",
            today_date=today_date,
            lower_bound_message=lower_bound_message,
            upper_bound_message=upper_bound_message,
            conditional_disclaimer="",
            # Metric 字段
            resolved_normalized=resolved_normalized,
            real_value=cp_real,
            lower_bound=q.lower_bound,
            upper_bound=q.upper_bound,
            question_id=q.id_of_question,
            page_url=q.page_url,
        ).with_inputs(*NUMERIC_INPUTS))

        if len(trainset) >= max_examples:
            break

    logger.info(f"Numeric trainset: {len(trainset)} examples (skipped={skipped})")
    return trainset


# ─────────────────────────── 序列化 / 反序列化 ───────────────────────────

def save_trainset(trainset: list[dspy.Example], path: Path, extra_fields: list[str]):
    """保存训练集到 JSON。"""
    records = []
    for ex in trainset:
        rec = {f: getattr(ex, f, "") for f in ex._input_keys}
        for f in extra_fields:
            rec[f] = getattr(ex, f, None)
        records.append(rec)

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(records, ensure_ascii=False, indent=2))
    logger.info(f"Saved {len(records)} examples to {path}")


def load_trainset(path: Path, input_fields: tuple) -> list[dspy.Example]:
    """从 JSON 加载训练集。"""
    records = json.loads(path.read_text())
    trainset = [dspy.Example(**r).with_inputs(*input_fields) for r in records]
    logger.info(f"Loaded {len(trainset)} examples from {path}")
    return trainset


# ─────────────────────────── Main ───────────────────────────

async def main():
    parser = argparse.ArgumentParser(
        description="Build DSPy training sets from Metaculus resolved questions (community prediction as proxy label)"
    )
    parser.add_argument("--max-binary", type=int, default=300, help="Max binary questions")
    parser.add_argument("--max-mc", type=int, default=200, help="Max MC questions")
    parser.add_argument("--max-numeric", type=int, default=200, help="Max numeric questions")
    parser.add_argument(
        "--min-forecasters", type=int, default=30,
        help="Minimum number of forecasters (default: 30, ensures quality community prediction)"
    )
    args = parser.parse_args()

    logger.info(f"Building trainsets from Metaculus resolved questions (min_forecasters={args.min_forecasters})")

    # 1. Binary
    logger.info("\n" + "=" * 60)
    logger.info("Building Binary Trainset")
    logger.info("=" * 60)
    binary_questions = await fetch_resolved_questions("binary", args.max_binary, args.min_forecasters)
    binary_trainset = build_binary_trainset(binary_questions, args.max_binary)
    save_trainset(
        binary_trainset, METACULUS_BINARY_OUTPUT,
        ["resolved_value", "community_prediction", "question_id", "page_url"]
    )

    # 2. Multiple Choice
    logger.info("\n" + "=" * 60)
    logger.info("Building Multiple Choice Trainset")
    logger.info("=" * 60)
    mc_questions = await fetch_resolved_questions("multiple_choice", args.max_mc, args.min_forecasters)
    mc_trainset = build_mc_trainset(mc_questions, args.max_mc)
    save_trainset(
        mc_trainset, METACULUS_MC_OUTPUT,
        ["resolved_index", "options_list", "question_id", "page_url"]
    )

    # 3. Numeric
    logger.info("\n" + "=" * 60)
    logger.info("Building Numeric Trainset")
    logger.info("=" * 60)
    numeric_questions = await fetch_resolved_questions("numeric", args.max_numeric, args.min_forecasters)
    numeric_trainset = build_numeric_trainset(numeric_questions, args.max_numeric)
    save_trainset(
        numeric_trainset, METACULUS_NUMERIC_OUTPUT,
        ["resolved_normalized", "real_value", "lower_bound", "upper_bound", "question_id", "page_url"]
    )

    # 总结
    print("\n" + "=" * 60)
    print("Metaculus 训练集构建完成 (resolved questions + community prediction as proxy):")
    print(f"   Binary:   {len(binary_trainset)} 条 -> {METACULUS_BINARY_OUTPUT}")
    print(f"   MC:       {len(mc_trainset)} 条 -> {METACULUS_MC_OUTPUT}")
    print(f"   Numeric:  {len(numeric_trainset)} 条 -> {METACULUS_NUMERIC_OUTPUT}")
    print("\n下一步: poetry run python optimize_forecaster.py --source metaculus")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
