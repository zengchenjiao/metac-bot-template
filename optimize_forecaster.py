"""
使用 Autocast 训练集优化 Binary / MC / Numeric 三种 Forecaster。
用法: poetry run python optimize_forecaster.py [--type binary|mc|numeric|all]

流程:
  1. 加载本地训练集（如不存在则先运行 build_trainset.py）
  2. 初始化 DSPy LM
  3. 评估 baseline（无 few-shot）
  4. 运行 BootstrapFewShot 优化（选择 few-shot demos）
  5. 评估 optimized（有 few-shot）
  6. 打印两轮对比评分
"""
import argparse
import logging
import re
import os
from pathlib import Path

import dspy
import dotenv

from build_trainset import (
    BINARY_OUTPUT, MC_OUTPUT, NUMERIC_OUTPUT,
    load_binary, load_mc, load_numeric,
    build_binary_trainset, build_mc_trainset, build_numeric_trainset,
    save_binary, save_mc, save_numeric,
    _load_dataset,
    autocast_normalize,
)
from dspy_forecaster import (
    BinaryForecaster, MultipleChoiceForecaster, NumericForecaster,
    configure_dspy_lm,
    binary_metric, mc_metric, numeric_metric,
    optimize_forecaster,
    OPTIMIZED_BINARY_PATH, OPTIMIZED_MC_PATH, OPTIMIZED_NUMERIC_PATH,
    DSPyForecasterHub,
)

dotenv.load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

TRAIN_SIZE = 150   # 用于优化的样本数（每种题型）
EVAL_SIZE  = 40    # 用于评估的样本数（每种题型）


# ─────────────────────────── Eval helpers ───────────────────────────

def _extract_binary_prob(prediction) -> float | None:
    match = re.search(r"(\d+(?:\.\d+)?)\s*%", str(prediction.probability))
    return float(match.group(1)) / 100.0 if match else None


def _extract_mc_probs(prediction, options: list) -> list[float]:
    prob_text = str(prediction.probabilities)
    probs = []
    total = 0.0
    for opt in options:
        pattern = re.escape(str(opt)) + r"[:\s]+(\d+(?:\.\d+)?)"
        m = re.search(pattern, prob_text, re.IGNORECASE)
        p = float(m.group(1)) if m else 0.0
        if p > 1.0:
            p /= 100.0
        probs.append(p)
        total += p
    if total > 0:
        probs = [p / total for p in probs]
    else:
        n = len(options)
        probs = [1.0 / n] * n
    return probs


def _extract_numeric_median(prediction) -> float | None:
    text = str(prediction.percentiles)
    matches = re.findall(r"Percentile\s+(\d+)\s*:\s*([\d.eE+\-]+)", text)
    if not matches:
        return None
    pmap = {int(p): float(v) for p, v in matches}
    if 50 in pmap:
        return pmap[50]
    if 40 in pmap and 60 in pmap:
        return (pmap[40] + pmap[60]) / 2.0
    vals = sorted(pmap.values())
    return vals[len(vals) // 2]


def evaluate_binary(module, evalset) -> dict:
    briers, correct, errors = [], 0, 0
    for ex in evalset:
        try:
            pred = module(
                question_text=ex.question_text,
                background_info=ex.background_info,
                resolution_criteria=ex.resolution_criteria,
                fine_print=ex.fine_print,
                research=ex.research,
                today_date=ex.today_date,
                conditional_disclaimer=ex.conditional_disclaimer,
            )
            prob = _extract_binary_prob(pred)
            if prob is None:
                errors += 1
                continue
            actual = ex.resolved_value
            briers.append((prob - actual) ** 2)
            if (prob >= 0.5) == (actual >= 0.5):
                correct += 1
        except Exception as e:
            logger.warning(f"Binary eval error [{ex.question_id}]: {e}")
            errors += 1
    n = len(briers)
    return {"n": n, "errors": errors,
            "avg_brier": sum(briers) / n if n else None,
            "accuracy": correct / n if n else None}


def evaluate_mc(module, evalset) -> dict:
    briers, correct, errors = [], 0, 0
    for ex in evalset:
        try:
            options = ex.options_list if hasattr(ex, "options_list") else []
            pred = module(
                question_text=ex.question_text,
                options=ex.options,
                background_info=ex.background_info,
                resolution_criteria=ex.resolution_criteria,
                fine_print=ex.fine_print,
                research=ex.research,
                today_date=ex.today_date,
                conditional_disclaimer=ex.conditional_disclaimer,
            )
            probs = _extract_mc_probs(pred, options)
            idx = int(ex.resolved_index)
            brier = sum((probs[i] - (1.0 if i == idx else 0.0)) ** 2
                        for i in range(len(options))) / 2.0
            briers.append(brier)
            if probs.index(max(probs)) == idx:
                correct += 1
        except Exception as e:
            logger.warning(f"MC eval error [{ex.question_id}]: {e}")
            errors += 1
    n = len(briers)
    return {"n": n, "errors": errors,
            "avg_brier": sum(briers) / n if n else None,
            "accuracy": correct / n if n else None}


def evaluate_numeric(module, evalset) -> dict:
    maes, errors = [], 0
    for ex in evalset:
        try:
            pred = module(
                question_text=ex.question_text,
                background_info=ex.background_info,
                resolution_criteria=ex.resolution_criteria,
                fine_print=ex.fine_print,
                unit_of_measure=ex.unit_of_measure,
                research=ex.research,
                today_date=ex.today_date,
                lower_bound_message=ex.lower_bound_message,
                upper_bound_message=ex.upper_bound_message,
                conditional_disclaimer=ex.conditional_disclaimer,
            )
            median = _extract_numeric_median(pred)
            if median is None:
                errors += 1
                continue
            choices = ex.choices_meta if hasattr(ex, "choices_meta") else {}
            median_normalized = autocast_normalize(median, choices) if choices else max(0.0, min(1.0, median))
            maes.append(abs(median_normalized - ex.resolved_normalized))
        except Exception as e:
            logger.warning(f"Numeric eval error [{ex.question_id}]: {e}")
            errors += 1
    n = len(maes)
    return {"n": n, "errors": errors,
            "avg_mae": sum(maes) / n if n else None}


# ─────────────────────────── Per-type optimization ───────────────────────────

def _load_data(output_path, load_fn, build_fn, raw):
    if output_path.exists():
        full = load_fn()
    else:
        full = build_fn(raw, TRAIN_SIZE + EVAL_SIZE)
    return full[:TRAIN_SIZE], full[TRAIN_SIZE:TRAIN_SIZE + EVAL_SIZE]


def run_binary(raw=None):
    print("\n" + "=" * 55)
    print("  Binary Forecaster")
    print("=" * 55)

    trainset, evalset = _load_data(BINARY_OUTPUT, load_binary, build_binary_trainset, raw)
    logger.info(f"Binary — train: {len(trainset)}, eval: {len(evalset)}")

    # Round 1: baseline (no few-shot)
    print("\n  [Round 1] Baseline (no few-shot)")
    baseline = evaluate_binary(BinaryForecaster(), evalset)

    # Round 2: optimized (with few-shot)
    print("  [Round 2] Optimized (with few-shot)")
    optimized = optimize_forecaster(BinaryForecaster(), trainset, binary_metric)
    result = evaluate_binary(optimized, evalset)

    optimized.save(OPTIMIZED_BINARY_PATH)
    _print_comparison("Binary", baseline, result, metric="brier")
    return baseline, result


def run_mc(raw=None):
    print("\n" + "=" * 55)
    print("  MultipleChoice Forecaster")
    print("=" * 55)

    trainset, evalset = _load_data(MC_OUTPUT, load_mc, build_mc_trainset, raw)
    logger.info(f"MC — train: {len(trainset)}, eval: {len(evalset)}")

    print("\n  [Round 1] Baseline (no few-shot)")
    baseline = evaluate_mc(MultipleChoiceForecaster(), evalset)

    print("  [Round 2] Optimized (with few-shot)")
    optimized = optimize_forecaster(MultipleChoiceForecaster(), trainset, mc_metric)
    result = evaluate_mc(optimized, evalset)

    optimized.save(OPTIMIZED_MC_PATH)
    _print_comparison("MC", baseline, result, metric="brier")
    return baseline, result


def run_numeric(raw=None):
    print("\n" + "=" * 55)
    print("  Numeric Forecaster")
    print("=" * 55)

    trainset, evalset = _load_data(NUMERIC_OUTPUT, load_numeric, build_numeric_trainset, raw)
    logger.info(f"Numeric — train: {len(trainset)}, eval: {len(evalset)}")

    print("\n  [Round 1] Baseline (no few-shot)")
    baseline = evaluate_numeric(NumericForecaster(), evalset)

    print("  [Round 2] Optimized (with few-shot)")
    optimized = optimize_forecaster(NumericForecaster(), trainset, numeric_metric)
    result = evaluate_numeric(optimized, evalset)

    optimized.save(OPTIMIZED_NUMERIC_PATH)
    _print_comparison("Numeric", baseline, result, metric="mae")
    return baseline, result


# ─────────────────────────── Print helpers ───────────────────────────

def _print_comparison(name: str, baseline: dict, result: dict, metric: str):
    print(f"\n{'─'*55}")
    print(f"  {name} — Baseline vs Optimized")
    print(f"{'─'*55}")
    print(f"  {'':22} {'Baseline':>10}  {'Optimized':>10}  {'Delta':>10}")
    print(f"  {'─'*22} {'─'*10}  {'─'*10}  {'─'*10}")

    if metric == "brier":
        b = baseline["avg_brier"]
        r = result["avg_brier"]
        ab = baseline.get("accuracy")
        ar = result.get("accuracy")
        if b is not None and r is not None:
            print(f"  {'Brier Score (↓)':<22} {b:>10.4f}  {r:>10.4f}  {r-b:>+10.4f}")
        if ab is not None and ar is not None:
            print(f"  {'Accuracy (↑)':<22} {ab:>9.2%}  {ar:>9.2%}  {ar-ab:>+9.2%}")
    else:
        b = baseline["avg_mae"]
        r = result["avg_mae"]
        if b is not None and r is not None:
            print(f"  {'MAE (↓)':<22} {b:>10.4f}  {r:>10.4f}  {r-b:>+10.4f}")

    print(f"  {'Eval samples':<22} {baseline['n']:>10}  {result['n']:>10}")
    print(f"  {'Errors':<22} {baseline['errors']:>10}  {result['errors']:>10}")

    if b is not None and r is not None:
        if r < b:
            print(f"\n  ✅ 优化有效，{metric.upper()} 降低 {b-r:.4f}")
        elif r == b:
            print(f"\n  ➖ 无变化")
        else:
            print(f"\n  ⚠️  优化后指标略有上升，few-shot demos 可能不适合当前评估集")


# ─────────────────────────── Main ───────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Optimize DSPy forecasters with Autocast dataset")
    parser.add_argument(
        "--type", choices=["binary", "mc", "numeric", "all"], default="all",
        help="Which forecaster to optimize (default: all)"
    )
    args = parser.parse_args()

    configure_dspy_lm(model="gpt-4o-mini", temperature=0.3)

    need_download = (
        (args.type in ("binary", "all") and not BINARY_OUTPUT.exists()) or
        (args.type in ("mc", "all")     and not MC_OUTPUT.exists()) or
        (args.type in ("numeric", "all") and not NUMERIC_OUTPUT.exists())
    )
    raw = _load_dataset() if need_download else None

    if args.type in ("binary", "all"):
        run_binary(raw)

    if args.type in ("mc", "all"):
        run_mc(raw)

    if args.type in ("numeric", "all"):
        run_numeric(raw)

    print("\n" + "=" * 55)
    print("  优化完成，模型已保存：")
    for path in [OPTIMIZED_BINARY_PATH, OPTIMIZED_MC_PATH, OPTIMIZED_NUMERIC_PATH]:
        if os.path.exists(path):
            print(f"  ✅ {path}")
    print("=" * 55 + "\n")

    DSPyForecasterHub.reload()
    logger.info("DSPyForecasterHub reloaded with optimized models.")


if __name__ == "__main__":
    main()
