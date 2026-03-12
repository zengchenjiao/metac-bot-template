"""
使用 Autocast 训练集优化 Binary / MC / Numeric 三种 Forecaster。
用法: poetry run python optimize_forecaster.py [--type binary|mc|numeric|all]

流程:
  1. 加载本地训练集（如不存在则先运行 build_trainset.py）
  2. 初始化 DSPy LM
  3. 对每种题型运行 BootstrapFewShot 优化
  4. 保存优化后的模型
  5. 打印优化前后的指标对比
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
            # Normalize model output (real units) to [0,1] before comparing
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

def run_binary(raw=None):
    print("\n" + "=" * 55)
    print("  Binary Forecaster 优化")
    print("=" * 55)

    if BINARY_OUTPUT.exists():
        full = load_binary()
    else:
        full = build_binary_trainset(raw, TRAIN_SIZE + EVAL_SIZE)
        save_binary(full)

    trainset = full[:TRAIN_SIZE]
    evalset  = full[TRAIN_SIZE:TRAIN_SIZE + EVAL_SIZE]
    logger.info(f"Binary — train: {len(trainset)}, eval: {len(evalset)}")

    baseline = evaluate_binary(BinaryForecaster(), evalset)
    logger.info(f"Baseline → Brier: {baseline['avg_brier']:.4f}, Acc: {baseline['accuracy']:.2%}")

    optimized = optimize_forecaster(BinaryForecaster(), trainset, binary_metric, max_bootstrapped_demos=4)
    result    = evaluate_binary(optimized, evalset)
    logger.info(f"Optimized → Brier: {result['avg_brier']:.4f}, Acc: {result['accuracy']:.2%}")

    optimized.save(OPTIMIZED_BINARY_PATH)
    _print_comparison("Binary", baseline, result, metric="brier")
    return baseline, result


def run_mc(raw=None):
    print("\n" + "=" * 55)
    print("  MultipleChoice Forecaster 优化")
    print("=" * 55)

    if MC_OUTPUT.exists():
        full = load_mc()
    else:
        full = build_mc_trainset(raw, TRAIN_SIZE + EVAL_SIZE)
        save_mc(full)

    trainset = full[:TRAIN_SIZE]
    evalset  = full[TRAIN_SIZE:TRAIN_SIZE + EVAL_SIZE]
    logger.info(f"MC — train: {len(trainset)}, eval: {len(evalset)}")

    baseline  = evaluate_mc(MultipleChoiceForecaster(), evalset)
    logger.info(f"Baseline → Brier: {baseline['avg_brier']:.4f}, Acc: {baseline['accuracy']:.2%}")

    optimized = optimize_forecaster(MultipleChoiceForecaster(), trainset, mc_metric, max_bootstrapped_demos=4)
    result    = evaluate_mc(optimized, evalset)
    logger.info(f"Optimized → Brier: {result['avg_brier']:.4f}, Acc: {result['accuracy']:.2%}")

    optimized.save(OPTIMIZED_MC_PATH)
    _print_comparison("MC", baseline, result, metric="brier")
    return baseline, result


def run_numeric(raw=None):
    print("\n" + "=" * 55)
    print("  Numeric Forecaster 优化")
    print("=" * 55)

    if NUMERIC_OUTPUT.exists():
        full = load_numeric()
    else:
        full = build_numeric_trainset(raw, TRAIN_SIZE + EVAL_SIZE)
        save_numeric(full)

    trainset = full[:TRAIN_SIZE]
    evalset  = full[TRAIN_SIZE:TRAIN_SIZE + EVAL_SIZE]
    logger.info(f"Numeric — train: {len(trainset)}, eval: {len(evalset)}")

    baseline  = evaluate_numeric(NumericForecaster(), evalset)
    logger.info(f"Baseline → MAE: {baseline['avg_mae']:.4f}")

    optimized = optimize_forecaster(NumericForecaster(), trainset, numeric_metric, max_bootstrapped_demos=4)
    result    = evaluate_numeric(optimized, evalset)
    logger.info(f"Optimized → MAE: {result['avg_mae']:.4f}")

    optimized.save(OPTIMIZED_NUMERIC_PATH)
    _print_comparison("Numeric", baseline, result, metric="mae")
    return baseline, result


# ─────────────────────────── Print helpers ───────────────────────────

def _print_comparison(name: str, baseline: dict, result: dict, metric: str):
    print(f"\n{'─'*55}")
    print(f"  {name} 优化结果")
    print(f"{'─'*55}")
    if metric == "brier":
        b = baseline["avg_brier"]
        r = result["avg_brier"]
        ab = baseline.get("accuracy")
        ar = result.get("accuracy")
        print(f"  {'Brier Score (↓)':<22} {b:>8.4f}  →  {r:>8.4f}  ({r-b:>+.4f})")
        if ab is not None:
            print(f"  {'Accuracy (↑)':<22} {ab:>8.2%}  →  {ar:>8.2%}  ({ar-ab:>+.2%})")
        if r < b:
            print(f"  ✅ 优化成功，Brier 降低 {b-r:.4f}")
        else:
            print(f"  ⚠️  未见明显提升")
    else:  # mae
        b = baseline["avg_mae"]
        r = result["avg_mae"]
        print(f"  {'MAE (↓)':<22} {b:>8.4f}  →  {r:>8.4f}  ({r-b:>+.4f})")
        if r < b:
            print(f"  ✅ 优化成功，MAE 降低 {b-r:.4f}")
        else:
            print(f"  ⚠️  未见明显提升")
    print(f"  Errors: {result['errors']}")


# ─────────────────────────── Main ───────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Optimize DSPy forecasters with Autocast dataset")
    parser.add_argument(
        "--type", choices=["binary", "mc", "numeric", "all"], default="all",
        help="Which forecaster to optimize (default: all)"
    )
    args = parser.parse_args()

    configure_dspy_lm(model="gpt-4o", temperature=0.3)

    # 只下载一次数据集，三种题型共用
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
    print("  所有优化完成，模型已保存：")
    for path in [OPTIMIZED_BINARY_PATH, OPTIMIZED_MC_PATH, OPTIMIZED_NUMERIC_PATH]:
        if os.path.exists(path):
            print(f"  ✅ {path}")
    print("\n  重启 bot 后将自动加载优化后的模型。")
    print("=" * 55 + "\n")

    DSPyForecasterHub.reload()
    logger.info("DSPyForecasterHub reloaded with optimized models.")


if __name__ == "__main__":
    main()
