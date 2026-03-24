"""
使用 Autocast 或 Metaculus 训练集优化 Binary / MC / Numeric 三种 Forecaster。
用法:
  poetry run python optimize_forecaster.py [--type binary|mc|numeric|all] [--source autocast|metaculus]

流程:
  1. 加载本地训练集（如不存在则先运行对应的 build_*_trainset.py）
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

from training.build_trainset import (
    BINARY_OUTPUT as AUTOCAST_BINARY_OUTPUT,
    MC_OUTPUT as AUTOCAST_MC_OUTPUT,
    NUMERIC_OUTPUT as AUTOCAST_NUMERIC_OUTPUT,
    load_binary as load_autocast_binary,
    load_mc as load_autocast_mc,
    load_numeric as load_autocast_numeric,
    build_binary_trainset as build_autocast_binary,
    build_mc_trainset as build_autocast_mc,
    build_numeric_trainset as build_autocast_numeric,
    save_binary, save_mc, save_numeric,
    _load_dataset,
    autocast_normalize,
)
from training.build_metaculus_trainset import (
    METACULUS_BINARY_OUTPUT, METACULUS_MC_OUTPUT, METACULUS_NUMERIC_OUTPUT,
    BINARY_INPUTS, MC_INPUTS, NUMERIC_INPUTS,
    load_trainset as load_metaculus_trainset,
)
from forecaster.dspy_forecaster import (
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


def evaluate_numeric(module, evalset, source: str = "autocast") -> dict:
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

            if source == "metaculus":
                # Metaculus: 直接用 lower_bound/upper_bound 线性归一化
                lb = float(ex.lower_bound)
                ub = float(ex.upper_bound)
                if ub <= lb:
                    errors += 1
                    continue
                median_normalized = max(0.0, min(1.0, (median - lb) / (ub - lb)))
            else:
                # Autocast: 用 log-scale 归一化
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

def _load_data_autocast(output_path, load_fn, build_fn, raw):
    if output_path.exists():
        full = load_fn()
    else:
        full = build_fn(raw, TRAIN_SIZE + EVAL_SIZE)
    return full[:TRAIN_SIZE], full[TRAIN_SIZE:TRAIN_SIZE + EVAL_SIZE]


def _load_data_metaculus(output_path, input_fields):
    if not output_path.exists():
        raise FileNotFoundError(
            f"Metaculus trainset not found at {output_path}. "
            f"Run `poetry run python build_metaculus_trainset.py` first."
        )
    full = load_metaculus_trainset(output_path, input_fields)
    return full[:TRAIN_SIZE], full[TRAIN_SIZE:TRAIN_SIZE + EVAL_SIZE]


def run_binary(source: str = "autocast", raw=None):
    print("\n" + "=" * 55)
    print(f"  Binary Forecaster (source: {source})")
    print("=" * 55)

    if source == "metaculus":
        trainset, evalset = _load_data_metaculus(METACULUS_BINARY_OUTPUT, BINARY_INPUTS)
    else:
        trainset, evalset = _load_data_autocast(AUTOCAST_BINARY_OUTPUT, load_autocast_binary, build_autocast_binary, raw)
    logger.info(f"Binary — train: {len(trainset)}, eval: {len(evalset)}")

    print("\n  [Round 1] Baseline (no few-shot)")
    baseline = evaluate_binary(BinaryForecaster(), evalset)

    print("  [Round 2] Optimized (with few-shot)")
    optimized = optimize_forecaster(BinaryForecaster(), trainset, binary_metric)
    result = evaluate_binary(optimized, evalset)

    optimized.save(OPTIMIZED_BINARY_PATH)
    _print_comparison("Binary", baseline, result, metric="brier")
    return baseline, result


def run_mc(source: str = "autocast", raw=None):
    print("\n" + "=" * 55)
    print(f"  MultipleChoice Forecaster (source: {source})")
    print("=" * 55)

    if source == "metaculus":
        trainset, evalset = _load_data_metaculus(METACULUS_MC_OUTPUT, MC_INPUTS)
    else:
        trainset, evalset = _load_data_autocast(AUTOCAST_MC_OUTPUT, load_autocast_mc, build_autocast_mc, raw)
    logger.info(f"MC — train: {len(trainset)}, eval: {len(evalset)}")

    print("\n  [Round 1] Baseline (no few-shot)")
    baseline = evaluate_mc(MultipleChoiceForecaster(), evalset)

    print("  [Round 2] Optimized (with few-shot)")
    optimized = optimize_forecaster(MultipleChoiceForecaster(), trainset, mc_metric)
    result = evaluate_mc(optimized, evalset)

    optimized.save(OPTIMIZED_MC_PATH)
    _print_comparison("MC", baseline, result, metric="brier")
    return baseline, result


def run_numeric(source: str = "autocast", raw=None):
    print("\n" + "=" * 55)
    print(f"  Numeric Forecaster (source: {source})")
    print("=" * 55)

    if source == "metaculus":
        trainset, evalset = _load_data_metaculus(METACULUS_NUMERIC_OUTPUT, NUMERIC_INPUTS)
    else:
        trainset, evalset = _load_data_autocast(AUTOCAST_NUMERIC_OUTPUT, load_autocast_numeric, build_autocast_numeric, raw)
    logger.info(f"Numeric — train: {len(trainset)}, eval: {len(evalset)}")

    print("\n  [Round 1] Baseline (no few-shot)")
    baseline = evaluate_numeric(NumericForecaster(), evalset, source)

    print("  [Round 2] Optimized (with few-shot)")
    optimized = optimize_forecaster(NumericForecaster(), trainset, numeric_metric)
    result = evaluate_numeric(optimized, evalset, source)

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
    parser = argparse.ArgumentParser(description="Optimize DSPy forecasters")
    parser.add_argument(
        "--type", choices=["binary", "mc", "numeric", "all"], default="all",
        help="Which forecaster to optimize (default: all)"
    )
    parser.add_argument(
        "--source", choices=["autocast", "metaculus"], default="metaculus",
        help="Training data source (default: metaculus)"
    )
    args = parser.parse_args()

    configure_dspy_lm(model="gpt-4o-mini", temperature=0.3)

    if args.source == "autocast":
        need_download = (
            (args.type in ("binary", "all") and not AUTOCAST_BINARY_OUTPUT.exists()) or
            (args.type in ("mc", "all")     and not AUTOCAST_MC_OUTPUT.exists()) or
            (args.type in ("numeric", "all") and not AUTOCAST_NUMERIC_OUTPUT.exists())
        )
        raw = _load_dataset() if need_download else None
    else:
        raw = None

    if args.type in ("binary", "all"):
        run_binary(args.source, raw)

    if args.type in ("mc", "all"):
        run_mc(args.source, raw)

    if args.type in ("numeric", "all"):
        run_numeric(args.source, raw)

    print("\n" + "=" * 55)
    print(f"  优化完成 (source: {args.source})，模型已保存：")
    for path in [OPTIMIZED_BINARY_PATH, OPTIMIZED_MC_PATH, OPTIMIZED_NUMERIC_PATH]:
        if os.path.exists(path):
            print(f"  ✅ {path}")
    print("=" * 55 + "\n")

    DSPyForecasterHub.reload()
    logger.info("DSPyForecasterHub reloaded with optimized models.")


if __name__ == "__main__":
    main()
