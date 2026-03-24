#!/usr/bin/env python3
"""
成本估算工具 - 在运行前估算预测成本
用法: poetry run python estimate_cost.py [问题数量]
"""

import sys


def estimate_cost(num_questions, config):
    """估算预测成本"""

    # 模型价格（每 1M tokens）
    model_prices = {
        "gpt-4o-mini": {"input": 0.15, "output": 0.60},
        "gpt-4o": {"input": 2.50, "output": 10.00},
        "gpt-3.5-turbo": {"input": 0.50, "output": 1.50},
    }

    model = config.get("model", "gpt-4o-mini")
    research_reports = config.get("research_reports_per_question", 1)
    predictions_per_report = config.get("predictions_per_research_report", 5)

    # 估算 token 使用量（基于实际测试数据）
    tokens_per_prediction = {
        "input": 2000,   # 研究 + 问题描述 + 提示词
        "output": 500,   # 预测推理
    }

    tokens_per_research = {
        "input": 1500,   # 问题 + 新闻文章
        "output": 800,   # 研究总结
    }

    # 计算总 token 使用量
    total_predictions = num_questions * research_reports * predictions_per_report
    total_research = num_questions * research_reports

    input_tokens = (
        total_predictions * tokens_per_prediction["input"] +
        total_research * tokens_per_research["input"]
    )

    output_tokens = (
        total_predictions * tokens_per_prediction["output"] +
        total_research * tokens_per_research["output"]
    )

    # 计算成本
    prices = model_prices.get(model, model_prices["gpt-4o-mini"])
    input_cost = (input_tokens / 1_000_000) * prices["input"]
    output_cost = (output_tokens / 1_000_000) * prices["output"]
    total_cost = input_cost + output_cost

    # 估算时间（基于实际测试）
    time_per_question = 0.7  # 分钟
    total_time = num_questions * time_per_question

    return {
        "num_questions": num_questions,
        "model": model,
        "research_reports": research_reports,
        "predictions_per_report": predictions_per_report,
        "total_predictions": total_predictions,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "input_cost": input_cost,
        "output_cost": output_cost,
        "total_cost": total_cost,
        "total_time_minutes": total_time,
        "cost_per_question": total_cost / num_questions,
    }


def print_estimate(result):
    """打印估算结果"""
    print("\n" + "="*60)
    print("📊 成本估算结果")
    print("="*60 + "\n")

    print(f"配置:")
    print(f"  模型: {result['model']}")
    print(f"  问题数量: {result['num_questions']}")
    print(f"  研究报告/问题: {result['research_reports']}")
    print(f"  预测/报告: {result['predictions_per_report']}")
    print(f"  总预测数: {result['total_predictions']}")
    print()

    print(f"Token 使用:")
    print(f"  输入 tokens: {result['input_tokens']:,}")
    print(f"  输出 tokens: {result['output_tokens']:,}")
    print(f"  总计: {result['input_tokens'] + result['output_tokens']:,}")
    print()

    print(f"成本估算:")
    print(f"  输入成本: ${result['input_cost']:.4f}")
    print(f"  输出成本: ${result['output_cost']:.4f}")
    print(f"  总成本: ${result['total_cost']:.4f}")
    print(f"  平均成本/问题: ${result['cost_per_question']:.4f}")
    print()

    print(f"时间估算:")
    print(f"  总时间: {result['total_time_minutes']:.1f} 分钟")
    print(f"  平均时间/问题: {result['total_time_minutes']/result['num_questions']:.1f} 分钟")
    print()

    # 成本警告
    if result['total_cost'] > 1.0:
        print("⚠️  警告: 预计成本超过 $1.00")
    elif result['total_cost'] > 0.5:
        print("⚠️  注意: 预计成本超过 $0.50")
    else:
        print("✅ 成本在合理范围内")

    print("\n" + "="*60 + "\n")


def compare_configs():
    """比较不同配置的成本"""
    configs = [
        {
            "name": "快速测试",
            "model": "gpt-4o-mini",
            "research_reports_per_question": 1,
            "predictions_per_research_report": 3,
        },
        {
            "name": "标准使用",
            "model": "gpt-4o-mini",
            "research_reports_per_question": 1,
            "predictions_per_research_report": 5,
        },
        {
            "name": "高质量",
            "model": "gpt-4o-mini",
            "research_reports_per_question": 2,
            "predictions_per_research_report": 7,
        },
        {
            "name": "竞赛模式",
            "model": "gpt-4o",
            "research_reports_per_question": 3,
            "predictions_per_research_report": 10,
        },
    ]

    print("\n" + "="*60)
    print("📊 配置对比（10 个问题）")
    print("="*60 + "\n")

    print(f"{'配置':<12} {'模型':<15} {'成本':<10} {'时间':<10} {'质量':<8}")
    print("-" * 60)

    for config in configs:
        result = estimate_cost(10, config)
        quality = "⭐" * (configs.index(config) + 2)
        print(f"{config['name']:<12} {config['model']:<15} "
              f"${result['total_cost']:<9.4f} "
              f"{result['total_time_minutes']:<9.1f}m {quality:<8}")

    print("\n" + "="*60 + "\n")


def main():
    """主函数"""
    if len(sys.argv) > 1:
        if sys.argv[1] == "compare":
            compare_configs()
            return

        try:
            num_questions = int(sys.argv[1])
        except ValueError:
            print("❌ 错误: 请提供有效的问题数量")
            print("用法: poetry run python estimate_cost.py [问题数量]")
            print("或者: poetry run python estimate_cost.py compare")
            sys.exit(1)
    else:
        num_questions = 10  # 默认值

    # 当前配置（从 main.py 读取）
    config = {
        "model": "gpt-4o-mini",
        "research_reports_per_question": 1,
        "predictions_per_research_report": 5,
    }

    result = estimate_cost(num_questions, config)
    print_estimate(result)

    # 提供优化建议
    print("💡 优化建议:")
    print()

    if result['total_cost'] > 0.5:
        print("  降低成本:")
        print("    - 减少 predictions_per_research_report 到 3")
        print("    - 使用 gpt-3.5-turbo 模型")
        print("    - 减少 research_reports_per_question 到 1")
        print()

    print("  提高质量:")
    print("    - 增加 predictions_per_research_report 到 7")
    print("    - 使用 gpt-4o 模型")
    print("    - 增加 research_reports_per_question 到 2")
    print()

    print("  查看配置对比:")
    print("    poetry run python estimate_cost.py compare")
    print()


if __name__ == "__main__":
    main()
