#!/usr/bin/env python3
"""
检查锦标赛状态的实用工具
用法: poetry run python check_tournament.py [tournament_id]
"""

import sys
import asyncio
from forecasting_tools import MetaculusClient


async def check_tournament(tournament_id):
    """检查指定锦标赛的状态"""
    client = MetaculusClient()

    print(f"\n{'='*60}")
    print(f"检查锦标赛 ID: {tournament_id}")
    print(f"{'='*60}\n")

    try:
        # 获取锦标赛中的问题
        questions = client.get_all_open_questions_from_tournament(tournament_id)

        print(f"✅ 找到 {len(questions)} 个开放问题\n")

        if questions:
            print("前 10 个问题:")
            print("-" * 60)
            for i, q in enumerate(questions[:10], 1):
                print(f"{i}. {q.question_text[:70]}...")
                print(f"   URL: {q.page_url}")
                print(f"   类型: {q.question_type}")
                print()
        else:
            print("⚠️  该锦标赛当前没有开放的问题")
            print("\n可能的原因:")
            print("  1. 锦标赛尚未开始")
            print("  2. 锦标赛已经结束")
            print("  3. 你的账户没有访问权限")
            print("  4. 所有问题都已关闭")

    except Exception as e:
        print(f"❌ 错误: {type(e).__name__}: {e}")
        return False

    return True


def main():
    """主函数"""
    # 默认锦标赛 ID
    default_tournaments = {
        "AIB Spring 2026": 32916,
        "AIB Fall 2025": 32813,
        "Metaculus Cup": 32828,
        "MiniBench": "minibench",
    }

    if len(sys.argv) > 1:
        # 使用命令行参数
        tournament_id = sys.argv[1]
        # 尝试转换为整数
        try:
            tournament_id = int(tournament_id)
        except ValueError:
            pass  # 保持为字符串（如 "minibench"）

        asyncio.run(check_tournament(tournament_id))
    else:
        # 检查所有默认锦标赛
        print("\n检查所有默认锦标赛...\n")

        for name, tid in default_tournaments.items():
            print(f"\n{'='*60}")
            print(f"锦标赛: {name} (ID: {tid})")
            print(f"{'='*60}")

            questions = MetaculusClient().get_all_open_questions_from_tournament(tid)
            print(f"开放问题数: {len(questions)}")

            if questions:
                print(f"示例问题: {questions[0].question_text[:60]}...")


if __name__ == "__main__":
    main()
