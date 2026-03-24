import requests
import json
import time
from datetime import datetime

BASE_URL = "https://www.metaculus.com/api"
# 强烈建议在官网 Settings -> API Access 生成一个 Token 填在这里
TOKEN = "5de9ea44e6781ddee34e350ea7d1be16ea672c0e" 

def get_open_questions(limit=10):
    """获取开放的预测问题列表"""
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36',
        'Accept': 'application/json',
    }
    if TOKEN:
        headers['Authorization'] = f'Token {TOKEN}'

    try:
        # 获取开放状态的预测问题，按预测者数量排序
        posts_url = f"{BASE_URL}/posts/?limit={limit}&status=open&order_by=-forecasts_count"
        print(f"正在获取开放的预测问题: {posts_url}\n")
        r = requests.get(posts_url, headers=headers)
        r.raise_for_status()
        posts = r.json().get('results', [])

        print(f"找到 {len(posts)} 个开放问题:\n")

        for i, post in enumerate(posts, 1):
            print(f"{i}. {post['title']}")
            print(f"   ID: {post['id']}")
            print(f"   状态: {post['status']}")
            print(f"   预测者数: {post['nr_forecasters']}")
            print(f"   预测数: {post['forecasts_count']}")
            print(f"   截止时间: {post.get('scheduled_close_time', 'N/A')}")

            # 如果是多选题组，显示子问题
            if post.get('group_of_questions') and post['group_of_questions'].get('questions'):
                questions = post['group_of_questions']['questions']
                print(f"   子问题数: {len(questions)}")
                for q in questions[:3]:
                    print(f"      - {q.get('label', q.get('title', 'N/A'))}")
            print()

    except Exception as e:
        print(f"发生错误: {e}")

def download_all_open_questions(output_file="metaculus_questions.json"):
    """下载所有比赛中的问题并保存到JSON文件"""
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36',
        'Accept': 'application/json',
    }
    if TOKEN:
        headers['Authorization'] = f'Token {TOKEN}'

    import os
    all_questions = []
    limit = 100
    request_interval = 2
    max_retries = 5

    # 断点续传：如果文件已存在，从上次中断处继续
    if os.path.exists(output_file):
        with open(output_file, 'r', encoding='utf-8') as f:
            existing = json.load(f)
        all_questions = existing.get('questions', [])
        print(f"发现已有数据，从第 {len(all_questions)} 条继续下载...")

    offset = len(all_questions)

    try:
        while True:
            posts_url = f"{BASE_URL}/posts/?limit={limit}&offset={offset}&status=open&order_by=-forecasts_count"
            print(f"正在获取第 {offset//limit + 1} 页数据 (offset={offset})...")

            # 带重试的请求
            for attempt in range(max_retries):
                r = requests.get(posts_url, headers=headers)
                if r.status_code == 429:
                    wait = 10 * (attempt + 1)
                    print(f"  触发速率限制，等待 {wait} 秒后重试 ({attempt+1}/{max_retries})...")
                    time.sleep(wait)
                    continue
                r.raise_for_status()
                break
            else:
                print("超过最大重试次数，停止下载")
                break

            data = r.json()
            results = data.get('results', [])
            if not results:
                print("没有更多数据了")
                break

            all_questions.extend(results)
            print(f"  已获取 {len(results)} 个问题，累计 {len(all_questions)} 个")

            if not data.get('next'):
                print("已到达最后一页")
                break

            offset += limit
            time.sleep(request_interval)  # 避免触发速率限制

        # 保存到JSON文件
        output_data = {
            'download_time': datetime.now().isoformat(),
            'total_count': len(all_questions),
            'questions': all_questions
        }

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)

        print(f"\n✓ 成功下载 {len(all_questions)} 个问题")
        print(f"✓ 已保存到: {output_file}")

        # 打印统计信息
        print("\n统计信息:")
        print(f"  总问题数: {len(all_questions)}")
        forecasters = [q.get('nr_forecasters', 0) for q in all_questions]
        print(f"  平均预测者数: {sum(forecasters)/len(forecasters):.1f}")

        return all_questions

    except Exception as e:
        print(f"发生错误: {e}")
        return None

if __name__ == "__main__":
    # 下载所有问题
    download_all_open_questions()

    # 或者只查看前10个问题
    # get_open_questions()