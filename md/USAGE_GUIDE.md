# Metaculus 预测机器人使用指南

## 快速开始

```bash
# 测试模式 - 预测 2 个示例问题
poetry run python main.py --mode test_questions

# 锦标赛模式 - 预测所有开放问题
poetry run python main.py --mode tournament

# Metaculus Cup 模式
poetry run python main.py --mode metaculus_cup
```

## 预测配置

在 `main.py` 底部修改：

```python
template_bot = SpringTemplateBot2026(
    research_reports_per_question=1,      # 研究报告数量
    predictions_per_research_report=1,    # 每个报告的预测数量
    publish_reports_to_metaculus=True,     # 是否发布到 Metaculus
    skip_previously_forecasted_questions=True,
)
```

## 模型配置

```python
llms={
    "default": GeneralLlm(
        model="gpt-4o",                       # 模型
        temperature=0.3,                       # 温度
        timeout=180,                           # 超时（秒）
        allowed_tries=2,                       # 重试次数
        api_base="https://api.wlai.vip/v1",   # API 地址
    ),
    "researcher": "tavily/news-search",        # 搜索来源
    "summarizer": GeneralLlm(...),             # 摘要模型
    "parser": GeneralLlm(...),                 # 解析模型
}
```

## 自定义测试问题

编辑 `main.py` 中的 `EXAMPLE_QUESTIONS` 列表：

```python
EXAMPLE_QUESTIONS = [
    "https://www.metaculus.com/questions/578/",
    "https://www.metaculus.com/questions/22427/",
    # 添加你的问题 URL
]
```

## DSPy 优化

```bash
# 构建训练集
poetry run python training/build_trainset.py
poetry run python training/build_metaculus_trainset.py

# 优化（Autocast 数据源）
poetry run python training/optimize_forecaster.py --type all --source autocast

# 优化（Metaculus 数据源）
poetry run python training/optimize_forecaster.py --type all --source metaculus
```

优化后的模型保存到 `json/optimized_*_forecaster.json`，运行时自动加载。

部署到 GitHub Actions 时需要将优化模型设为 Secrets，参见 [SETUP_SECRETS.md](SETUP_SECRETS.md)。

## 工具脚本

```bash
# 检查锦标赛状态
poetry run python tools/check_tournament.py
poetry run python tools/check_tournament.py 32916

# 估算成本
poetry run python tools/estimate_cost.py 10
poetry run python tools/estimate_cost.py compare
```

## 并发控制

在 `main.py` 中：

```python
_max_concurrent_questions = 1  # 同时处理的问题数量，建议 1-3
```

## 调试

```bash
# 保存完整日志
poetry run python main.py --mode test_questions 2>&1 | tee run.log
```

在 `main.py` 中开启 DEBUG 级别：

```python
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
```

## 常见问题

**Q: API 超时**
增加 timeout 到 180-300 秒。

**Q: 锦标赛返回 0 个问题**
检查锦标赛是否已开放，先用 `test_questions` 模式测试。

**Q: 成本太高**
减少 `predictions_per_research_report`，或使用更便宜的模型。

**Q: 预测质量不够**
增加 `predictions_per_research_report`，使用 `gpt-4o` 模型。
