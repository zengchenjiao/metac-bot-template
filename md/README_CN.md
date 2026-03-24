# Metaculus 预测机器人 - 中文文档

基于 AI 的 Metaculus 预测锦标赛机器人，使用多角色 Agent 系统 + DSPy 优化 + Tavily 新闻搜索。

## 功能特点

- 5 角色并行预测（Base Rate / News / Contrarian / Community Anchor / Domain Expert）
- Meta-predictor 综合多角色输出，生成最终预测
- DSPy BootstrapFewShot 优化，自动选择 few-shot 示例
- Tavily API 新闻搜索，按角色差异化搜索策略
- 无社区预测时自动降级为 4 角色模式
- 支持 Binary / MC / Numeric / Date / Conditional 问题
- GitHub Actions 每 20 分钟自动运行

## 前置要求

- Python 3.11+
- Poetry
- API Keys:
  - Metaculus Token
  - OpenAI API Key（或云雾 API Key）
  - Tavily API Key

## 快速开始

### 1. 安装依赖

```bash
poetry install
```

### 2. 配置环境变量

```bash
cp .env.template .env
```

编辑 `.env`：

```env
METACULUS_TOKEN=你的_metaculus_token
OPENAI_API_KEY=你的_openai_api_key
TAVILY_API_KEY=你的_tavily_api_key
```

### 3. 运行测试

```bash
# 快速测试脚本
./quick_test.sh

# 或直接运行
poetry run python main.py --mode test_questions
```

## 运行模式

```bash
# 测试模式 - 预测 2 个示例问题
poetry run python main.py --mode test_questions

# 锦标赛模式 - 预测锦标赛中所有开放问题
poetry run python main.py --mode tournament

# Metaculus Cup 模式
poetry run python main.py --mode metaculus_cup
```

## 项目结构

```
metac-bot-template/
├── main.py                          # 入口
├── main_with_no_framework.py        # 独立版本
├── quick_test.sh
│
├── forecaster/                      # 核心预测模块
│   ├── dspy_forecaster.py           # DSPy Signature + Hub
│   ├── multi_role_forecaster.py     # 多角色 Agent + Meta-predictor
│   ├── agent_forecaster.py          # LangGraph 迭代 Agent
│   └── tavily_searcher.py           # Tavily 搜索封装
│
├── training/                        # 训练 & 优化
│   ├── build_trainset.py            # Autocast 训练集
│   ├── build_metaculus_trainset.py  # Metaculus 训练集
│   └── optimize_forecaster.py       # DSPy 优化
│
├── tools/                           # 工具脚本
│   ├── check_tournament.py          # 锦标赛检查
│   ├── estimate_cost.py             # 成本估算
│   └── metaculus_api.py             # Metaculus 数据下载
│
├── json/                            # 训练集 & 优化模型
├── autocast/                        # Autocast 原始数据
├── integrations/                    # 第三方集成
└── md/                              # 文档
```

## 核心配置

在 `main.py` 中修改：

```python
template_bot = SpringTemplateBot2026(
    research_reports_per_question=1,
    predictions_per_research_report=1,
    publish_reports_to_metaculus=True,
    skip_previously_forecasted_questions=True,
    llms={
        "default": GeneralLlm(
            model="gpt-4o",
            temperature=0.3,
            timeout=180,
            api_base="https://api.wlai.vip/v1",
        ),
        "researcher": "tavily/news-search",
        ...
    },
)
```

## DSPy 优化

```bash
# 使用 Autocast 数据优化
poetry run python training/optimize_forecaster.py --type all --source autocast

# 使用 Metaculus 数据优化
poetry run python training/optimize_forecaster.py --type all --source metaculus
```

优化后的模型保存到 `json/optimized_*_forecaster.json`，运行时自动加载。

## 工具脚本

```bash
# 检查锦标赛状态
poetry run python tools/check_tournament.py

# 估算成本
poetry run python tools/estimate_cost.py 10
```

## 常见问题

**API 超时** — 增加 timeout 到 180-300 秒

**锦标赛返回 0 个问题** — 检查锦标赛是否已开放，先用 test_questions 模式测试

**成本控制** — 减少 predictions_per_research_report，或使用更便宜的模型

## 相关链接

- [Metaculus](https://www.metaculus.com/)
- [forecasting-tools](https://github.com/Metaculus/forecasting-tools)
- [Tavily API](https://tavily.com/)
- [DSPy](https://github.com/stanfordnlp/dspy)

最后更新: 2026-03-20
