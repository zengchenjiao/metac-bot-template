# 项目架构

## 1. 目录结构

```
metac-bot-template/
├── main.py                          # 入口，SpringTemplateBot2026
├── main_with_no_framework.py        # 独立版本（无内部依赖）
├── quick_test.sh                    # 快速测试脚本
├── pyproject.toml / poetry.lock
├── .env / .env.template
│
├── forecaster/                      # 核心预测模块
│   ├── dspy_forecaster.py           # DSPy Signature + Hub 单例
│   ├── multi_role_forecaster.py     # 5 角色 Agent + Meta-predictor
│   ├── agent_forecaster.py          # LangGraph 迭代 Agent（备用）
│   └── tavily_searcher.py           # Tavily 搜索封装
│
├── training/                        # 训练 & 优化
│   ├── build_trainset.py            # Autocast 训练集构建
│   ├── build_metaculus_trainset.py  # Metaculus 训练集构建
│   └── optimize_forecaster.py       # DSPy BootstrapFewShot 优化
│
├── tools/                           # 工具脚本
│   ├── check_tournament.py          # 锦标赛状态检查
│   ├── estimate_cost.py             # 成本估算
│   └── metaculus_api.py             # Metaculus API 数据下载
│
├── json/                            # 训练集 & 优化模型
├── autocast/                        # Autocast 原始数据
├── integrations/                    # 第三方集成（LightningRod）
├── md/                              # 文档
└── .github/workflows/               # CI/CD
```

## 2. 主流程（多角色系统）

```
GitHub Actions (每20分钟)
        │
        ▼
    main.py  SpringTemplateBot2026
    LLM: gpt-4o (云雾 API)
    研究: Tavily API
        │
        ▼
  ForecastBot 父类 (forecasting-tools)
    for question in open_questions:
      ├─ skip if already forecasted
      ├─ run_research() → "" (空，研究在 Agent 内部)
      ├─ _run_forecast_on_*(question)
      │     │
      │     ▼
      │   检测 community prediction 是否可用
      │   (question.num_forecasters > 0 ?)
      │     │
      │     ▼
      │   run_all_role_agents() ─── 并行运行 4~5 个角色
      │     │
      │     ▼
      │   meta_predict() ─── 综合所有角色输出
      │     │
      │     ▼
      │   structure_output() → 结构化解析
      │
      ├─ aggregate predictions
      └─ submit to Metaculus API
```

## 3. 多角色 Agent 系统

```
run_all_role_agents()
  │
  ├─ Base Rate Analyst ──── 历史基准率 + 参考类预测
  │    search: "historical statistics frequency rate {question}"
  │    topic: general
  │
  ├─ News Analyst ───────── 最新新闻动态
  │    search: "latest news update 2026 {question}"
  │    topic: news
  │
  ├─ Contrarian Analyst ─── 逆向思维 + 尾部风险
  │    search: "criticism risk unlikely scenario {question}"
  │    topic: general
  │
  ├─ Community Anchor ───── 社区共识锚定（无社区预测时跳过）
  │    search: "expert forecast prediction consensus {question}"
  │    topic: general
  │
  └─ Domain Expert ──────── 领域专业知识
       search: "research study report data analysis {question}"
       topic: general

每个角色的流程 (LangGraph):
  research_node ──→ forecast_node ──→ END
  (Tavily搜索)      (DSPy ChainOfThought)

        │ 所有角色结果
        ▼
  meta_predict()
    gpt-4o 综合 4~5 个角色的推理和预测
    ├─ 有社区预测: 给 Community Anchor 额外权重
    └─ 无社区预测: 给 Base Rate + Domain Expert 额外权重
        │
        ▼
    最终预测文本
```

## 4. 问题类型路由

```
question
   │
   ├─ Binary ────→ 多角色系统 → BinaryPrediction (0.01~0.99)
   ├─ MC ─────────→ 多角色系统 → PredictedOptionList (概率分布)
   ├─ Numeric ───→ 多角色系统 → NumericDistribution (百分位)
   ├─ Date ──────→ 直接 prompt → DatePercentile (无多角色)
   └─ Conditional → 拆成 parent/child/yes/no 分别递归预测
```

## 5. DSPy 训练/优化流程（离线）

```
数据源                        构建训练集                    优化
─────                        ─────────                    ────
Autocast (HuggingFace)  →  build_trainset.py         →  optimize_forecaster.py
                              json/autocast_*_trainset     │
                                                           │  BootstrapFewShot
Metaculus API           →  build_metaculus_trainset.py     │  max_demos=4
                              json/metaculus_*_trainset     │  metric: Brier/MAE
                                                           ▼
                                                    json/optimized_*_forecaster.json
                                                    (运行时由 DSPyForecasterHub 加载)
```

## 6. 外部依赖

```
┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│  云雾 API     │  │  Tavily API  │  │ Metaculus API│
│  (OpenAI代理) │  │  (新闻搜索)   │  │ (问题/提交)  │
│              │  │              │  │              │
│ gpt-4o      │  │ advanced     │  │ 拉取问题     │
│ temp=0.3    │  │ 5条/角色     │  │ 提交预测     │
│ timeout=180s│  │ news+general │  │              │
└──────┬───────┘  └──────┬───────┘  └──────┬───────┘
       │                 │                 │
       ▼                 ▼                 ▼
  DSPy LM            TavilySearcher   MetaculusClient
  GeneralLlm         (每角色独立搜索)   (forecasting-tools)
  (forecast/meta/
   structure_output)
```

## 7. 单问题 LLM 调用统计

```
多角色系统 (Binary/MC/Numeric):
┌─────────────────────────────┬──────────┬──────────┐
│ 步骤                        │ 调用次数  │ 模型     │
├─────────────────────────────┼──────────┼──────────┤
│ Tavily 搜索 (每角色1次)      │ 4~5      │ -        │
│ DSPy CoT 预测 (每角色1次)    │ 4~5      │ gpt-4o   │
│ Meta-predictor 综合          │ 1        │ gpt-4o   │
│ _prompt_to_forecast (推理)   │ 1        │ gpt-4o   │
│ structure_output (解析+验证) │ 2        │ gpt-4o   │
├─────────────────────────────┼──────────┼──────────┤
│ 单次预测合计                 │ 12~14    │          │
└─────────────────────────────┴──────────┴──────────┘
```
