# 项目架构方案图

## 1. 系统全局视图

```
┌─────────────────────────────────────────────────────────────────────┐
│                     GitHub Actions (每20分钟)                        │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │ 1. checkout repo                                             │  │
│  │ 2. poetry install                                            │  │
│  │ 3. base64 decode → optimized_*_forecaster.json (⚠️ 路径问题)  │  │
│  │ 4. poetry run python main.py --mode tournament               │  │
│  └───────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        main.py                                      │
│                  SpringTemplateBot2026                               │
│                                                                     │
│  配置:                                                              │
│    research_reports_per_question = 1                                │
│    predictions_per_research_report = 5                              │
│    skip_previously_forecasted = True                                │
│    LLM: gpt-4o (云雾 API https://api.wlai.vip/v1)                  │
│                                                                     │
│  运行模式:                                                          │
│    tournament ──→ 锦标赛 32916 全部开放问题                          │
│    metaculus_cup → Metaculus Cup 问题                                │
│    test_questions → 2个示例问题                                      │
└─────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│              ForecastBot 父类流程 (forecasting-tools)                │
│                                                                     │
│  MetaculusClient.get_questions(tournament_id)                       │
│         │                                                           │
│         ▼                                                           │
│  for question in open_questions:                                    │
│    ├─ skip if already forecasted                                    │
│    ├─ run_research(question) × 1  ──→ 返回 "" (空)                  │
│    ├─ _run_forecast_on_*(question, research) × 5                    │
│    ├─ aggregate predictions (中位数/均值)                             │
│    └─ submit to Metaculus API                                       │
└─────────────────────────────────────────────────────────────────────┘
```

## 2. 问题类型路由

```
                    question
                       │
       ┌───────────┬───┴───────┬──────────────┐
       ▼           ▼           ▼              ▼
    Binary      MC/多选     Numeric         Date        Conditional
       │           │           │              │              │
       ▼           ▼           ▼              ▼              ▼
  ┌────────────────────────────────────┐  直接 prompt    拆成4个子问题
  │      LangGraph Agent 循环          │  (无 Agent)    parent/child/
  │  research → forecast → reflect     │  (无新闻研究)   yes/no
  │      (详见第3节)                    │               分别递归预测
  └────────────────────────────────────┘
       │           │           │              │              │
       ▼           ▼           ▼              ▼              ▼
  _binary_prompt  _mc_prompt  _numeric_prompt _date_prompt  ConditionalPrediction
  _to_forecast()  _to_forecast() _to_forecast() _to_forecast()
       │           │           │              │              │
       ▼           ▼           ▼              ▼              ▼
                structure_output → 结构化解析 → 提交 Metaculus
```

## 3. LangGraph Agent 详细流程（核心）

```
build_initial_state()
  question_text, background_info, resolution_criteria,
  fine_print, question_type, options, unit/bounds...
  research_results=[], iterations=0, confidence=0.0
         │
         ▼
┌─────────────────────────────────────────────────────────┐
│                                                         │
│  ┌─────────────────────────────────────────────┐        │
│  │         research_node                       │        │
│  │                                             │        │
│  │  搜索词 = reflection_notes 或               │        │
│  │          question_text[:150]                │        │
│  │                                             │        │
│  │  TavilySearcher.search_news()               │        │
│  │    ├─ Tavily API (advanced, news topic)     │        │
│  │    ├─ 最多 10 条结果                         │        │
│  │    └─ 每条截断 500 字符                      │        │
│  │                                             │        │
│  │  → 追加到 research_results[]                │        │
│  └──────────────────┬──────────────────────────┘        │
│                     │                                   │
│                     ▼                                   │
│  ┌─────────────────────────────────────────────┐        │
│  │         forecast_node                       │        │
│  │                                             │        │
│  │  DSPyForecasterHub.get_instance() (单例)    │        │
│  │    ├─ 首次初始化时 configure_dspy_lm()      │        │
│  │    └─ 加载 json/optimized_*_forecaster.json │        │
│  │                                             │        │
│  │  合并所有 research_results                   │        │
│  │                                             │        │
│  │  按 question_type 分发:                     │        │
│  │    binary  → BinaryForecaster               │        │
│  │    mc      → MultipleChoiceForecaster       │        │
│  │    numeric → NumericForecaster              │        │
│  │                                             │        │
│  │  每个 Forecaster 内部:                      │        │
│  │    dspy.ChainOfThought(Signature)           │        │
│  │    ├─ 注入 few-shot demos (如有优化模型)     │        │
│  │    ├─ 输入: 问题+研究+背景+标准             │        │
│  │    └─ 输出: reasoning + probability/        │        │
│  │             probabilities/percentiles       │        │
│  │                                             │        │
│  │  → prediction_text = reasoning + answer     │        │
│  └──────────────────┬──────────────────────────┘        │
│                     │                                   │
│                     ▼                                   │
│  ┌─────────────────────────────────────────────┐        │
│  │         reflect_node                        │        │
│  │                                             │        │
│  │  gpt-4o 评估预测质量                        │        │
│  │  输入: 问题 + 搜索摘要 + 预测文本            │        │
│  │  输出 JSON:                                 │        │
│  │    {                                        │        │
│  │      "confidence": 0.0~1.0,                 │        │
│  │      "missing_info": "...",                 │        │
│  │      "next_query": "下一轮搜索词"            │        │
│  │    }                                        │        │
│  │                                             │        │
│  │  iterations += 1                            │        │
│  └──────────────────┬──────────────────────────┘        │
│                     │                                   │
│                     ▼                                   │
│  ┌─────────────────────────────────────────────┐        │
│  │         should_continue (路由)               │        │
│  │                                             │        │
│  │  iterations >= 3 ──────────→ END            │        │
│  │  confidence >= 0.65 ───────→ END            │        │
│  │  confidence < 0.65 ───────→ research_node   │        │
│  │         (带 next_query)        ↑             │        │
│  │                                │             │        │
│  └────────────────────────────────┘             │        │
│                                                         │
└─────────────────────────────────────────────────────────┘
         │
         ▼ prediction_text
```

## 4. Agent 输出 → 结构化提交

```
prediction_text (DSPy 原始输出)
         │
         ▼
┌─────────────────────────────────────────────┐
│  gpt-4o (default llm)                       │
│  再次推理，生成 reasoning 文本                │
└──────────────────┬──────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────┐
│  structure_output() (parser llm)            │
│  将 reasoning 解析为结构化对象:              │
│                                             │
│  Binary → BinaryPrediction                  │
│    └─ prediction_in_decimal: 0.01~0.99      │
│                                             │
│  MC → PredictedOptionList                   │
│    └─ [{option: "A", probability: 0.4}, ...]│
│                                             │
│  Numeric → list[Percentile]                 │
│    └─ NumericDistribution.from_question()   │
│                                             │
│  验证: num_validation_samples=2             │
└──────────────────┬──────────────────────────┘
                   │
                   ▼
         ReasonedPrediction
         (prediction_value + reasoning)
                   │
                   ▼  × 5 次预测聚合
         Submit to Metaculus API
```

## 5. DSPy 训练/优化流程（离线）

```
┌──────────────────────────────────────────────────────────┐
│  build_trainset.py                                       │
│                                                          │
│  Autocast 数据集 (HuggingFace / 本地缓存)                │
│    └─ json/autocast_raw.json (90MB)                      │
│         │                                                │
│         ├─ build_binary_trainset()  → 300条              │
│         ├─ build_mc_trainset()      → 200条              │
│         └─ build_numeric_trainset() → 200条              │
│              │                                           │
│              ▼                                           │
│         json/autocast_*_trainset.json                    │
└──────────────────────────────────────────────────────────┘
                        │
                        ▼
┌──────────────────────────────────────────────────────────┐
│  optimize_forecaster.py                                  │
│                                                          │
│  对每种题型:                                              │
│    trainset[:150] → 训练                                 │
│    trainset[150:190] → 评估                              │
│                                                          │
│    BootstrapFewShotWithRandomSearch                      │
│      ├─ max_bootstrapped_demos = 4                       │
│      ├─ max_labeled_demos = 8                            │
│      ├─ num_candidate_programs = 10                      │
│      └─ metric: Brier Score (binary/mc) / MAE (numeric) │
│                                                          │
│    输出: json/optimized_*_forecaster.json                │
│    (包含 few-shot examples，运行时自动加载)               │
└──────────────────────────────────────────────────────────┘
```

## 6. 外部依赖关系

```
┌─────────────────────────────────────────────────────────┐
│                    外部 API 服务                         │
│                                                         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │  云雾 API     │  │  Tavily API  │  │ Metaculus API│  │
│  │  (OpenAI代理) │  │  (新闻搜索)   │  │ (问题/提交)  │  │
│  │              │  │              │  │              │  │
│  │ gpt-4o      │  │ advanced     │  │ 拉取问题     │  │
│  │ temp=0.3    │  │ news topic   │  │ 提交预测     │  │
│  │ timeout=180s│  │ max 10 results│ │ 发布报告     │  │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘  │
│         │                 │                 │          │
│         ▼                 ▼                 ▼          │
│  ┌─────────────────────────────────────────────────┐   │
│  │              调用方                              │   │
│  │                                                 │   │
│  │  DSPy LM (forecast_node)     ← 云雾 API        │   │
│  │  GeneralLlm (reflect_node)   ← 云雾 API        │   │
│  │  GeneralLlm (default/parser) ← 云雾 API        │   │
│  │  TavilySearcher              ← Tavily API      │   │
│  │  MetaculusClient             ← Metaculus API    │   │
│  └─────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
```

## 7. 单问题 LLM 调用统计

```
每次预测 (1 of 5):
  ┌─────────────────────────────┬──────────┬──────────┐
  │ 步骤                        │ 调用次数  │ 模型     │
  ├─────────────────────────────┼──────────┼──────────┤
  │ forecast_node (DSPy CoT)    │ 1        │ gpt-4o   │
  │ reflect_node (评估)          │ 1        │ gpt-4o   │
  │ 如低置信度再循环 (×N轮)      │ +2/轮    │ gpt-4o   │
  │ _prompt_to_forecast (推理)   │ 1        │ gpt-4o   │
  │ structure_output (解析+验证) │ 2        │ gpt-4o   │
  ├─────────────────────────────┼──────────┼──────────┤
  │ 单次预测合计                 │ 5~11     │          │
  │ × 5 次预测/问题             │ 25~55    │          │
  └─────────────────────────────┴──────────┴──────────┘
```

## 8. 已知问题 & 优化方向

```
⚠️  已知问题:
  1. GitHub Actions 中 optimized_*.json 输出到根目录
     但代码现在期望 json/ 目录 → CI 中模型加载失败
  2. Date/Conditional 问题不走 Agent，无新闻研究
  3. 每个问题 5 次预测，每次都独立创建 Agent
     → 研究不共享，重复搜索相同内容
  4. _prompt_to_forecast 对 DSPy 输出再调一次 LLM
     → 可能引入不必要的偏差和成本

🔧 可优化方向:
  1. 修复 CI 路径，或统一为根目录
  2. 让 Date/Conditional 也走 Agent 获取研究
  3. 共享研究结果，5 次预测复用同一份 research
  4. 考虑去掉 _prompt_to_forecast 中的二次推理
  5. 调整 reflect 阈值和循环次数平衡成本/质量
```
