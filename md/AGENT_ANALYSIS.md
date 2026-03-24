# Agent 系统分析

## 概览

项目中有 **2 套 Agent 系统** + **1 个 Meta-Predictor**：

| Agent 系统 | 文件 | 状态 | 用途 |
|-----------|------|------|------|
| **多角色 Agent 系统** | `forecaster/multi_role_forecaster.py` | ✅ 主力使用 | 5 个专业角色并行预测 + Meta-predictor 综合 |
| **迭代 Agent 系统** | `forecaster/agent_forecaster.py` | 🔄 备用 | 单 Agent 迭代搜索 + 反思循环 |
| **DSPy Forecaster Hub** | `forecaster/dspy_forecaster.py` | ✅ 核心组件 | DSPy Signature + 优化模型加载 |

---

## 1. 多角色 Agent 系统（主力）

**文件**: `forecaster/multi_role_forecaster.py`

### 1.1 系统架构

```
run_all_role_agents()
    │
    ├─ 并行运行 4~5 个角色 Agent
    │   │
    │   ├─ Base Rate Analyst ──────── 历史基准率 + 参考类预测
    │   ├─ News Analyst ──────────── 最新新闻动态
    │   ├─ Contrarian Analyst ────── 逆向思维 + 尾部风险
    │   ├─ Community Anchor ───────── 社区共识锚定（有社区预测时）
    │   └─ Domain Expert ──────────── 领域专业知识
    │
    └─ meta_predict() ──────────────── 综合所有角色输出
```

### 1.2 五个角色详解

#### 1.2.1 Base Rate Analyst（基准率分析师）

**职责**: 历史统计 + 参考类预测

**搜索策略**:
- 前缀: `"historical statistics frequency rate"`
- 主题: `general`
- 示例: `"historical statistics frequency rate Will X happen?"`

**DSPy Signature 特点**:
- 强制先识别参考类（reference class）
- 要求给出具体历史基准率百分比
- 从基准率出发，逐步调整
- 输出格式: `base rate + adjustments = final probability`

**推理步骤**:
```
(a) 识别最相关的参考类
(b) 该参考类的历史基准率是多少？（具体百分比）
(c) 哪些因素支持偏离基准率？
(d) 每个因素的调整幅度和方向
(e) 最终概率 = 基准率 + 调整总和
```

---

#### 1.2.2 News Analyst（新闻分析师）

**职责**: 最新动态 + 近期事件

**搜索策略**:
- 前缀: `"latest news update 2026"`
- 主题: `news`（Tavily 新闻专用搜索）
- 示例: `"latest news update 2026 Will X happen?"`

**DSPy Signature 特点**:
- 聚焦最近 7-30 天的新闻
- 要求引用具体文章
- 评估新闻来源可信度
- 量化每条新闻对概率的影响

**推理步骤**:
```
(a) 默认/先验预期是什么？
(b) 最近 7-30 天有哪些直接相关的新闻？引用具体文章
(c) 每条新闻如何改变概率？方向是什么？
(d) 新闻来源是否可信？排除猜测和观点
(e) 近期发展导致的净概率变化
```

---

#### 1.2.3 Contrarian Analyst（逆向分析师）

**职责**: 挑战共识 + 尾部风险

**搜索策略**:
- 前缀: `"criticism risk unlikely scenario counterargument"`
- 主题: `general`
- 示例: `"criticism risk unlikely scenario Will X happen?"`

**DSPy Signature 特点**:
- 主动寻找反对共识的证据
- 识别黑天鹅事件
- 考虑被忽视的风险
- 如果共识太自信，推回去

**推理步骤**:
```
(a) 显而易见/共识预测是什么？
(b) 反对共识的最强论据是什么？
(c) 哪些尾部风险或黑天鹅可能改变结果？
(d) 历史上有类似情况下共识错误的例子吗？
(e) 基于这些担忧，概率应该偏离共识多少？
```

---

#### 1.2.4 Community Anchor Analyst（社区锚定分析师）

**职责**: 群体智慧 + 专家共识

**搜索策略**:
- 前缀: `"expert forecast prediction consensus"`
- 主题: `general`
- 示例: `"expert forecast prediction consensus Will X happen?"`

**DSPy Signature 特点**:
- 以 Metaculus 社区预测为强基线
- 只在有强证据时才偏离
- 偏离幅度通常 5-10%
- 历史上社区预测优于大多数个人

**推理步骤**:
```
(a) 专家预测者和预测市场怎么说？给出具体数字
(b) Metaculus 社区预测是多少（如果已知）？
(c) 是否有强证据支持偏离社区共识？
(d) 如果偏离，解释原因和幅度（通常 5-10% max）
(e) 最终概率锚定在社区共识，微调
```

**特殊逻辑**:
- **当 `question.num_forecasters == 0` 时，此角色被跳过**
- 在 `run_all_role_agents()` 中动态检测 `has_community_prediction`
- 无社区预测时，系统降级为 4 角色模式

---

#### 1.2.5 Domain Expert Analyst（领域专家分析师）

**职责**: 深度技术知识 + 因果推理

**搜索策略**:
- 前缀: `"research study report data analysis"`
- 主题: `general`
- 示例: `"research study report data analysis Will X happen?"`

**DSPy Signature 特点**:
- 寻找学术论文、官方报告、政府数据
- 理解底层机制和因果因素
- 基于领域特定模型预测
- 不依赖表面新闻

**推理步骤**:
```
(a) 这个问题属于哪个领域？关键因果因素是什么？
(b) 领域特定模型或框架预测什么？
(c) 官方来源或学术研究有哪些相关数据？
(d) 领域模型中的关键不确定性是什么？
(e) 基于领域专业知识和因果分析的最终概率
```

---

### 1.3 每个角色的 LangGraph 流程

```
START
  │
  ▼
role_research_node ──── 调用 TavilySearcher
  │                     - 使用角色特定的 search_prefix
  │                     - 使用角色特定的 topic (news/general)
  │                     - max_results=5
  │                     - search_depth="advanced"
  ▼
role_forecast_node ──── 调用 DSPy ChainOfThought
  │                     - 使用角色特定的 Signature
  │                     - 输入: question + research
  │                     - 输出: reasoning + prediction
  ▼
END
```

**特点**:
- **无反思循环**（与迭代 Agent 不同）
- **单次搜索 + 单次预测**
- **并行运行 4~5 个角色**（`asyncio.gather`）

---

### 1.4 Meta-Predictor（方案 C）

**函数**: `meta_predict()`

**职责**: 综合所有角色的推理和预测，生成最终预测

**输入**:
- 所有角色的 `reasoning_text` 和 `prediction_text`
- `has_community_prediction` 标志

**LLM 配置**:
- 模型: `gpt-4o`
- 温度: `0.2`（低温，更确定性）
- 超时: `120s`

**综合策略**:
```
1. 识别角色间的共识 → 可能接近真相
2. 识别角色间的分歧 → 权衡证据质量
3. 有社区预测时:
   - 给 Community Anchor 额外权重（群体智慧是强基线）
4. 无社区预测时:
   - 给 Base Rate + Domain Expert 额外权重（作为主要锚点）
5. 对 Contrarian 保持谨慎 → 只在证据充分时采纳
6. 输出最终预测 + 2-3 句综合推理
```

**输出格式**:
- Binary: `"Probability: ZZ%"`
- MC: `"OptionName: XX%\n..."`
- Numeric: `"Percentile 10: XX\n..."`

---

## 2. 迭代 Agent 系统（备用）

**文件**: `forecaster/agent_forecaster.py`

### 2.1 系统架构

```
START
  │
  ▼
research_node ──── TavilySearcher.search_news()
  │                - max_results=10
  │                - 累积到 research_results[]
  ▼
forecast_node ──── DSPyForecasterHub.forecast_*()
  │                - 使用累积的所有研究
  │                - 输出 prediction_text
  ▼
reflect_node ───── LLM 评估预测质量
  │                - 输出 confidence (0.0-1.0)
  │                - 输出 next_query (下次搜索建议)
  ▼
should_continue?
  │
  ├─ confidence < 0.65 且 iterations < 3 ──→ 回到 research_node
  │
  └─ 否则 ──→ END
```

### 2.2 核心参数

```python
MAX_ITERATIONS = 3
CONFIDENCE_THRESHOLD = 0.65
```

### 2.3 Reflect Node 逻辑

**LLM 配置**:
- 模型: `gpt-4o`
- 温度: `0.3`
- 超时: `60s`

**输入**:
- 问题文本
- 所有搜索历史（`research_results[]`）
- 当前预测（`prediction_text`）

**输出** (JSON):
```json
{
  "confidence": 0.75,
  "missing_info": "需要更多关于 X 的数据",
  "next_query": "specific search query for missing info"
}
```

**决策逻辑**:
- `confidence >= 0.65` → 预测有充分支持，停止
- `confidence < 0.65` → 需要更多研究，继续循环
- `iterations >= 3` → 强制停止

---

### 2.4 与多角色系统的对比

| 维度 | 多角色系统 | 迭代系统 |
|------|-----------|---------|
| **Agent 数量** | 5 个并行 | 1 个迭代 |
| **搜索策略** | 每个角色差异化搜索 | 单一搜索，根据反思调整 |
| **循环次数** | 1 次（无循环） | 最多 3 次 |
| **反思机制** | 无（Meta-predictor 综合） | 有（reflect_node） |
| **并行性** | 高（5 个角色并行） | 低（串行迭代） |
| **速度** | 快（1 轮） | 慢（最多 3 轮） |
| **多样性** | 高（5 种视角） | 低（单一视角迭代） |
| **当前使用** | ✅ 主力 | 🔄 备用 |

---

## 3. DSPy Forecaster Hub（核心组件）

**文件**: `forecaster/dspy_forecaster.py`

### 3.1 职责

- **配置 DSPy LM**（云雾 API）
- **定义 DSPy Signatures**（Binary / MC / Numeric）
- **加载优化模型**（BootstrapFewShot）
- **提供统一接口**（`forecast_binary()` / `forecast_multiple_choice()` / `forecast_numeric()`）

### 3.2 Singleton 模式

```python
hub = DSPyForecasterHub.get_instance(model="gpt-4o", temperature=0.3)
```

- 全局单例，避免重复配置 LM
- 自动加载优化模型（如果存在）
- 支持 `reload()` 强制重新加载

### 3.3 优化模型路径

```python
OPTIMIZED_BINARY_PATH = "json/optimized_binary_forecaster.json"
OPTIMIZED_MC_PATH = "json/optimized_mc_forecaster.json"
OPTIMIZED_NUMERIC_PATH = "json/optimized_numeric_forecaster.json"
```

- 如果文件存在 → 加载优化的 few-shot examples
- 如果不存在 → 使用默认 Signature（zero-shot）

### 3.4 DSPy Signatures

**3 种问题类型**:
1. `BinaryForecastSignature` → 输出 `reasoning` + `probability`
2. `MultipleChoiceForecastSignature` → 输出 `reasoning` + `probabilities`
3. `NumericForecastSignature` → 输出 `reasoning` + `percentiles`

**共同特点**:
- 使用 `dspy.ChainOfThought` 包装
- 强调 step-by-step reasoning
- 输入包含: question / background / resolution_criteria / fine_print / research / today_date / conditional_disclaimer
- 输出格式严格（便于后续 `structure_output` 解析）

---

## 4. 搜索组件

**文件**: `forecaster/tavily_searcher.py`

### 4.1 TavilySearcher 类

**API**: Tavily API（新闻搜索专用）

**主要方法**:
```python
async def search_news(query: str, max_results: int = 10) -> str
```

**配置**:
- `search_depth="advanced"`
- `topic="news"` 或 `"general"`（根据角色）
- `max_results=5` 或 `10`
- `include_answer=False`

**输出格式**:
```
Article 1: Title
Published: 2026-03-20

Content (truncated to 500 chars)...

URL: https://...
---

Article 2: ...
```

---

## 5. 主流程集成

**文件**: `main.py` → `SpringTemplateBot2026`

### 5.1 Binary 问题流程

```python
async def _run_forecast_on_binary(question, research):
    # 1. 检测社区预测
    has_community = bool(question.num_forecasters)

    # 2. 运行多角色 Agent
    role_results = await run_all_role_agents(
        question_text=question.question_text,
        question_type="binary",
        has_community_prediction=has_community,
        ...
    )

    # 3. Meta-predictor 综合
    meta_output = await meta_predict(
        question_text=question.question_text,
        question_type="binary",
        role_results=role_results,
        has_community_prediction=has_community,
    )

    # 4. 解析为结构化预测
    return await self._binary_prompt_to_forecast(question, meta_output)
```

### 5.2 LLM 调用统计（单个 Binary 问题）

| 步骤 | 调用次数 | 模型 | 用途 |
|------|---------|------|------|
| Tavily 搜索 | 4~5 | - | 每个角色 1 次 |
| DSPy CoT 预测 | 4~5 | gpt-4o | 每个角色 1 次 |
| Meta-predictor | 1 | gpt-4o | 综合所有角色 |
| `_prompt_to_forecast` | 1 | gpt-4o | 推理文本生成 |
| `structure_output` | 2 | gpt-4o | 解析 + 验证 |
| **总计** | **12~14** | | |

---

## 6. 无社区预测时的降级策略

### 6.1 检测逻辑

```python
has_community = bool(question.num_forecasters)
```

- `num_forecasters > 0` → 有社区预测
- `num_forecasters == 0` → 无社区预测

### 6.2 降级行为

**在 `run_all_role_agents()` 中**:
```python
if not has_community_prediction:
    active_roles = {k: v for k, v in ROLES.items() if k != "community_anchor_analyst"}
    logger.info("[MultiRole] No community prediction available — skipping Community Anchor Analyst")
```

**结果**:
- 运行 4 个角色（跳过 Community Anchor）
- Meta-predictor 调整权重策略:
  - 给 Base Rate Analyst 和 Domain Expert 额外权重
  - 不再依赖 Community Anchor

---

## 7. 总结

### 7.1 当前使用的 Agent 系统

✅ **多角色 Agent 系统** (`multi_role_forecaster.py`)
- 5 个专业角色并行预测
- Meta-predictor 综合
- 无社区预测时自动降级为 4 角色

### 7.2 备用 Agent 系统

🔄 **迭代 Agent 系统** (`agent_forecaster.py`)
- 单 Agent 迭代搜索
- 反思循环（最多 3 次）
- 目前未在 `main.py` 中使用

### 7.3 核心组件

🔧 **DSPy Forecaster Hub** (`dspy_forecaster.py`)
- 所有 Agent 共享的预测引擎
- 支持优化模型加载
- 提供统一的 DSPy Signature

🔍 **Tavily Searcher** (`tavily_searcher.py`)
- 新闻搜索 API 封装
- 支持 news / general 主题
- 格式化输出

---

## 8. 推荐改进方向

### 8.1 短期优化

1. **动态角色选择**: 根据问题类型自动选择最相关的 3-4 个角色（而非固定 5 个）
2. **搜索结果缓存**: 相似问题共享搜索结果，减少 API 调用
3. **角色权重学习**: 根据历史表现动态调整 Meta-predictor 中的角色权重

### 8.2 长期优化

1. **混合系统**: 结合多角色并行 + 迭代反思（先并行，再迭代）
2. **自适应迭代**: 根据初始 confidence 决定是否需要迭代
3. **角色专业化**: 为不同领域（科技/政治/经济）训练专门的角色模型
