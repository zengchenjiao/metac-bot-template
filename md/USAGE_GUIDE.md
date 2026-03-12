# Metaculus 预测机器人使用指南

## 快速开始

### 1. 测试模式（推荐用于开发）
```bash
# 预测 2 个示例问题
poetry run python main.py --mode test_questions
```

### 2. 锦标赛模式
```bash
# 预测锦标赛中的所有开放问题
poetry run python main.py --mode tournament
```

### 3. Metaculus Cup 模式
```bash
# 预测 Metaculus Cup 中的问题
poetry run python main.py --mode metaculus_cup
```

## 高级配置

### 调整预测质量 vs 成本

在 main.py 第 664-665 行修改：

```python
template_bot = SpringTemplateBot2026(
    research_reports_per_question=1,      # 研究报告数量
    predictions_per_research_report=5,    # 每个报告的预测数量
    ...
)
```

**配置建议**：

| 场景 | research_reports | predictions_per_report | 成本 | 质量 |
|------|-----------------|----------------------|------|------|
| 快速测试 | 1 | 3 | 低 | 中 |
| 标准使用 | 1 | 5 | 中 | 高 |
| 高质量预测 | 2 | 7 | 高 | 很高 |
| 竞赛模式 | 3 | 10 | 很高 | 最高 |

### 更换模型

#### 选项 1: 使用更强大的模型
```python
"default": GeneralLlm(
    model="gpt-4o",  # 更强大但更贵
    temperature=0.3,
    timeout=180,
    allowed_tries=2,
    api_base="https://api.wlai.vip/v1",
),
```

#### 选项 2: 使用更便宜的模型
```python
"default": GeneralLlm(
    model="gpt-3.5-turbo",  # 更便宜但质量稍低
    temperature=0.3,
    timeout=180,
    allowed_tries=2,
    api_base="https://api.wlai.vip/v1",
),
```

### 自定义测试问题

编辑 main.py 第 703-706 行：

```python
EXAMPLE_QUESTIONS = [
    "https://www.metaculus.com/questions/578/",
    "https://www.metaculus.com/questions/22427/",
    # 添加你自己的问题 URL
    "https://www.metaculus.com/questions/YOUR_QUESTION_ID/",
]
```

## 研究来源配置

### 当前使用: Guardian News API
```python
"researcher": "guardian/news-search",
```

### 其他选项:

#### 1. Smart Searcher (网络搜索)
```python
"researcher": "smart-searcher/gpt-4o-mini",
```

#### 2. 不使用研究（仅基于问题描述）
```python
"researcher": "no_research",
```

#### 3. 使用 LLM 直接研究
```python
"researcher": GeneralLlm(
    model="gpt-4o-mini",
    temperature=0.3,
    timeout=180,
    api_base="https://api.wlai.vip/v1",
),
```

## 发布设置

### 自动发布到 Metaculus
```python
publish_reports_to_metaculus=True,  # 自动发布预测
```

### 仅保存到本地
```python
publish_reports_to_metaculus=False,  # 不发布
folder_to_save_reports_to="./forecasts",  # 保存到本地文件夹
```

## 性能优化

### 1. 并发控制

在 main.py 第 116-119 行：

```python
_max_concurrent_questions = 1  # 同时处理的问题数量
```

- 增加可以加快速度，但会增加 API 负载
- 建议值: 1-3

### 2. 跳过已预测的问题

```python
skip_previously_forecasted_questions=True,  # 跳过已预测的问题
```

### 3. 超时设置

```python
timeout=180,  # API 超时时间（秒）
```

- 云雾 API 建议: 180 秒
- OpenAI 官方 API: 可以设置为 60-120 秒

## 监控和调试

### 查看详细日志
```bash
poetry run python main.py --mode test_questions 2>&1 | tee run.log
```

### 查看成本统计
运行结束后会显示：
```
Total cost estimated: $X.XXXX
Average cost per question: $X.XXXX
Average time spent per question: X.XX minutes
```

### 调试模式

在 main.py 第 635 行修改日志级别：

```python
logging.basicConfig(
    level=logging.DEBUG,  # 改为 DEBUG 查看更多信息
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
```

## 常见问题

### Q1: API 超时怎么办？
**A**: 增加 timeout 值到 180-300 秒

### Q2: 成本太高怎么办？
**A**: 
- 减少 predictions_per_research_report
- 使用更便宜的模型（gpt-3.5-turbo）
- 减少 research_reports_per_question

### Q3: 预测质量不够好？
**A**:
- 增加 predictions_per_research_report
- 使用更强大的模型（gpt-4o）
- 增加 research_reports_per_question
- 使用 smart-searcher 获取更多信息

### Q4: 锦标赛返回 0 个问题？
**A**:
- 检查锦标赛是否已开放
- 确认你的账户有权限访问
- 使用 test_questions 模式测试

### Q5: 如何查看预测是否已发布到 Metaculus？
**A**: 访问你的 Metaculus 个人主页查看预测历史

## 最佳实践

1. **先用 test_questions 测试**
   - 确保配置正确
   - 检查成本和质量

2. **逐步增加规模**
   - 从少量问题开始
   - 监控成本和性能
   - 调整参数

3. **定期检查结果**
   - 查看预测准确性
   - 根据反馈调整策略

4. **备份配置**
   - 保存 .env 文件
   - 记录有效的参数组合

5. **监控 API 使用**
   - 注意 API 配额
   - 避免超出速率限制

## 示例工作流

### 开发阶段
```bash
# 1. 测试配置
poetry run python main.py --mode test_questions

# 2. 检查结果
cat run_summary.md

# 3. 调整参数（如果需要）
# 编辑 main.py

# 4. 重新测试
poetry run python main.py --mode test_questions
```

### 生产阶段
```bash
# 1. 在锦标赛上运行
poetry run python main.py --mode tournament

# 2. 监控运行
tail -f run.log

# 3. 检查发布状态
# 访问 Metaculus 网站确认
```

## 技术支持

- **GitHub Issues**: https://github.com/anthropics/claude-code/issues
- **Metaculus 文档**: https://www.metaculus.com/help/
- **Forecasting Tools**: https://github.com/Metaculus/forecasting-tools

---

**祝你预测成功！** 🎯
