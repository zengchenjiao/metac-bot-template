# Metaculus 预测机器人运行报告

## 运行时间
- 开始时间: 2026-03-06 03:12:56
- 结束时间: 2026-03-06 03:14:24
- 总耗时: 约 1.5 分钟

## 运行结果

### ✅ 成功预测 2 个问题

#### 1. 问题 578: 人类会在 2100 年前灭绝吗？
- **URL**: https://www.metaculus.com/questions/578
- **预测概率**: 15%
- **状态**: ✅ 成功
- **错误数**: 0

#### 2. 问题 22427: 2030 年前有多少新 AI 实验室会成为领先实验室？
- **URL**: https://www.metaculus.com/questions/22427
- **预测分布**:
  - 0 或 1 个: 10%
  - 2 或 3 个: 40% (最可能)
  - 4 或 5 个: 25%
  - 6 或 7 个: 15%
  - 8 或 9 个: 7%
  - 10 个或更多: 3%
- **状态**: ✅ 成功
- **错误数**: 0

## 性能指标

- **总成本**: $0.02306
- **平均每个问题成本**: $0.01153
- **平均每个问题耗时**: 0.72 分钟
- **成功率**: 100% (2/2)

## 配置信息

### API 配置
- **OpenAI API**: 云雾 API (https://api.wlai.vip/v1)
- **模型**: gpt-4o-mini
- **超时时间**: 180 秒
- **重试次数**: 2 次

### 研究配置
- **研究报告数/问题**: 1
- **预测数/研究报告**: 5
- **研究来源**: Guardian News API

## 关键修改

1. ✅ 配置云雾 API 地址: `https://api.wlai.vip/v1`
2. ✅ 增加超时时间: 40秒 → 180秒
3. ✅ 为所有模型统一配置 (default, summarizer, parser)
4. ✅ 添加云雾 API Key

## 下一步建议

### 1. 在真实锦标赛上运行
```bash
poetry run python main.py --mode tournament
```

### 2. 调整预测参数
编辑 main.py 第 664-665 行:
```python
research_reports_per_question=1,  # 增加可提高准确性
predictions_per_research_report=5,  # 增加可提高稳定性
```

### 3. 添加更多测试问题
编辑 main.py 第 703-706 行的 EXAMPLE_QUESTIONS 列表

### 4. 优化成本
- 当前成本: ~$0.01/问题
- 可以通过减少 predictions_per_research_report 降低成本
- 或使用更便宜的模型

## 注意事项

⚠️ **网络追踪警告**: 
- 出现了一些 "Network is unreachable" 警告
- 这些是 OpenAI 追踪功能的非致命错误
- 不影响核心预测功能

✅ **运行状态**: 完全正常，所有预测成功完成
