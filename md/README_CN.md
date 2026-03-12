# Metaculus 预测机器人 - 中文文档

这是一个基于 AI 的 Metaculus 预测机器人，使用 GPT-4o-mini 模型和 Guardian News API 进行研究和预测。

## 🎯 功能特点

- ✅ 自动从 Guardian News 获取相关新闻
- ✅ 使用 GPT-4o-mini 生成预测推理
- ✅ 支持二元、多选、数值和日期问题
- ✅ 自动发布预测到 Metaculus
- ✅ 成本追踪和性能监控
- ✅ 支持云雾 API

## 📋 前置要求

- Python 3.12+
- Poetry
- 有效的 API Keys:
  - Metaculus Token
  - OpenAI API Key (或云雾 API Key)
  - Guardian API Key

## 🚀 快速开始

### 1. 安装依赖

```bash
poetry install
```

### 2. 配置环境变量

复制 `.env.template` 到 `.env` 并填入你的 API keys:

```bash
cp .env.template .env
```

编辑 `.env`:

```env
METACULUS_TOKEN=你的_metaculus_token
OPENAI_API_KEY=你的_openai_api_key
GUARDIAN_API_KEY=你的_guardian_api_key
```

### 3. 运行测试

```bash
# 方式 1: 使用快速测试脚本
./quick_test.sh

# 方式 2: 直接运行
poetry run python main.py --mode test_questions
```

## 📊 运行模式

### 1. 测试模式（推荐用于开发）

预测 2 个示例问题：

```bash
poetry run python main.py --mode test_questions
```

### 2. 锦标赛模式

预测指定锦标赛中的所有开放问题：

```bash
poetry run python main.py --mode tournament
```

### 3. Metaculus Cup 模式

预测 Metaculus Cup 中的问题：

```bash
poetry run python main.py --mode metaculus_cup
```

## 🔧 配置说明

### 核心配置

在 `main.py` 中修改机器人配置：

```python
template_bot = SpringTemplateBot2026(
    research_reports_per_question=1,      # 每个问题的研究报告数
    predictions_per_research_report=5,    # 每个报告生成的预测数
    publish_reports_to_metaculus=True,    # 是否发布到 Metaculus
    skip_previously_forecasted_questions=True,  # 跳过已预测的问题
)
```

### 模型配置

```python
llms={
    "default": GeneralLlm(
        model="gpt-4o-mini",              # 模型名称
        temperature=0.3,                   # 温度参数
        timeout=180,                       # 超时时间（秒）
        allowed_tries=2,                   # 重试次数
        api_base="https://api.wlai.vip/v1",  # API 地址（云雾）
    ),
    "researcher": "guardian/news-search",  # 研究来源
}
```

## 📈 性能指标

当前配置的性能：

- **成本**: ~$0.01 / 问题
- **速度**: ~0.7 分钟 / 问题
- **成功率**: 100%

## 🛠️ 实用工具

### 1. 检查锦标赛状态

```bash
# 检查所有默认锦标赛
poetry run python check_tournament.py

# 检查特定锦标赛
poetry run python check_tournament.py 32916
```

### 2. 快速测试

```bash
./quick_test.sh
```

### 3. 查看使用指南

```bash
cat USAGE_GUIDE.md
```

## 📝 配置建议

### 质量 vs 成本权衡

| 场景 | research_reports | predictions_per_report | 成本 | 质量 |
|------|-----------------|----------------------|------|------|
| 快速测试 | 1 | 3 | 低 | 中 |
| 标准使用 | 1 | 5 | 中 | 高 |
| 高质量预测 | 2 | 7 | 高 | 很高 |
| 竞赛模式 | 3 | 10 | 很高 | 最高 |

### 模型选择

| 模型 | 成本 | 质量 | 速度 | 推荐场景 |
|------|------|------|------|---------|
| gpt-3.5-turbo | 低 | 中 | 快 | 快速测试 |
| gpt-4o-mini | 中 | 高 | 中 | 标准使用 |
| gpt-4o | 高 | 很高 | 慢 | 竞赛模式 |

## 🔍 研究来源选项

### 1. Guardian News API（当前使用）

```python
"researcher": "guardian/news-search"
```

- ✅ 高质量新闻来源
- ✅ 免费 API
- ⚠️ 仅限英文新闻

### 2. Smart Searcher（网络搜索）

```python
"researcher": "smart-searcher/gpt-4o-mini"
```

- ✅ 更广泛的信息来源
- ✅ 自动搜索优化
- ⚠️ 成本较高

### 3. 无研究模式

```python
"researcher": "no_research"
```

- ✅ 最快速度
- ✅ 最低成本
- ⚠️ 质量较低

## 🐛 常见问题

### Q1: API 超时错误

**问题**: `litellm.Timeout: APITimeoutError - Request timed out`

**解决方案**:
```python
timeout=180,  # 增加超时时间到 180-300 秒
```

### Q2: 锦标赛返回 0 个问题

**问题**: `Retrieved 0 questions from tournament`

**解决方案**:
1. 检查锦标赛是否已开放
2. 使用 `check_tournament.py` 验证
3. 先用 `test_questions` 模式测试

### Q3: 成本过高

**解决方案**:
- 减少 `predictions_per_research_report`
- 使用 `gpt-3.5-turbo` 模型
- 减少 `research_reports_per_question`

### Q4: 预测质量不够

**解决方案**:
- 增加 `predictions_per_research_report`
- 使用 `gpt-4o` 模型
- 增加 `research_reports_per_question`
- 使用 `smart-searcher` 研究来源

## 📚 项目结构

```
metac-bot-template/
├── main.py                 # 主程序
├── guardian_searcher.py    # Guardian API 集成
├── .env                    # 环境变量（不要提交到 git）
├── .env.template           # 环境变量模板
├── pyproject.toml          # 项目依赖
├── check_tournament.py     # 锦标赛检查工具
├── quick_test.sh           # 快速测试脚本
├── USAGE_GUIDE.md          # 详细使用指南
├── run_summary.md          # 运行结果总结
└── README_CN.md            # 中文文档（本文件）
```

## 🔐 安全注意事项

1. **不要提交 `.env` 文件到 git**
2. **定期轮换 API keys**
3. **监控 API 使用量**
4. **设置合理的超时和重试限制**

## 📊 监控和日志

### 查看详细日志

```bash
poetry run python main.py --mode test_questions 2>&1 | tee run.log
```

### 启用调试模式

在 `main.py` 中修改：

```python
logging.basicConfig(
    level=logging.DEBUG,  # 改为 DEBUG
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
```

## 🎯 最佳实践

1. **先测试后部署**
   - 始终先用 `test_questions` 模式测试
   - 验证配置和成本

2. **逐步扩展**
   - 从少量问题开始
   - 监控性能和成本
   - 根据结果调整参数

3. **定期检查**
   - 查看预测准确性
   - 监控 API 使用
   - 优化配置

4. **备份配置**
   - 保存有效的参数组合
   - 记录成本和性能数据

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

## 📄 许可证

本项目基于 Metaculus 的模板，遵循相应的开源许可证。

## 🔗 相关链接

- [Metaculus 官网](https://www.metaculus.com/)
- [Forecasting Tools 文档](https://github.com/Metaculus/forecasting-tools)
- [Guardian API 文档](https://open-platform.theguardian.com/)
- [云雾 API](https://api.wlai.vip/)

## 📞 技术支持

- **GitHub Issues**: [提交问题](https://github.com/anthropics/claude-code/issues)
- **Metaculus 社区**: [访问论坛](https://www.metaculus.com/questions/)

---

**祝你预测成功！** 🎯🚀

最后更新: 2026-03-06
