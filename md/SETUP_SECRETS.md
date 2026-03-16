# GitHub Secrets 设置指南

## 步骤1：复制 JSON 文件内容

在本地终端运行以下命令，复制三个文件的内容：

### Binary Forecaster
```bash
cat json/optimized_binary_forecaster.json | pbcopy  # macOS
# 或
cat json/optimized_binary_forecaster.json | xclip -selection clipboard  # Linux
# 或直接打开文件复制内容
```

### MC Forecaster
```bash
cat json/optimized_mc_forecaster.json | pbcopy  # macOS
```

### Numeric Forecaster
```bash
cat json/optimized_numeric_forecaster.json | pbcopy  # macOS
```

## 步骤2：在 GitHub 添加 Secrets

1. 打开你的 GitHub 仓库
2. 进入 **Settings** → **Secrets and variables** → **Actions**
3. 点击 **New repository secret**
4. 添加以下三个 secrets：

### Secret 1: OPTIMIZED_BINARY_FORECASTER
- Name: `OPTIMIZED_BINARY_FORECASTER`
- Value: 粘贴 `optimized_binary_forecaster.json` 的完整内容

### Secret 2: OPTIMIZED_MC_FORECASTER
- Name: `OPTIMIZED_MC_FORECASTER`
- Value: 粘贴 `optimized_mc_forecaster.json` 的完整内容

### Secret 3: OPTIMIZED_NUMERIC_FORECASTER
- Name: `OPTIMIZED_NUMERIC_FORECASTER`
- Value: 粘贴 `optimized_numeric_forecaster.json` 的完整内容

## 步骤3：提交工作流修改

工作流文件已经修改完成，提交并推送：

```bash
git add .github/workflows/run_bot_on_tournament.yaml
git add .github/workflows/run_bot_on_metaculus_cup.yaml
git commit -m "Add optimized forecaster setup in workflows"
git push
```

## 验证

下次工作流运行时，会在日志中看到：
```
Loaded optimized binary forecaster from optimized_binary_forecaster.json
Loaded optimized multiple_choice forecaster from optimized_mc_forecaster.json
Loaded optimized numeric forecaster from optimized_numeric_forecaster.json
```

如果看到这些日志，说明优化后的 few-shot 示例已成功加载。

## 注意事项

- GitHub Secrets 有 64KB 大小限制，这三个文件都在限制内
- Secrets 内容不会出现在日志中，保持私密
- 如果需要更新优化模型，重新运行 `optimize_forecaster.py` 后，更新对应的 Secret 即可
