#!/bin/bash
# 快速测试脚本 - 运行预测并显示结果

echo "=================================="
echo "Metaculus 预测机器人 - 快速测试"
echo "=================================="
echo ""

# 检查是否在正确的目录
if [ ! -f "main.py" ]; then
    echo "❌ 错误: 请在项目根目录运行此脚本"
    exit 1
fi

# 检查 .env 文件
if [ ! -f ".env" ]; then
    echo "❌ 错误: .env 文件不存在"
    exit 1
fi

# 检查 API key
if grep -q "OPENAI_API_KEY=1234567890" .env; then
    echo "⚠️  警告: OPENAI_API_KEY 仍然是占位符"
    echo "请在 .env 文件中设置真实的 API key"
    exit 1
fi

echo "✅ 环境检查通过"
echo ""

# 运行测试
echo "开始运行测试..."
echo "预计耗时: 1-2 分钟"
echo ""

poetry run python main.py --mode test_questions 2>&1 | tee test_output.log

# 检查退出码
if [ $? -eq 0 ]; then
    echo ""
    echo "=================================="
    echo "✅ 测试成功完成！"
    echo "=================================="
    echo ""

    # 提取关键信息
    echo "📊 运行统计:"
    grep -A 3 "Stats for passing reports:" test_output.log | tail -3

    echo ""
    echo "📝 完整日志已保存到: test_output.log"
    echo ""
    echo "下一步:"
    echo "  1. 查看详细日志: cat test_output.log"
    echo "  2. 运行锦标赛: poetry run python main.py --mode tournament"
    echo "  3. 查看使用指南: cat USAGE_GUIDE.md"
else
    echo ""
    echo "=================================="
    echo "❌ 测试失败"
    echo "=================================="
    echo ""
    echo "请检查日志文件: test_output.log"
    echo "常见问题:"
    echo "  - API key 是否正确"
    echo "  - 网络连接是否正常"
    echo "  - API 配额是否充足"
fi
