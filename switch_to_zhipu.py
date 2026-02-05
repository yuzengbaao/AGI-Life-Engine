#!/usr/bin/env python3
"""
快速切换到智谱GLM-4.7
"""

import sys
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

print("=" * 80)
print("切换到智谱GLM-4.7")
print("=" * 80)

# 检查环境变量
import os
from dotenv import load_dotenv

load_dotenv()

zhipu_key = os.getenv('ZHIPU_API_KEY')
zhipu_model = os.getenv('ZHIPU_MODEL', 'glm-4')

print(f"\n当前配置:")
print(f"  ZHIPU_API_KEY: {'✅ 已配置' if zhipu_key else '❌ 未配置'}")
print(f"  ZHIPU_MODEL: {zhipu_model}")

if not zhipu_key:
    print("\n❌ 错误: ZHIPU_API_KEY 未配置")
    print("\n请在 .env 文件中配置:")
    print("  ZHIPU_API_KEY=your_api_key_here")
    sys.exit(1)

# 推荐使用GLM-4.7
print(f"\n推荐模型: glm-4.7 (最新旗舰)")
print(f"  当前模型: {zhipu_model}")

if zhipu_model != 'glm-4.7':
    print(f"\n建议更新 .env:")
    print(f"  ZHIPU_MODEL=glm-4.7")

# 测试连接
print("\n测试智谱GLM连接...")
try:
    from AGI_AUTONOMOUS_CORE_V6_2 import ZhipuLLM
    import asyncio

    async def test():
        llm = ZhipuLLM()
        result = await llm.generate(
            "Hello! 请简单回复：智谱GLM-4.7已就绪。",
            max_tokens=100
        )
        print(f"\n✅ 连接成功!")
        print(f"  模型回复: {result[:100]}...")
        return True

    success = asyncio.run(test())
    
    if success:
        print("\n" + "=" * 80)
        print("✅ 智谱GLM配置成功!")
        print("=" * 80)
        print("\n下一步:")
        print("  1. 修改 test_multi_file_v2.py")
        print("  2. 将 DeepSeekLLM() 改为 ZhipuLLM()")
        print("  3. 运行: python test_multi_file_v2.py")
        print("\n预期效果:")
        print("  - 32K token输出限制")
        print("  - 一次生成600-800行完整模块")
        print("  - 无截断问题")
        
except Exception as e:
    print(f"\n❌ 连接失败: {e}")
    print("\n请检查:")
    print("  1. API Key是否正确")
    print("  2. 是否安装了 zhipuai: pip install zhipuai")
    sys.exit(1)
