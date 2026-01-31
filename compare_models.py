#!/usr/bin/env python3
"""
基座模型快速对比测试脚本
用于测试不同基座模型的基本功能和性能
"""

import asyncio
import os
import time
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

try:
    from AGI_AUTONOMOUS_CORE_V6_1_MULTI_BASE import BaseLLM, BaseModel
except ImportError:
    print("Error: Cannot import BaseLLM from AGI_AUTONOMOUS_CORE_V6_1_MULTI_BASE.py")
    sys.exit(1)


# 测试提示词
TEST_PROMPTS = {
    "code_generation": """Write a Python function to merge two sorted lists efficiently.

Include:
1. Type hints
2. Error handling
3. Example usage in docstring
4. Time complexity analysis

Return only the complete Python code.
""",

    "architecture_design": """Design the architecture for a URL shortener service.

Include:
1. Core components
2. Database schema
3. API endpoints
4. Caching strategy

Return as a structured markdown document.
""",

    "creative_task": """Generate 3 creative ideas for an AI-powered todo app that goes beyond traditional task management.

For each idea:
1. Name
2. Core concept
3. Key feature
4. Why it's innovative

Be creative and unexpected!
"""
}


async def test_model(model_type: BaseModel, test_name: str, prompt: str):
    """测试单个模型"""
    print(f"\n{'='*70}")
    print(f"Testing: {model_type.value.upper()} - {test_name}")
    print(f"{'='*70}")

    try:
        llm = BaseLLM(model_type)

        if not llm.client:
            print(f"❌ {model_type.value}: Client not initialized (check API KEY)")
            return {
                "model": model_type.value,
                "test": test_name,
                "status": "error",
                "error": "Client not initialized"
            }

        print(f"✅ {model_type.value}: Client initialized")
        print(f"   Model: {llm.model}")

        # 执行测试
        start_time = time.time()

        print(f"\n⏳ Sending prompt...")
        response = await llm.generate(
            prompt,
            max_tokens=2000,
            temperature=0.7
        )

        end_time = time.time()
        duration = end_time - start_time

        # 统计
        response_length = len(response)
        word_count = len(response.split())

        print(f"\n✅ Response received")
        print(f"   Duration: {duration:.2f}s")
        print(f"   Response length: {response_length} chars")
        print(f"   Word count: {word_count}")
        print(f"\n📄 Response preview:")
        print(f"   {response[:200]}...")

        return {
            "model": model_type.value,
            "test": test_name,
            "status": "success",
            "duration": duration,
            "response_length": response_length,
            "word_count": word_count,
            "response_preview": response[:500]
        }

    except Exception as e:
        print(f"\n❌ Error: {e}")
        return {
            "model": model_type.value,
            "test": test_name,
            "status": "error",
            "error": str(e)
        }


async def run_comparison():
    """运行所有模型对比"""
    print("\n" + "="*70)
    print("BASE MODEL COMPARISON TEST")
    print("="*70)

    # 检查可用的模型
    available_models = []
    if os.getenv("DEEPSEEK_API_KEY"):
        available_models.append(BaseModel.DEEPSEEK)
    if os.getenv("ZHIPU_API_KEY"):
        available_models.append(BaseModel.ZHIPU)
    if os.getenv("KIMI_API_KEY"):
        available_models.append(BaseModel.KIMI)
    if os.getenv("QWEN_API_KEY"):
        available_models.append(BaseModel.QWEN)
    if os.getenv("GEMINI_API_KEY"):
        available_models.append(BaseModel.GEMINI)

    if not available_models:
        print("\n❌ No API keys found!")
        print("\nPlease configure .env file with at least one of:")
        print("  - DEEPSEEK_API_KEY")
        print("  - ZHIPU_API_KEY")
        print("  - KIMI_API_KEY")
        print("  - QWEN_API_KEY")
        print("  - GEMINI_API_KEY")
        return

    print(f"\n✅ Found {len(available_models)} configured model(s):")
    for model in available_models:
        print(f"   - {model.value}")

    # 选择测试
    print("\n" + "="*70)
    print("Select test to run:")
    print("="*70)
    print("1. Code Generation (Python merge function)")
    print("2. Architecture Design (URL shortener)")
    print("3. Creative Task (AI todo app ideas)")
    print("4. All tests")

    choice = input("\nEnter choice (1-4): ").strip()

    tests_to_run = []
    if choice == "1":
        tests_to_run = [("code_generation", TEST_PROMPTS["code_generation"])]
    elif choice == "2":
        tests_to_run = [("architecture_design", TEST_PROMPTS["architecture_design"])]
    elif choice == "3":
        tests_to_run = [("creative_task", TEST_PROMPTS["creative_task"])]
    elif choice == "4":
        tests_to_run = list(TEST_PROMPTS.items())
    else:
        print("Invalid choice. Running all tests.")
        tests_to_run = list(TEST_PROMPTS.items())

    # 运行测试
    all_results = []

    for test_name, prompt in tests_to_run:
        print(f"\n\n{'#'*70}")
        print(f"# TEST: {test_name.upper()}")
        print(f"{'#'*70}")

        # 并行测试所有模型
        tasks = [
            test_model(model, test_name, prompt)
            for model in available_models
        ]

        results = await asyncio.gather(*tasks)
        all_results.extend(results)

        # 对比结果
        print(f"\n\n{'='*70}")
        print(f"RESULTS: {test_name.upper()}")
        print(f"{'='*70}")

        successful_results = [r for r in results if r["status"] == "success"]

        if successful_results:
            print(f"\n📊 Performance Comparison:")
            print(f"{'Model':<15} {'Duration':<12} {'Length':<12} {'Words':<12}")
            print(f"{'-'*60}")

            for r in sorted(successful_results, key=lambda x: x["duration"]):
                print(f"{r['model']:<15} {r['duration']:<12.2f} {r['response_length']:<12} {r['word_count']:<12}")

            # 找出最快和最慢
            fastest = min(successful_results, key=lambda x: x["duration"])
            slowest = max(successful_results, key=lambda x: x["duration"])
            longest = max(successful_results, key=lambda x: x["response_length"])

            print(f"\n🏆 Fastest: {fastest['model']} ({fastest['duration']:.2f}s)")
            print(f"🐌 Slowest: {slowest['model']} ({slowest['duration']:.2f}s)")
            print(f"📝 Longest response: {longest['model']} ({longest['response_length']} chars)")

    # 保存结果
    import json
    from datetime import datetime

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"data/model_comparison_{timestamp}.json"

    os.makedirs("data", exist_ok=True)

    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "models_tested": [m.value for m in available_models],
            "results": all_results
        }, f, indent=2, default=str)

    print(f"\n\n✅ Results saved to: {results_file}")

    # 最终总结
    print(f"\n\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")

    successful = len([r for r in all_results if r["status"] == "success"])
    failed = len([r for r in all_results if r["status"] == "error"])

    print(f"Total tests: {len(all_results)}")
    print(f"✅ Successful: {successful}")
    print(f"❌ Failed: {failed}")

    if successful > 0:
        avg_duration = sum([r["duration"] for r in all_results if r["status"] == "success"]) / successful
        print(f"⏱️  Average duration: {avg_duration:.2f}s")


if __name__ == "__main__":
    try:
        # Load .env file
        try:
            from dotenv import load_dotenv
            load_dotenv()
        except:
            pass

        asyncio.run(run_comparison())
    except KeyboardInterrupt:
        print("\n\n⚠️  Test interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
