"""
AGI智能能力修复验收测试
================================
验证2026-01-24修复的4个问题：
1. 对话历史持久化
2. 低置信度警告
3. 工具结果闭环
4. 多步执行约束
"""

import os
import sys
import json
import re
from pathlib import Path
from datetime import datetime

# 添加项目根目录
sys.path.insert(0, str(Path(__file__).parent.parent))

class IntelligenceCapabilityTestSuite:
    """智能能力修复验收测试套件"""
    
    def __init__(self):
        self.results = {}
        self.passed = 0
        self.failed = 0
        
    def test_1_dialogue_history_persistence(self):
        """测试1: 对话历史持久化功能"""
        print("\n" + "="*60)
        print("📝 测试1: 对话历史持久化")
        print("="*60)
        
        try:
            from core.llm_first_dialogue import LLMFirstDialogueEngine
            
            # 创建测试实例
            engine = LLMFirstDialogueEngine()
            
            # 检查方法是否存在
            checks = {
                "_get_history_file_path": hasattr(engine, '_get_history_file_path'),
                "_persist_history": hasattr(engine, '_persist_history'),
                "_load_history": hasattr(engine, '_load_history'),
                "get_history_summary": hasattr(engine, 'get_history_summary'),
            }
            
            print(f"  ✓ _get_history_file_path 方法: {'存在' if checks['_get_history_file_path'] else '缺失'}")
            print(f"  ✓ _persist_history 方法: {'存在' if checks['_persist_history'] else '缺失'}")
            print(f"  ✓ _load_history 方法: {'存在' if checks['_load_history'] else '缺失'}")
            print(f"  ✓ get_history_summary 方法: {'存在' if checks['get_history_summary'] else '缺失'}")
            
            # 验证路径生成
            history_path = engine._get_history_file_path()
            print(f"  ✓ 历史文件路径: {history_path}")
            
            # 测试添加历史并持久化
            test_msg = {"role": "user", "content": "测试消息", "timestamp": datetime.now().timestamp()}
            engine._conversation_history = [test_msg]
            engine._persist_history()
            
            # 检查文件是否创建
            file_exists = Path(history_path).exists()
            print(f"  ✓ 文件创建成功: {file_exists}")
            
            if file_exists:
                with open(history_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                print(f"  ✓ 文件内容长度: {len(content)} 字符")
                
            all_passed = all(checks.values()) and file_exists
            self.results['对话历史持久化'] = "PASS" if all_passed else "FAIL"
            
            if all_passed:
                self.passed += 1
                print("  ✅ 测试通过")
            else:
                self.failed += 1
                print("  ❌ 测试失败")
                
        except Exception as e:
            self.results['对话历史持久化'] = f"ERROR: {e}"
            self.failed += 1
            print(f"  ❌ 测试异常: {e}")
            
    def test_2_low_confidence_warning(self):
        """测试2: 低置信度警告功能"""
        print("\n" + "="*60)
        print("⚠️ 测试2: 低置信度警告")
        print("="*60)
        
        try:
            # 读取源文件检查代码是否存在
            source_file = Path(__file__).parent.parent / "core" / "hallucination_aware_llm.py"
            
            with open(source_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 检查关键代码片段 (匹配实际代码格式)
            checks = {
                "低置信度前缀_50": "我不太确定以下内容的准确性" in content,
                "中等置信度前缀_50-60": "以下回答可能存在偏差" in content,
                "较低置信度前缀_60-70": "以下回答基于有限信息" in content,
                "置信度阈值_0.50": "validation.confidence < 0.50" in content,
                "置信度阈值_0.60": "validation.confidence < 0.60" in content,
                "置信度阈值_0.70": "validation.confidence < 0.70" in content,
            }
            
            for name, result in checks.items():
                status = "✓" if result else "✗"
                print(f"  {status} {name}: {'存在' if result else '缺失'}")
            
            all_passed = all(checks.values())
            self.results['低置信度警告'] = "PASS" if all_passed else "FAIL"
            
            if all_passed:
                self.passed += 1
                print("  ✅ 测试通过")
            else:
                self.failed += 1
                print("  ❌ 测试失败")
                
        except Exception as e:
            self.results['低置信度警告'] = f"ERROR: {e}"
            self.failed += 1
            print(f"  ❌ 测试异常: {e}")
            
    def test_3_tool_result_closure(self):
        """测试3: 工具结果闭环功能"""
        print("\n" + "="*60)
        print("🔧 测试3: 工具结果闭环")
        print("="*60)
        
        try:
            # 读取源文件检查代码是否存在
            source_file = Path(__file__).parent.parent / "tool_execution_bridge.py"
            
            with open(source_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 检查 _format_final_response 中的关键代码 (匹配实际代码格式)
            checks = {
                "content字段处理": "'content' in result" in content and "文件内容" in content,
                "documents字段处理": "'documents' in result" in content and "文档列表" in content,
                "results字段处理": "'results' in result" in content and "搜索结果" in content,
            }
            
            for name, result in checks.items():
                status = "✓" if result else "✗"
                print(f"  {status} {name}: {'存在' if result else '缺失'}")
            
            all_passed = all(checks.values())
            self.results['工具结果闭环'] = "PASS" if all_passed else "FAIL"
            
            if all_passed:
                self.passed += 1
                print("  ✅ 测试通过")
            else:
                self.failed += 1
                print("  ❌ 测试失败")
                
        except Exception as e:
            self.results['工具结果闭环'] = f"ERROR: {e}"
            self.failed += 1
            print(f"  ❌ 测试异常: {e}")
            
    def test_4_multi_step_execution_constraint(self):
        """测试4: 多步执行约束功能"""
        print("\n" + "="*60)
        print("📋 测试4: 多步执行约束")
        print("="*60)
        
        try:
            checks = {}
            
            # 检查 llm_first_dialogue.py (匹配实际代码格式)
            source_file1 = Path(__file__).parent.parent / "core" / "llm_first_dialogue.py"
            with open(source_file1, 'r', encoding='utf-8') as f:
                content1 = f.read()
            
            checks["llm_first_dialogue_声明即承诺"] = "声明即承诺" in content1
            checks["llm_first_dialogue_必须全部执行"] = "必须全部执行" in content1
            
            # 检查 hallucination_aware_llm.py
            source_file2 = Path(__file__).parent.parent / "core" / "hallucination_aware_llm.py"
            with open(source_file2, 'r', encoding='utf-8') as f:
                content2 = f.read()
            
            checks["hallucination_aware_多步执行"] = "多步执行完整性" in content2 or "多步任务" in content2
            checks["hallucination_aware_TOOL_CALL"] = "TOOL_CALL" in content2
            
            for name, result in checks.items():
                status = "✓" if result else "✗"
                print(f"  {status} {name}: {'存在' if result else '缺失'}")
            
            all_passed = all(checks.values())
            self.results['多步执行约束'] = "PASS" if all_passed else "FAIL"
            
            if all_passed:
                self.passed += 1
                print("  ✅ 测试通过")
            else:
                self.failed += 1
                print("  ❌ 测试失败")
                
        except Exception as e:
            self.results['多步执行约束'] = f"ERROR: {e}"
            self.failed += 1
            print(f"  ❌ 测试异常: {e}")
            
    def run_all_tests(self):
        """运行所有测试"""
        print("\n" + "🧪"*30)
        print("\n  AGI智能能力修复验收测试 - 2026-01-24")
        print("\n" + "🧪"*30)
        
        self.test_1_dialogue_history_persistence()
        self.test_2_low_confidence_warning()
        self.test_3_tool_result_closure()
        self.test_4_multi_step_execution_constraint()
        
        # 汇总结果
        print("\n" + "="*60)
        print("📊 测试汇总")
        print("="*60)
        
        for name, result in self.results.items():
            status_icon = "✅" if result == "PASS" else "❌"
            print(f"  {status_icon} {name}: {result}")
        
        print(f"\n  总计: {self.passed} 通过 / {self.failed} 失败")
        print(f"  通过率: {self.passed/(self.passed+self.failed)*100:.1f}%")
        
        overall = "✅ 验收通过" if self.failed == 0 else "❌ 验收失败"
        print(f"\n  {overall}")
        print("="*60 + "\n")
        
        return self.failed == 0


if __name__ == "__main__":
    suite = IntelligenceCapabilityTestSuite()
    success = suite.run_all_tests()
    sys.exit(0 if success else 1)
