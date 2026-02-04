#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
批量回归测试：对多个真实模块运行自修改能力的完整流程测试
测试流程: propose → sandbox → apply → rollback
"""

import sys
import os
import time
import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.self_modifying_engine import SelfModifyingEngine


class BatchRegressionTester:
    """批量回归测试器"""
    
    # 选择适合测试的真实模块（中小型、独立性强）
    TARGET_MODULES = [
        "core/math_utils.py",
        "core/event_bus.py",
        "core/checkpoint.py",
        "core/logger.py",
        "core/monitor.py",
        "core/metrics_tracker.py",
        "core/entropy_regulator.py",
        "core/motivation.py",
        "core/insight_utilities.py",
        "core/topology_tools.py",
    ]
    
    def __init__(self):
        self.engine = SelfModifyingEngine()
        self.results: List[Dict[str, Any]] = []
        self.start_time = datetime.now()
        
    def run_single_module_test(self, module_path: str) -> Dict[str, Any]:
        """对单个模块运行完整回归测试"""
        result = {
            "module": module_path,
            "propose": {"success": False, "error": None, "patch_lines": 0},
            "sandbox": {"success": False, "error": None},
            "apply": {"success": False, "error": None},
            "rollback": {"success": False, "error": None},
            "overall": "PENDING",
            "duration_ms": 0,
        }
        
        full_path = PROJECT_ROOT / module_path
        if not full_path.exists():
            result["overall"] = "SKIP_NOT_FOUND"
            return result
            
        start = time.time()
        
        try:
            # Step 1: Propose (无LLM)
            # target_module格式: "core.math_utils" (点分隔模块名)
            module_name = module_path.replace("/", ".").replace("\\", ".").replace(".py", "")
            print(f"  [1/4] Propose patch for {module_name}...")
            patch = self.engine.propose_patch(
                target_module=module_name,
                issue_description="[BatchTest] 添加性能监控装饰器或增强日志",
                use_llm=False,
                patch_strategy="auto"
            )
            
            if patch and hasattr(patch, 'modified_code') and patch.modified_code:
                result["propose"]["success"] = True
                # 计算diff行数
                from difflib import unified_diff
                diff = list(unified_diff(
                    patch.original_code.splitlines(keepends=True),
                    patch.modified_code.splitlines(keepends=True)
                ))
                result["propose"]["patch_lines"] = len(diff)
            else:
                result["propose"]["error"] = "No patch generated"
                result["overall"] = "FAIL_PROPOSE"
                result["duration_ms"] = int((time.time() - start) * 1000)
                return result
                
        except Exception as e:
            result["propose"]["error"] = str(e)
            result["overall"] = "FAIL_PROPOSE"
            result["duration_ms"] = int((time.time() - start) * 1000)
            return result
            
        try:
            # Step 2: Sandbox test
            # sandbox_test 返回 Tuple[bool, Dict]
            print(f"  [2/4] Sandbox test...")
            sandbox_result = self.engine.sandbox_test(patch)
            if isinstance(sandbox_result, tuple):
                sandbox_ok, sandbox_details = sandbox_result
            else:
                sandbox_ok = sandbox_result
            result["sandbox"]["success"] = sandbox_ok
            
            if not sandbox_ok:
                result["sandbox"]["error"] = "Sandbox validation failed"
                result["overall"] = "FAIL_SANDBOX"
                result["duration_ms"] = int((time.time() - start) * 1000)
                return result
                
        except Exception as e:
            result["sandbox"]["error"] = str(e)
            result["overall"] = "FAIL_SANDBOX"
            result["duration_ms"] = int((time.time() - start) * 1000)
            return result
            
        try:
            # Step 3: Apply patch
            # apply_or_reject 接受 CodePatch，返回 ModificationRecord
            print(f"  [3/4] Apply patch...")
            mod_record = self.engine.apply_or_reject(patch, force_apply=True)
            
            if mod_record and hasattr(mod_record, 'status'):
                from core.self_modifying_engine import ModificationStatus
                result["apply"]["success"] = (mod_record.status == ModificationStatus.APPLIED)
                record_id = mod_record.id
            else:
                result["apply"]["error"] = "apply_or_reject returned invalid result"
                result["overall"] = "FAIL_APPLY"
                result["duration_ms"] = int((time.time() - start) * 1000)
                return result
                
            if not result["apply"]["success"]:
                result["apply"]["error"] = f"Status: {mod_record.status.value if hasattr(mod_record, 'status') else 'unknown'}"
                result["overall"] = "FAIL_APPLY"
                result["duration_ms"] = int((time.time() - start) * 1000)
                return result
                
        except Exception as e:
            result["apply"]["error"] = str(e)
            result["overall"] = "FAIL_APPLY"
            result["duration_ms"] = int((time.time() - start) * 1000)
            return result
            
        try:
            # Step 4: Rollback
            print(f"  [4/4] Rollback...")
            rollback_ok = self.engine.rollback(record_id)
            result["rollback"]["success"] = rollback_ok
            
            if not rollback_ok:
                result["rollback"]["error"] = "Rollback returned False"
                result["overall"] = "WARN_ROLLBACK_FAILED"
            else:
                result["overall"] = "PASS"
                
        except Exception as e:
            result["rollback"]["error"] = str(e)
            result["overall"] = "WARN_ROLLBACK_FAILED"
            
        result["duration_ms"] = int((time.time() - start) * 1000)
        return result
        
    def run_all_tests(self) -> Dict[str, Any]:
        """运行所有模块的批量测试"""
        print("=" * 60)
        print("AGI Self-Modification Batch Regression Test")
        print("=" * 60)
        print(f"测试模块数: {len(self.TARGET_MODULES)}")
        print(f"开始时间: {self.start_time.isoformat()}")
        print("-" * 60)
        
        for i, module in enumerate(self.TARGET_MODULES, 1):
            print(f"\n[{i}/{len(self.TARGET_MODULES)}] Testing: {module}")
            result = self.run_single_module_test(module)
            self.results.append(result)
            
            status_icon = "✅" if result["overall"] == "PASS" else (
                "⚠️" if "WARN" in result["overall"] else "❌"
            )
            print(f"  Result: {status_icon} {result['overall']} ({result['duration_ms']}ms)")
            
        return self.generate_summary()
        
    def generate_summary(self) -> Dict[str, Any]:
        """生成测试摘要"""
        total = len(self.results)
        passed = sum(1 for r in self.results if r["overall"] == "PASS")
        warned = sum(1 for r in self.results if "WARN" in r["overall"])
        failed = sum(1 for r in self.results if "FAIL" in r["overall"])
        skipped = sum(1 for r in self.results if "SKIP" in r["overall"])
        
        total_duration = sum(r["duration_ms"] for r in self.results)
        
        summary = {
            "timestamp": datetime.now().isoformat(),
            "total_modules": total,
            "passed": passed,
            "warned": warned,
            "failed": failed,
            "skipped": skipped,
            "pass_rate": f"{passed/total*100:.1f}%" if total > 0 else "N/A",
            "total_duration_ms": total_duration,
            "avg_duration_ms": total_duration // total if total > 0 else 0,
            "details": self.results,
        }
        
        print("\n" + "=" * 60)
        print("BATCH REGRESSION TEST SUMMARY")
        print("=" * 60)
        print(f"Total:   {total} modules")
        print(f"Passed:  {passed} ✅")
        print(f"Warned:  {warned} ⚠️")
        print(f"Failed:  {failed} ❌")
        print(f"Skipped: {skipped} ⏭️")
        print(f"Pass Rate: {summary['pass_rate']}")
        print(f"Total Duration: {total_duration}ms")
        print("=" * 60)
        
        # 输出失败详情
        if failed > 0:
            print("\nFailed Modules Detail:")
            for r in self.results:
                if "FAIL" in r["overall"]:
                    print(f"  - {r['module']}: {r['overall']}")
                    for stage in ["propose", "sandbox", "apply", "rollback"]:
                        if r[stage].get("error"):
                            print(f"      {stage}: {r[stage]['error']}")
                            
        return summary


def main():
    """主入口"""
    tester = BatchRegressionTester()
    summary = tester.run_all_tests()
    
    # 保存结果到JSON
    output_path = PROJECT_ROOT / "core" / "batch_regression_results.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\n结果已保存: {output_path}")
    
    # 返回退出码
    if summary["failed"] > 0:
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
