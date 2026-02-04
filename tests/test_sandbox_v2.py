import unittest
import os
import sys
import shutil

# 将项目根目录加入路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.research.lab import ResearchLab, ShadowRunner

class TestSandboxV2(unittest.TestCase):
    def setUp(self):
        self.project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        self.lab = ResearchLab(sandbox_dir="data/sandbox_test/lab")
        self.runner = ShadowRunner(project_root=self.project_root, sandbox_base="data/sandbox_test/shadow")

    def tearDown(self):
        # 清理测试数据
        if os.path.exists("data/sandbox_test"):
            try:
                shutil.rmtree("data/sandbox_test")
            except:
                pass

    def test_research_lab_isolation(self):
        print("\n=== 测试 ResearchLab 隔离性 ===")
        # 应该失败，因为 'os' 受限
        code = "import os\nprint(os.getcwd())"
        result = self.lab.run_experiment(code, "test_iso")
        print(f"结果 (应包含安全违规): {result[:50]}...")
        self.assertIn("安全违规", result)

    def test_research_lab_allowed(self):
        print("\n=== 测试 ResearchLab 允许的模块 ===")
        # 应该通过
        code = "import math\nprint(math.pi)"
        result = self.lab.run_experiment(code, "test_math")
        print(f"结果: {result}")
        self.assertIn("3.14", result)

    def test_shadow_runner_dry_run(self):
        print("\n=== 测试 ShadowRunner 空跑机制 ===")
        # 创建一个影子环境
        # 我们将尝试影子化 'core.research.lab' (模块本身)
        # 我们不需要真正修改它来测试 dry_run，只需传递空字典意味着"使用原始版本"
        # 但该方法需要写入文件。
        # 让我们编写一个模拟文件，假装它是核心的一部分
        
        modified_files = {
            "core/test_module.py": "def hello():\n    return 'Hello from Shadow'"
        }
        
        shadow_path = self.runner.create_shadow_env(modified_files)
        print(f"影子环境创建于: {shadow_path}")
        
        # 对新模块进行空跑
        success, output = self.runner.dry_run(shadow_path, "core.test_module")
        print(f"空跑输出: {output}")
        self.assertTrue(success)
        
        # 验证我们也可以导入真实模块 (因为 PYTHONPATH 包含 project_root)
        success_real, output_real = self.runner.dry_run(shadow_path, "core.research.lab")
        print(f"真实模块空跑输出: {output_real}")
        self.assertTrue(success_real)
        
        self.runner.cleanup(shadow_path)

if __name__ == '__main__':
    unittest.main()
