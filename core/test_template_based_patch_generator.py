import unittest
from core.template_based_patch_generator import TemplateBasedPatchGenerator

class TestTemplateBasedPatchGenerator(unittest.TestCase):
    def setUp(self):
        self.generator = TemplateBasedPatchGenerator()
        self.example_code = """
def foo():
    pass
"""
        self.template = {
            "desc": "将pass替换为raise NotImplementedError()",
            "code": "def foo():\n    raise NotImplementedError()\n"
        }
        self.generator.add_template(self.template)

    def test_template_patch(self):
        patch = self.generator.generate_patch(self.example_code, "将pass替换为raise NotImplementedError()", strategy="template")
        self.assertIn("raise NotImplementedError", patch)

    def test_ast_diff_patch(self):
        patch = self.generator.generate_patch(self.example_code, "", strategy="ast_diff")
        self.assertIn("raise NotImplementedError", patch)

    def test_genetic_patch(self):
        code = """
def a():
    return 1
def b():
    return 2
"""
        patch = self.generator.generate_patch(code, "", strategy="genetic")
        self.assertNotEqual(patch, code)

    def test_auto_strategy(self):
        patch = self.generator.generate_patch(self.example_code, "将pass替换为raise NotImplementedError()", strategy="auto")
        self.assertIn("raise NotImplementedError", patch)

    def test_list_and_clear_templates(self):
        self.assertTrue(len(self.generator.list_templates()) > 0)
        self.generator.clear_templates()
        self.assertEqual(len(self.generator.list_templates()), 0)

if __name__ == "__main__":
    unittest.main()
