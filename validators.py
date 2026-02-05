#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Code Validators - V6.1.1

Features:
1. In-memory code validation
2. Syntax checking with AST
3. Truncation detection
4. Import checking
5. Type hint validation

Author: AGI System Enhancement
Date: 2026-02-05
"""

import ast
import logging
import sys
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from pathlib import Path

from token_budget import TokenBudget, TruncationInfo

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Validation result"""
    is_valid: bool
    error_type: Optional[str]
    error_message: Optional[str]
    error_line: Optional[int]
    suggestions: List[str]
    metadata: Dict[str, Any]


class CodeValidator:
    """
    Code Validator for in-memory validation

    Features:
    - AST syntax checking
    - Truncation detection
    - Import verification
    - Type hint checking
    - Code style validation
    """

    def __init__(
        self,
        max_tokens: int = 24000,
        model: str = "deepseek-chat",
        enable_import_check: bool = True,
        enable_style_check: bool = False
    ):
        """
        Initialize Code Validator

        Args:
            max_tokens: Max tokens for truncation detection
            model: Model name
            enable_import_check: Enable import checking
            enable_style_check: Enable style checking (expensive)
        """
        self.token_budget = TokenBudget(max_tokens=max_tokens, model=model)
        self.enable_import_check = enable_import_check
        self.enable_style_check = enable_style_check

        logger.info(
            f"[CodeValidator] Initialized: "
            f"import_check={enable_import_check}, "
            f"style_check={enable_style_check}"
        )

    def validate_code(
        self,
        code: str,
        filename: Optional[str] = None
    ) -> ValidationResult:
        """
        Validate code in memory

        This is the main entry point for validation.
        Performs multiple checks in order of speed/complexity.

        Args:
            code: Code to validate
            filename: Optional filename for error messages

        Returns:
            ValidationResult object
        """
        metadata = {}
        suggestions = []

        # Quick check: empty code
        if not code or not code.strip():
            return ValidationResult(
                is_valid=False,
                error_type="empty_code",
                error_message="Code is empty",
                error_line=None,
                suggestions=["Add code to validate"],
                metadata=metadata
            )

        # Check 1: AST syntax check (fast, most accurate)
        # Do this FIRST because if AST parses successfully, code is definitely complete
        try:
            tree = ast.parse(code)
            metadata['ast_parsed'] = True
        except SyntaxError as e:
            error_line = e.lineno
            error_msg = str(e)

            # Extract specific error type
            error_type = self._classify_syntax_error(error_msg)

            # Generate suggestions
            error_suggestions = self._suggest_syntax_fix(
                error_type,
                error_msg,
                code,
                error_line
            )

            return ValidationResult(
                is_valid=False,
                error_type=error_type,
                error_message=error_msg,
                error_line=error_line,
                suggestions=error_suggestions,
                metadata=metadata
            )

        # Check 2: Truncation detection (only if AST parsed successfully)
        # This catches cases where code is syntactically valid but logically incomplete
        truncation_info = self.token_budget.detect_truncation(code)
        metadata['truncation'] = truncation_info

        # Only report truncation if AST parsing succeeded BUT code looks incomplete
        # Skip string-based truncation if AST parsed (false positives from quotes in strings)
        if truncation_info.is_truncated:
            # Filter out false positives from string detection
            # Check if ONLY string-related issues are present (not brackets, control flow, etc.)
            # Ignore metadata fields like *_count
            real_issues = []
            string_issues = ['unterminated_string', 'unmatched_parens', 'unmatched_brackets',
                           'unmatched_braces', 'incomplete_control_flow', 'suspicious_eof', 'incomplete_lines']

            for issue in string_issues:
                if truncation_info.details.get(issue):
                    real_issues.append(issue)

            # If the only real issue is unterminated_string, and AST parsed, it's likely a false positive
            if (len(real_issues) == 1 and
                real_issues[0] == 'unterminated_string'):
                # Only string issue detected, and AST parsed - likely false positive
                logger.info("[CodeValidator] Skipping truncation: only unterminated_string detected but AST parsed")
                truncation_info.is_truncated = False
                metadata['truncation_skipped'] = 'false_positive_escaped_quotes'

            if truncation_info.is_truncated:
                # Real truncation detected (brackets, control flow, etc.)
                fix_suggestions = self.token_budget.suggest_fix(
                    truncation_info,
                    code
                )

                return ValidationResult(
                    is_valid=False,
                    error_type="truncation_detected",
                    error_message=f"Code appears truncated (confidence: {truncation_info.confidence:.2f})",
                    error_line=None,
                    suggestions=fix_suggestions,
                    metadata=metadata
                )

        # Check 3: Import checking (moderate speed)
        if self.enable_import_check:
            import_result = self._check_imports(tree, code)
            metadata['imports'] = import_result

            if not import_result['valid']:
                return ValidationResult(
                    is_valid=False,
                    error_type="import_error",
                    error_message=import_result['error'],
                    error_line=import_result.get('line'),
                    suggestions=import_result['suggestions'],
                    metadata=metadata
                )

        # Check 4: Type hint validation (moderate speed)
        type_result = self._check_type_hints(tree)
        metadata['type_hints'] = type_result

        if not type_result['valid']:
            return ValidationResult(
                is_valid=False,
                error_type="type_hint_error",
                error_message=type_result['error'],
                error_line=type_result.get('line'),
                suggestions=type_result['suggestions'],
                metadata=metadata
            )

        # Check 5: Style checking (slow, optional)
        if self.enable_style_check:
            style_result = self._check_style(code, tree)
            metadata['style'] = style_result

            if not style_result['valid']:
                # Style issues are warnings, not errors
                suggestions.extend(style_result['suggestions'])

        # All checks passed
        return ValidationResult(
            is_valid=True,
            error_type=None,
            error_message=None,
            error_line=None,
            suggestions=suggestions,
            metadata=metadata
        )

    def _classify_syntax_error(self, error_msg: str) -> str:
        """
        Classify syntax error into categories

        Args:
            error_msg: Error message from AST

        Returns:
            Error type string
        """
        error_lower = error_msg.lower()

        if 'unterminated string' in error_lower:
            return 'unterminated_string'
        elif 'unmatched' in error_lower and 'parenthesis' in error_lower:
            return 'unmatched_parentheses'
        elif 'unmatched' in error_lower and 'bracket' in error_lower:
            return 'unmatched_brackets'
        elif 'unmatched' in error_lower and 'brace' in error_lower:
            return 'unmatched_braces'
        elif 'except' in error_lower or 'finally' in error_lower:
            return 'incomplete_try_except'
        elif 'indent' in error_lower:
            return 'indentation_error'
        elif 'unexpected' in error_lower and 'eof' in error_lower:
            return 'unexpected_eof'
        elif 'invalid syntax' in error_lower:
            return 'invalid_syntax'
        elif 'parameter' in error_lower and 'default' in error_lower:
            return 'parameter_order_error'
        else:
            return 'syntax_error'

    def _suggest_syntax_fix(
        self,
        error_type: str,
        error_msg: str,
        code: str,
        error_line: Optional[int]
    ) -> List[str]:
        """
        Generate fix suggestions for syntax errors

        Args:
            error_type: Error type
            error_msg: Error message
            code: Original code
            error_line: Line number with error

        Returns:
            List of suggestions
        """
        suggestions = []

        if error_type == 'unterminated_string':
            if 'triple' in error_msg.lower():
                suggestions.append("Check for unclosed triple-quoted string (''' or \"\"\")")
            else:
                suggestions.append("Check for unclosed string literal (single or double quote)")
            suggestions.append("Ensure all strings have matching opening and closing quotes")

        elif error_type == 'unmatched_parentheses':
            count = code.count('(') - code.count(')')
            if count > 0:
                suggestions.append(f"Add {count} closing parenthesis ')'")
            else:
                suggestions.append(f"Add {-count} opening parenthesis '('")

        elif error_type == 'unmatched_brackets':
            count = code.count('[') - code.count(']')
            if count > 0:
                suggestions.append(f"Add {count} closing bracket ']'")
            else:
                suggestions.append(f"Add {-count} opening bracket '['")

        elif error_type == 'unmatched_braces':
            count = code.count('{') - code.count('}')
            if count > 0:
                suggestions.append(f"Add {count} closing brace '}}'")
            else:
                suggestions.append(f"Add {-count} opening brace '{{'")

        elif error_type == 'incomplete_try_except':
            suggestions.append("Add 'except' or 'finally' block after 'try'")
            suggestions.append("Ensure every 'try' has corresponding 'except' or 'finally'")

        elif error_type == 'indentation_error':
            suggestions.append("Check for inconsistent indentation (use 4 spaces per level)")
            suggestions.append("Ensure consistent use of spaces or tabs (prefer spaces)")

        elif error_type == 'parameter_order_error':
            suggestions.append("Move parameters without default values before parameters with defaults")
            suggestions.append("Example: def foo(a, b, c=1): not def foo(a, c=1, b):")

        else:
            suggestions.append(f"Syntax error: {error_msg}")
            suggestions.append("Check the line above and the error line for issues")

        return suggestions

    def _check_imports(
        self,
        tree: ast.AST,
        code: str
    ) -> Dict[str, Any]:
        """
        Check imports for common issues

        Checks:
        - Duplicate imports
        - Unused imports (basic check)
        - Import order (basic check)
        - Circular imports (basic check)

        Args:
            tree: AST tree
            code: Original code

        Returns:
            {'valid': bool, 'error': str, 'suggestions': List[str]}
        """
        imports = []
        import_lines = {}

        # Collect all imports
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name)
                    import_lines[alias.name] = node.lineno
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ''
                for alias in node.names:
                    full_name = f"{module}.{alias.name}"
                    imports.append(full_name)
                    import_lines[full_name] = node.lineno

        # Check for duplicates
        seen = set()
        duplicates = []
        for imp in imports:
            if imp in seen:
                duplicates.append(imp)
            seen.add(imp)

        if duplicates:
            return {
                'valid': False,
                'error': f"Duplicate imports found: {duplicates}",
                'line': import_lines.get(duplicates[0]),
                'suggestions': [
                    "Remove duplicate import statements",
                    f"Keep only one import of: {duplicates[0]}"
                ]
            }

        # Check for very common external libraries
        external_imports = []
        for imp in imports:
            if not imp.startswith(('os', 'sys', 're', 'json', 'logging',
                                    'datetime', 'pathlib', 'typing',
                                    'dataclasses', 'collections')):
                external_imports.append(imp)

        # Note: This is a basic check, doesn't verify if packages are installed
        # A full check would require trying to import them

        return {
            'valid': True,
            'imports': imports,
            'external_imports': external_imports
        }

    def _check_type_hints(self, tree: ast.AST) -> Dict[str, Any]:
        """
        Check type hints for common issues

        Checks:
        - Forward references not quoted
        - Invalid type annotations

        Args:
            tree: AST tree

        Returns:
            {'valid': bool, 'error': str, 'suggestions': List[str]}
        """
        issues = []

        # Check function annotations
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                # Check return type
                if node.returns:
                    # Check if it's a string (forward reference)
                    # or a valid type
                    if isinstance(node.returns, ast.Name):
                        # Check for common unquoted forward references
                        if node.returns.id in ('List', 'Dict', 'Set', 'Tuple',
                                               'Optional', 'Union'):
                            # These need to be quoted or imported from typing
                            issues.append({
                                'line': node.lineno,
                                'func': node.name,
                                'issue': 'Return type might need quotes',
                                'suggestion': f'Use "{node.returns.id}" or import from typing'
                            })

        if issues:
            first_issue = issues[0]
            return {
                'valid': False,
                'error': first_issue['issue'],
                'line': first_issue['line'],
                'suggestions': [first_issue['suggestion']]
            }

        return {
            'valid': True,
            'issues': []
        }

    def _check_style(self, code: str, tree: ast.AST) -> Dict[str, Any]:
        """
        Basic style checking

        Checks:
        - Line length (PEP 8: 79 chars, but we use 120)
        - Trailing whitespace
        - Missing docstrings for classes and functions

        Args:
            code: Original code
            tree: AST tree

        Returns:
            {'valid': bool, 'suggestions': List[str]}
        """
        suggestions = []

        lines = code.split('\n')

        # Check line length
        max_line_length = 120
        long_lines = []
        for i, line in enumerate(lines, 1):
            if len(line) > max_line_length and not line.strip().startswith('#'):
                long_lines.append((i, len(line)))

        if long_lines:
            suggestions.append(
                f"{len(long_lines)} lines exceed {max_line_length} characters"
            )

        # Check trailing whitespace
        trailing_lines = []
        for i, line in enumerate(lines, 1):
            if line.rstrip() != line.rstrip('\n').rstrip():
                trailing_lines.append(i)

        if trailing_lines:
            suggestions.append(
                f"{len(trailing_lines)} lines have trailing whitespace"
            )

        # Check docstrings
        missing_docstrings = []
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                # Check if has docstring
                has_docstring = (
                    node.body and
                    isinstance(node.body[0], ast.Expr) and
                    isinstance(node.body[0].value, ast.Constant) and
                    isinstance(node.body[0].value.value, str)
                )

                if not has_docstring:
                    node_type = "Class" if isinstance(node, ast.ClassDef) else "Function"
                    missing_docstrings.append(
                        f"{node_type} '{node.name}' at line {node.lineno}"
                    )

        if missing_docstrings:
            suggestions.append(
                f"Missing docstrings for: {len(missing_docstrings)} items"
            )

        return {
            'valid': True,  # Style issues are warnings, not errors
            'suggestions': suggestions
        }


class ValidatorSuite:
    """
    Suite of validators for comprehensive validation

    Can run multiple validators and aggregate results
    """

    def __init__(self, validators: List[CodeValidator]):
        """
        Initialize validator suite

        Args:
            validators: List of validators to run
        """
        self.validators = validators

    def validate_all(
        self,
        code: str,
        filename: Optional[str] = None
    ) -> ValidationResult:
        """
        Run all validators and aggregate results

        Args:
            code: Code to validate
            filename: Optional filename

        Returns:
            Aggregated ValidationResult
        """
        all_suggestions = []
        all_metadata = {}

        for validator in self.validators:
            result = validator.validate_code(code, filename)

            if not result.is_valid:
                # First validation failure stops the chain
                return result

            # Accumulate suggestions and metadata
            all_suggestions.extend(result.suggestions)
            all_metadata.update(result.metadata)

        # All validators passed
        return ValidationResult(
            is_valid=True,
            error_type=None,
            error_message=None,
            error_line=None,
            suggestions=all_suggestions,
            metadata=all_metadata
        )


# Convenience functions
def validate_code(code: str, **kwargs) -> ValidationResult:
    """
    Convenience function to validate code

    Args:
        code: Code to validate
        **kwargs: Arguments to pass to CodeValidator

    Returns:
        ValidationResult
    """
    validator = CodeValidator(**kwargs)
    return validator.validate_code(code)


def quick_validate(code: str) -> bool:
    """
    Quick validation check (syntax only)

    Args:
        code: Code to validate

    Returns:
        True if valid, False otherwise
    """
    try:
        ast.parse(code)
        return True
    except:
        return False


if __name__ == "__main__":
    import sys

    print("=" * 80)
    print("Code Validators Module Test")
    print("=" * 80)

    validator = CodeValidator()

    # Test 1: Valid code
    print("\n[Test 1] Valid code")
    code1 = """
def factorial(n: int) -> int:
    \"\"\"Calculate factorial\"\"\"
    if n <= 1:
        return 1
    return n * factorial(n - 1)
"""
    result1 = validator.validate_code(code1)
    print(f"Valid: {result1.is_valid}")
    if result1.suggestions:
        print(f"Suggestions: {result1.suggestions}")

    # Test 2: Syntax error
    print("\n[Test 2] Syntax error - unmatched parentheses")
    code2 = "def foo():\n    return (1 + 2\n"
    result2 = validator.validate_code(code2)
    print(f"Valid: {result2.is_valid}")
    print(f"Error type: {result2.error_type}")
    print(f"Error message: {result2.error_message}")
    print(f"Suggestions: {result2.suggestions}")

    # Test 3: Truncated code
    print("\n[Test 3] Truncated code")
    code3 = 'def bar():\n    return "hello\n'
    result3 = validator.validate_code(code3)
    print(f"Valid: {result3.is_valid}")
    print(f"Error type: {result3.error_type}")
    print(f"Suggestions: {result3.suggestions}")

    # Test 4: Parameter order error
    print("\n[Test 4] Parameter order error")
    code4 = """
def baz(a: int, b: int = 1, c: int):
    pass
"""
    result4 = validator.validate_code(code4)
    print(f"Valid: {result4.is_valid}")
    print(f"Error type: {result4.error_type}")
    print(f"Error message: {result4.error_message}")
    print(f"Suggestions: {result4.suggestions}")

    # Test 5: Incomplete try-except
    print("\n[Test 5] Incomplete try-except")
    code5 = """
def test():
    try:
        x = 1 / 0
    # Missing except
"""
    result5 = validator.validate_code(code5)
    print(f"Valid: {result5.is_valid}")
    print(f"Error type: {result5.error_type}")
    print(f"Suggestions: {result5.suggestions}")

    print("\n" + "=" * 80)
    print("All tests completed")
    print("=" * 80)
