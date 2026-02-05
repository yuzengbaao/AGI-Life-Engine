#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LLM Semantic Fixer - V6.1.1

Features:
1. LLM-based semantic code fixing
2. Multi-round retry mechanism
3. Fix validation and verification
4. Structure similarity checking
5. Fallback strategies

Author: AGI System Enhancement
Date: 2026-02-05
"""

import ast
import logging
import re
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum

from validators import CodeValidator, ValidationResult

logger = logging.getLogger(__name__)


class FixStrategy(Enum):
    """Fix strategy enumeration"""
    LLM_SEMANTIC = "llm_semantic"  # Use LLM for semantic fixing
    HEURISTIC = "heuristic"         # Use heuristic rules
    FALLBACK = "fallback"           # Fallback to skeleton


@dataclass
class FixResult:
    """Result of a fix attempt"""
    success: bool
    fixed_code: Optional[str]
    attempts: int
    strategy_used: FixStrategy
    error_message: Optional[str]
    metadata: Dict[str, Any]


class LLMSemanticFixer:
    """
    LLM-based Semantic Code Fixer

    Uses LLM to fix syntax and semantic errors in generated code.
    Features multi-round retry with validation.
    """

    def __init__(
        self,
        llm_client,  # DeepSeekLLM or compatible client
        max_attempts: int = 3,
        temperature: float = 0.1,
        enable_structure_check: bool = True,
        similarity_threshold: float = 0.8
    ):
        """
        Initialize LLM Semantic Fixer

        Args:
            llm_client: LLM client with generate() method
            max_attempts: Maximum fix attempts per error
            temperature: LLM temperature (lower = more deterministic)
            enable_structure_check: Enable structure similarity checking
            similarity_threshold: Minimum structure similarity threshold
        """
        self.llm = llm_client
        self.max_attempts = max_attempts
        self.temperature = temperature
        self.enable_structure_check = enable_structure_check
        self.similarity_threshold = similarity_threshold

        self.validator = CodeValidator()

        logger.info(
            f"[LLMSemanticFixer] Initialized: "
            f"max_attempts={max_attempts}, "
            f"temperature={temperature}, "
            f"structure_check={enable_structure_check}"
        )

    async def fix_code(
        self,
        code: str,
        validation_result: ValidationResult,
        context: Optional[Dict] = None
    ) -> FixResult:
        """
        Fix code based on validation result

        Args:
            code: Original code with errors
            validation_result: Result from CodeValidator
            context: Optional context information

        Returns:
            FixResult object
        """
        if validation_result.is_valid:
            return FixResult(
                success=True,
                fixed_code=code,
                attempts=0,
                strategy_used=FixStrategy.LLM_SEMANTIC,
                error_message=None,
                metadata={'reason': 'Code already valid'}
            )

        logger.info(
            f"[LLMSemanticFixer] Attempting to fix: "
            f"{validation_result.error_type}"
        )

        # Try different strategies in order
        strategies = [
            FixStrategy.LLM_SEMANTIC,
            FixStrategy.HEURISTIC,
            FixStrategy.FALLBACK
        ]

        for strategy in strategies:
            result = await self._fix_with_strategy(
                code,
                validation_result,
                strategy,
                context
            )

            if result.success:
                logger.info(
                    f"[LLMSemanticFixer] Success with {strategy.value} "
                    f"after {result.attempts} attempts"
                )
                return result

        # All strategies failed
        return FixResult(
            success=False,
            fixed_code=None,
            attempts=0,
            strategy_used=FixStrategy.FALLBACK,
            error_message="All fix strategies failed",
            metadata={'original_error': validation_result.error_message}
        )

    async def _fix_with_strategy(
        self,
        code: str,
        validation_result: ValidationResult,
        strategy: FixStrategy,
        context: Optional[Dict]
    ) -> FixResult:
        """
        Fix code using specific strategy

        Args:
            code: Code to fix
            validation_result: Validation result
            strategy: Fix strategy to use
            context: Optional context

        Returns:
            FixResult
        """
        if strategy == FixStrategy.LLM_SEMANTIC:
            return await self._fix_with_llm(code, validation_result, context)
        elif strategy == FixStrategy.HEURISTIC:
            return self._fix_with_heuristics(code, validation_result)
        elif strategy == FixStrategy.FALLBACK:
            return self._fix_fallback(code, validation_result)
        else:
            return FixResult(
                success=False,
                fixed_code=None,
                attempts=0,
                strategy_used=strategy,
                error_message=f"Unknown strategy: {strategy}",
                metadata={}
            )

    async def _fix_with_llm(
        self,
        code: str,
        validation_result: ValidationResult,
        context: Optional[Dict]
    ) -> FixResult:
        """
        Fix code using LLM

        Multi-round retry with validation after each attempt.
        """
        error_type = validation_result.error_type or "unknown"
        error_message = validation_result.error_message or ""
        error_line = validation_result.error_line

        for attempt in range(1, self.max_attempts + 1):
            logger.info(f"[LLMSemanticFixer] LLM fix attempt {attempt}/{self.max_attempts}")

            # Build fix prompt
            prompt = self._build_fix_prompt(
                code,
                error_type,
                error_message,
                error_line,
                attempt,
                context
            )

            try:
                # Call LLM
                response = await self.llm.generate(
                    prompt,
                    max_tokens=8000,  # DeepSeek API limit: 8192
                    temperature=self.temperature
                )

                # Extract fixed code
                fixed_code = self._extract_code(response)

                if not fixed_code:
                    logger.warning(f"[LLMSemanticFixer] Attempt {attempt}: No code extracted")
                    continue

                # Validate fixed code
                fix_validation = self.validator.validate_code(fixed_code)

                if not fix_validation.is_valid:
                    logger.warning(
                        f"[LLMSemanticFixer] Attempt {attempt}: "
                        f"Fixed code still invalid: {fix_validation.error_type}"
                    )
                    continue

                # Check structure similarity (if enabled)
                if self.enable_structure_check:
                    similarity = self._calculate_structure_similarity(code, fixed_code)

                    if similarity < self.similarity_threshold:
                        logger.warning(
                            f"[LLMSemanticFixer] Attempt {attempt}: "
                            f"Structure similarity too low: {similarity:.2f}"
                        )
                        continue

                # Fix successful!
                logger.info(f"[LLMSemanticFixer] LLM fix successful on attempt {attempt}")

                return FixResult(
                    success=True,
                    fixed_code=fixed_code,
                    attempts=attempt,
                    strategy_used=FixStrategy.LLM_SEMANTIC,
                    error_message=None,
                    metadata={
                        'similarity': similarity if self.enable_structure_check else None,
                        'final_validation': fix_validation.metadata
                    }
                )

            except Exception as e:
                logger.error(f"[LLMSemanticFixer] Attempt {attempt} error: {e}")
                continue

        # All attempts failed
        return FixResult(
            success=False,
            fixed_code=None,
            attempts=self.max_attempts,
            strategy_used=FixStrategy.LLM_SEMANTIC,
            error_message=f"LLM fix failed after {self.max_attempts} attempts",
            metadata={}
        )

    def _build_fix_prompt(
        self,
        code: str,
        error_type: str,
        error_message: str,
        error_line: Optional[int],
        attempt: int,
        context: Optional[Dict]
    ) -> str:
        """
        Build fix prompt for LLM

        Args:
            code: Code with error
            error_type: Type of error
            error_message: Error message
            error_line: Line with error
            attempt: Attempt number
            context: Optional context

        Returns:
            Fix prompt string
        """
        # Get specific instructions based on error type
        specific_instructions = self._get_error_specific_instructions(error_type)

        # Build context section
        context_section = ""
        if context:
            if context.get('module_name'):
                context_section += f"Module: {context['module_name']}\n"
            if context.get('module_purpose'):
                context_section += f"Purpose: {context['module_purpose']}\n"

        # Build error location section
        error_location = ""
        if error_line:
            error_location = f"\nThe error is on or around line {error_line}."

        # Adjust instructions based on attempt number
        if attempt == 1:
            attempt_hint = "This is the first attempt. Be conservative in your fix."
        elif attempt == 2:
            attempt_hint = "Previous attempt failed. Try a different approach."
        else:
            attempt_hint = "Previous attempts failed. Focus on fixing ONLY the reported error."

        prompt = f"""Fix the Python code error described below.

{context_section}
Error Type: {error_type}
Error Message: {error_message}{error_location}

{specific_instructions}

{attempt_hint}

CRITICAL RULES:
1. Fix ONLY the reported error
2. Keep ALL other code EXACTLY the same
3. DO NOT add new features, methods, or logic
4. DO NOT refactor or optimize
5. Maintain the same code structure
6. Return ONLY the fixed Python code (no markdown, no explanation)

Code to fix:
```python
{code}
```

Return ONLY the complete fixed Python code:"""

        return prompt

    def _get_error_specific_instructions(self, error_type: str) -> str:
        """
        Get specific instructions for error type

        Args:
            error_type: Error type

        Returns:
            Specific instructions string
        """
        instructions = {
            'truncation_detected': """
The code appears to be truncated or incomplete.
Add the missing parts to complete the code.
""",
            'unterminated_string': """
There is an unterminated string literal.
Add the missing quote(s) to close the string.
Check both single (') and double (") quotes, including triple quotes (''' or \"\"\").
""",
            'unmatched_parentheses': """
There are unmatched parentheses.
Add the missing closing parenthesis ')' or opening parenthesis '('.
""",
            'unmatched_brackets': """
There are unmatched brackets.
Add the missing closing bracket ']' or opening bracket '['.
""",
            'unmatched_braces': """
There are unmatched braces.
Add the missing closing brace '}}' or opening brace '{{'.
""",
            'incomplete_try_except': """
There is a try block without corresponding except or finally.
Add 'except' or 'finally' block(s) as needed.
""",
            'indentation_error': """
There is an indentation error.
Check that all indentation is consistent (use 4 spaces per level).
""",
            'parameter_order_error': """
Parameters without default values follow parameters with defaults.
Reorder the parameters so that all parameters with defaults come last.
""",
            'unexpected_eof': """
The code ends unexpectedly.
Complete any incomplete statements or blocks.
""",
            'invalid_syntax': """
There is a syntax error.
Check for common issues like missing colons, brackets, or operators.
""",
            'syntax_error': """
There is a general syntax error.
Review the code for any syntax issues.
"""
        }

        return instructions.get(
            error_type,
            "Review and fix the syntax error."
        )

    def _extract_code(self, response: str) -> Optional[str]:
        """
        Extract code from LLM response

        Handles markdown code blocks and plain code.

        Args:
            response: LLM response

        Returns:
            Extracted code or None
        """
        if not response:
            return None

        response = response.strip()

        # Try to extract from markdown code blocks
        patterns = [
            r'```python\n(.*?)```',  # ```python ... ```
            r'```\n(.*?)```',        # ``` ... ```
            r'`(.*?)`',              # ` ... `
        ]

        for pattern in patterns:
            match = re.search(pattern, response, re.DOTALL)
            if match:
                code = match.group(1).strip()
                if code:
                    return code

        # If no code blocks found, check if response looks like code
        # More lenient check
        code_indicators = [
            'def ',
            'class ',
            'import ',
            'from ',
            '#',
            'return ',
            'print(',
            '    ',  # indented code
        ]

        if any(indicator in response for indicator in code_indicators):
            return response

        # Last resort: if response is multi-line and doesn't look like prose
        lines = response.split('\n')
        if len(lines) > 1:
            # Check if most lines look like code
            code_like_lines = sum(
                1 for line in lines
                if line.strip() and
                (line.strip().startswith(('def ', 'class ', 'import ', 'from ', '#', '    ')) or
                 '=' in line or 'return' in line or 'print' in line)
            )
            if code_like_lines / len([l for l in lines if l.strip()]) > 0.5:
                return response

        return None

    def _calculate_structure_similarity(
        self,
        original: str,
        fixed: str
    ) -> float:
        """
        Calculate structure similarity between original and fixed code

        Checks:
        - Number of functions
        - Number of classes
        - Number of top-level statements
        - Overall line count ratio

        Args:
            original: Original code
            fixed: Fixed code

        Returns:
            Similarity score (0.0 to 1.0)
        """
        try:
            orig_tree = ast.parse(original)
            fixed_tree = ast.parse(fixed)
        except:
            # If parsing fails, return 0
            return 0.0

        # Count functions
        orig_funcs = len([n for n in ast.walk(orig_tree) if isinstance(n, ast.FunctionDef)])
        fixed_funcs = len([n for n in ast.walk(fixed_tree) if isinstance(n, ast.FunctionDef)])

        # Count classes
        orig_classes = len([n for n in ast.walk(orig_tree) if isinstance(n, ast.ClassDef)])
        fixed_classes = len([n for n in ast.walk(fixed_tree) if isinstance(n, ast.ClassDef)])

        # Count top-level statements
        orig_stmts = len(orig_tree.body)
        fixed_stmts = len(fixed_tree.body)

        # Line count ratio
        orig_lines = len(original.split('\n'))
        fixed_lines = len(fixed.split('\n'))
        line_ratio = min(orig_lines, fixed_lines) / max(orig_lines, fixed_lines)

        # Calculate similarity score
        scores = []

        if orig_funcs > 0:
            scores.append(1.0 - abs(orig_funcs - fixed_funcs) / orig_funcs)
        else:
            scores.append(1.0 if fixed_funcs == 0 else 0.5)

        if orig_classes > 0:
            scores.append(1.0 - abs(orig_classes - fixed_classes) / orig_classes)
        else:
            scores.append(1.0 if fixed_classes == 0 else 0.5)

        if orig_stmts > 0:
            scores.append(1.0 - abs(orig_stmts - fixed_stmts) / orig_stmts)
        else:
            scores.append(1.0 if fixed_stmts == 0 else 0.5)

        scores.append(line_ratio)

        return sum(scores) / len(scores)

    def _fix_with_heuristics(
        self,
        code: str,
        validation_result: ValidationResult
    ) -> FixResult:
        """
        Fix code using heuristic rules

        Simple rule-based fixes for common errors.

        Args:
            code: Code to fix
            validation_result: Validation result

        Returns:
            FixResult
        """
        error_type = validation_result.error_type or "unknown"

        # Apply heuristic fixes based on error type
        if error_type == 'unmatched_parentheses':
            fixed = self._heuristic_fix_parens(code)
        elif error_type == 'unmatched_brackets':
            fixed = self._heuristic_fix_brackets(code)
        elif error_type == 'unmatched_braces':
            fixed = self._heuristic_fix_braces(code)
        elif error_type == 'unterminated_string':
            fixed = self._heuristic_fix_string(code)
        else:
            # No heuristic available
            return FixResult(
                success=False,
                fixed_code=None,
                attempts=1,
                strategy_used=FixStrategy.HEURISTIC,
                error_message="No heuristic available for this error type",
                metadata={}
            )

        # Validate the fixed code
        if fixed:
            fix_validation = self.validator.validate_code(fixed)

            if fix_validation.is_valid:
                return FixResult(
                    success=True,
                    fixed_code=fixed,
                    attempts=1,
                    strategy_used=FixStrategy.HEURISTIC,
                    error_message=None,
                    metadata={'heuristic_used': error_type}
                )

        # Heuristic fix failed
        return FixResult(
            success=False,
            fixed_code=None,
            attempts=1,
            strategy_used=FixStrategy.HEURISTIC,
            error_message="Heuristic fix did not resolve the error",
            metadata={}
        )

    def _heuristic_fix_parens(self, code: str) -> Optional[str]:
        """Heuristic fix for unmatched parentheses"""
        open_count = code.count('(')
        close_count = code.count(')')
        diff = open_count - close_count

        if diff > 0:
            # Add closing parens at the end
            return code + ')' * diff
        elif diff < 0:
            # Add opening parens at the start
            return '(' * (-diff) + code

        return None

    def _heuristic_fix_brackets(self, code: str) -> Optional[str]:
        """Heuristic fix for unmatched brackets"""
        open_count = code.count('[')
        close_count = code.count(']')
        diff = open_count - close_count

        if diff > 0:
            return code + ']' * diff
        elif diff < 0:
            return '[' * (-diff) + code

        return None

    def _heuristic_fix_braces(self, code: str) -> Optional[str]:
        """Heuristic fix for unmatched braces"""
        open_count = code.count('{')
        close_count = code.count('}')
        diff = open_count - close_count

        if diff > 0:
            return code + '}' * diff
        elif diff < 0:
            return '{' * (-diff) + code

        return None

    def _heuristic_fix_string(self, code: str) -> Optional[str]:
        """Heuristic fix for unterminated string"""
        lines = code.split('\n')

        for i in range(len(lines) - 1, -1, -1):
            line = lines[i]

            # Check for unclosed quotes
            if '"' in line and line.count('"') % 2 == 1:
                lines[i] = line + '"'
                return '\n'.join(lines)

            if "'" in line and line.count("'") % 2 == 1:
                lines[i] = line + "'"
                return '\n'.join(lines)

        return None

    def _fix_fallback(
        self,
        code: str,
        validation_result: ValidationResult
    ) -> FixResult:
        """
        Fallback strategy: return skeleton code

        Extract only the structure (class/function definitions) without implementations.

        Args:
            code: Code to fallback
            validation_result: Validation result

        Returns:
            FixResult with skeleton code
        """
        try:
            # Try to parse what we can
            tree = ast.parse(code, mode='exec')
        except:
            # Complete failure, return empty
            return FixResult(
                success=False,
                fixed_code=None,
                attempts=1,
                strategy_used=FixStrategy.FALLBACK,
                error_message="Cannot even parse structure",
                metadata={}
            )

        # Extract skeleton
        skeleton_lines = []
        code_lines = code.split('\n')

        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                if hasattr(node, 'lineno'):
                    line_idx = node.lineno - 1
                    if line_idx < len(code_lines):
                        skeleton_lines.append(code_lines[line_idx])

        skeleton = '\n'.join(skeleton_lines)

        # Add pass to empty bodies
        skeleton += "\n    pass  # TODO: implement"

        return FixResult(
            success=True,  # Mark as success (degraded)
            fixed_code=skeleton,
            attempts=1,
            strategy_used=FixStrategy.FALLBACK,
            error_message=None,
            metadata={
                'fallback_type': 'skeleton',
                'lines_kept': len(skeleton_lines)
            }
        )


# Convenience functions
async def fix_code_with_llm(
    code: str,
    validation_result: ValidationResult,
    llm_client,
    **kwargs
) -> FixResult:
    """
    Convenience function to fix code with LLM

    Args:
        code: Code to fix
        validation_result: Validation result
        llm_client: LLM client
        **kwargs: Arguments to LLMSemanticFixer

    Returns:
        FixResult
    """
    fixer = LLMSemanticFixer(llm_client, **kwargs)
    return await fixer.fix_code(code, validation_result)


if __name__ == "__main__":
    import asyncio
    from token_budget import TokenBudget

    # Mock LLM client for testing
    class MockLLM:
        async def generate(self, prompt, max_tokens=8000, temperature=0.7):
            # Simple mock that returns fixed code
            # For testing, we'll just add the missing closing elements

            if 'foo' in prompt:
                # Fix unmatched parentheses
                return "def foo():\n    return (1 + 2)\n"

            elif 'bar' in prompt:
                # Fix unterminated string
                return 'def bar():\n    return "hello"\n'

            elif 'baz' in prompt:
                # Already valid
                return "def baz():\n    return 42\n"

            else:
                return "# Fixed code\n"

    async def test():
        print("=" * 80)
        print("LLM Semantic Fixer Test")
        print("=" * 80)

        validator = CodeValidator()
        # Disable structure check for testing with mock LLM
        fixer = LLMSemanticFixer(
            MockLLM(),
            enable_structure_check=False  # Disable for mock testing
        )

        # Test 1: Unmatched parentheses
        print("\n[Test 1] Unmatched parentheses")
        code1 = "def foo():\n    return (1 + 2\n"
        result1 = validator.validate_code(code1)
        print(f"Original valid: {result1.is_valid}")
        print(f"Error type: {result1.error_type}")

        fix_result1 = await fixer.fix_code(code1, result1)
        print(f"Fix success: {fix_result1.success}")
        print(f"Attempts: {fix_result1.attempts}")
        print(f"Strategy: {fix_result1.strategy_used.value}")
        if fix_result1.fixed_code:
            print(f"Fixed code preview: {fix_result1.fixed_code[:100]}")

            # Verify the fix
            verify1 = validator.validate_code(fix_result1.fixed_code)
            print(f"Verified valid: {verify1.is_valid}")

        # Test 2: Unterminated string
        print("\n[Test 2] Unterminated string")
        code2 = 'def bar():\n    return "hello\n'
        result2 = validator.validate_code(code2)
        print(f"Original valid: {result2.is_valid}")

        fix_result2 = await fixer.fix_code(code2, result2)
        print(f"Fix success: {fix_result2.success}")
        if fix_result2.fixed_code:
            print(f"Fixed code preview: {fix_result2.fixed_code[:100]}")

            verify2 = validator.validate_code(fix_result2.fixed_code)
            print(f"Verified valid: {verify2.is_valid}")

        # Test 3: Already valid code
        print("\n[Test 3] Already valid code")
        code3 = "def baz():\n    return 42\n"
        result3 = validator.validate_code(code3)
        print(f"Original valid: {result3.is_valid}")

        fix_result3 = await fixer.fix_code(code3, result3)
        print(f"Fix success: {fix_result3.success}")
        print(f"Reason: {fix_result3.metadata.get('reason')}")

        # Test 4: Heuristic fix
        print("\n[Test 4] Heuristic fix (unmatched brackets)")
        code4 = "def test():\n    x = [1, 2, 3\n"
        result4 = validator.validate_code(code4)
        print(f"Original valid: {result4.is_valid}")

        fix_result4 = await fixer.fix_code(code4, result4)
        print(f"Fix success: {fix_result4.success}")
        print(f"Strategy: {fix_result4.strategy_used.value}")
        if fix_result4.fixed_code:
            print(f"Fixed code: {fix_result4.fixed_code}")
            verify4 = validator.validate_code(fix_result4.fixed_code)
            print(f"Verified valid: {verify4.is_valid}")

        print("\n" + "=" * 80)
        print("All tests completed")
        print("=" * 80)

    # Run tests
    asyncio.run(test())
