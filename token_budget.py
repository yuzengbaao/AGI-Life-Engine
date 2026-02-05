#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Token Budget Manager - V6.2.1 Enhanced

Features:
1. Token estimation
2. Code truncation detection
3. Token budget management
4. ENHANCED: Support for large file generation (24000 tokens)

Changelog:
- V6.2.1: Increased max_tokens from 8000 to 24000 (3x capacity)
- V6.2.1: Increased min_generation_tokens from 1000 to 3000
- V6.2.1: Optimized for generating 500-1000 line modules

Author: AGI System Enhancement
Date: 2026-02-05
"""

import re
import logging
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class TruncationInfo:
    """Truncation information"""
    is_truncated: bool
    details: Dict[str, bool]
    confidence: float
    suggestions: List[str]


class TokenBudget:
    """
    Token Budget Manager

    Features:
    - Estimate prompt and code tokens
    - Detect code truncation
    - Manage token budget
    """

    def __init__(
        self,
        max_tokens: int = 24000,
        model: str = "deepseek-chat",
        reserved_ratio: float = 0.1
    ):
        """
        Initialize Token Budget Manager

        Args:
            max_tokens: Model max tokens (increased to 24000 for larger files)
            model: Model name
            reserved_ratio: Reserved token ratio for generation
        """
        self.max_tokens = max_tokens
        self.model = model
        self.reserved_tokens = int(max_tokens * reserved_ratio)
        self.min_generation_tokens = 3000  # Increased from 1000 to support larger files

        # Token estimation config
        self.chars_per_token = 4

        logger.info(
            f"[TokenBudget] Initialized: max={max_tokens}, "
            f"reserved={self.reserved_tokens}, "
            f"available={max_tokens - self.reserved_tokens - 1000}"
        )

    def estimate_tokens(self, text: str) -> int:
        """
        Estimate token count for text

        Strategy:
        1. Try using tiktoken (if available)
        2. Fallback to char_count / 4

        Args:
            text: Text to estimate

        Returns:
            Estimated token count
        """
        if not text:
            return 0

        # Try tiktoken
        try:
            import tiktoken

            try:
                encoding = tiktoken.encoding_for_model(self.model)
                return len(encoding.encode(text))
            except KeyError:
                # Use cl100k_base as fallback
                encoding = tiktoken.get_encoding("cl100k_base")
                return len(encoding.encode(text))

        except ImportError:
            # Fallback: rough estimate
            return len(text) // self.chars_per_token

    def check_prompt_budget(
        self,
        prompt: str,
        min_generation_tokens: Optional[int] = None
    ) -> Tuple[bool, int, int]:
        """
        Check if prompt has enough token budget for generation

        Args:
            prompt: Prompt text
            min_generation_tokens: Min generation tokens (optional)

        Returns:
            (sufficient, prompt_tokens, available_tokens)
        """
        prompt_tokens = self.estimate_tokens(prompt)
        min_gen = min_generation_tokens or self.min_generation_tokens

        available = self.max_tokens - prompt_tokens - self.reserved_tokens
        sufficient = available >= min_gen

        if not sufficient:
            logger.warning(
                f"[TokenBudget] Insufficient: "
                f"prompt={prompt_tokens}, avail={available}, "
                f"required={min_gen}"
            )

        return sufficient, prompt_tokens, available

    def detect_truncation(self, code: str, detailed: bool = True) -> TruncationInfo:
        """
        Detect if code is truncated

        Detection strategies:
        1. Unbalanced brackets: (), [], {}
        2. Unterminated strings: single (')'), double ('")'), triple
        3. Incomplete control flow: try/except, if/else, for/while
        4. Suspicious EOF: ends with backslash, colon, etc.

        Args:
            code: Code to check
            detailed: Return detailed info

        Returns:
            TruncationInfo object
        """
        details = {}
        is_truncated = False
        suggestions = []

        # 1. Check bracket balance
        parens_ok = code.count('(') == code.count(')')
        brackets_ok = code.count('[') == code.count(']')
        braces_ok = code.count('{') == code.count('}')

        details['unmatched_parens'] = not parens_ok
        details['unmatched_brackets'] = not brackets_ok
        details['unmatched_braces'] = not braces_ok

        if not parens_ok:
            is_truncated = True
            suggestions.append("Unmatched parentheses '(' or ')'")

        if not brackets_ok:
            is_truncated = True
            suggestions.append("Unmatched brackets '[' or ']'")

        if not braces_ok:
            is_truncated = True
            suggestions.append("Unmatched braces '{' or '}'")

        # 2. Check string termination
        string_issues = self._check_unterminated_strings(code)
        details['unterminated_string'] = string_issues['has_issue']

        if string_issues['has_issue']:
            is_truncated = True
            details.update(string_issues['details'])
            suggestions.append(f"Unterminated {string_issues['type']} string")

        # 3. Check control flow completeness
        control_flow_issues = self._check_control_flow(code)
        details['incomplete_control_flow'] = control_flow_issues['has_issue']

        if control_flow_issues['has_issue']:
            is_truncated = True
            details.update(control_flow_issues['details'])
            suggestions.append(f"Incomplete {control_flow_issues['type']}")

        # 4. Check EOF
        eof_issues = self._check_eof(code)
        details['suspicious_eof'] = eof_issues['has_issue']

        if eof_issues['has_issue']:
            is_truncated = True
            details.update(eof_issues['details'])
            suggestions.append(f"Suspicious EOF: {eof_issues['reason']}")

        # 5. Check line completeness
        line_issues = self._check_incomplete_lines(code)
        details['incomplete_lines'] = line_issues['has_issue']

        if line_issues['has_issue']:
            is_truncated = True
            details.update(line_issues['details'])
            suggestions.append(f"Incomplete lines: {line_issues['count']}")

        # Calculate confidence
        confidence = self._calculate_confidence(details, is_truncated)

        return TruncationInfo(
            is_truncated=is_truncated,
            details=details if detailed else {},
            confidence=confidence,
            suggestions=suggestions
        )

    def _check_unterminated_strings(self, code: str) -> Dict:
        """
        Check for unterminated strings

        Returns:
            {'has_issue': bool, 'type': str, 'details': dict}
        """
        result = {
            'has_issue': False,
            'type': None,
            'details': {}
        }

        # Count quotes (excluding escaped)
        single_quotes = code.count("'") - code.count("\\'")
        double_quotes = code.count('"') - code.count('\\"')

        # Triple quotes - avoid syntax issues
        triple_single = code.count("'''")
        triple_double = code.count('"' + '"' + '"')  # Triple double quotes

        result['details']['single_quote_count'] = single_quotes
        result['details']['double_quote_count'] = double_quotes
        result['details']['triple_single_count'] = triple_single
        result['details']['triple_double_count'] = triple_double

        # Check triple quotes
        if triple_single % 2 != 0:
            result['has_issue'] = True
            result['type'] = "triple-single"
            result['details']['unterminated_triple_single'] = True

        if triple_double % 2 != 0:
            result['has_issue'] = True
            result['type'] = "triple-double"
            result['details']['unterminated_triple_double'] = True

        # Check normal quotes
        if single_quotes % 2 != 0 and triple_single % 2 == 0:
            result['has_issue'] = True
            if not result['type']:
                result['type'] = "single"

        if double_quotes % 2 != 0 and triple_double % 2 == 0:
            result['has_issue'] = True
            if not result['type']:
                result['type'] = "double"

        return result

    def _check_control_flow(self, code: str) -> Dict:
        """
        Check control flow completeness

        Checks: try/except, if/else, for/while
        """
        result = {
            'has_issue': False,
            'type': None,
            'details': {}
        }

        # Remove strings and comments
        clean_code = self._remove_strings_and_comments(code)

        # Count control flow keywords
        try_count = len(re.findall(r'\btry\s*:', clean_code))
        except_count = len(re.findall(r'\bexcept\b', clean_code))
        finally_count = len(re.findall(r'\bfinally\s*:', clean_code))

        if_count = len(re.findall(r'\bif\s+.*\s*:', clean_code))
        elif_count = len(re.findall(r'\belif\s+.*\s*:', clean_code))
        else_count = len(re.findall(r'\belse\s*:', clean_code))

        for_count = len(re.findall(r'\bfor\s+', clean_code))
        while_count = len(re.findall(r'\bwhile\s+', clean_code))

        with_count = len(re.findall(r'\bwith\s+', clean_code))

        result['details']['try_count'] = try_count
        result['details']['except_count'] = except_count
        result['details']['finally_count'] = finally_count
        result['details']['if_count'] = if_count
        result['details']['elif_count'] = elif_count
        result['details']['else_count'] = else_count
        result['details']['for_count'] = for_count
        result['details']['while_count'] = while_count
        result['details']['with_count'] = with_count

        # Check try/except
        if try_count > 0:
            if except_count + finally_count < try_count:
                result['has_issue'] = True
                result['type'] = "try-except"
                result['details']['missing_except_or_finally'] = True

        return result

    def _check_eof(self, code: str) -> Dict:
        """
        Check if EOF is suspicious

        Suspicious signs:
        - Ends with backslash (line continuation)
        - Ends with colon (might be missing block)
        - No final newline
        """
        result = {
            'has_issue': False,
            'reason': None,
            'details': {}
        }

        if not code:
            return result

        # Strip trailing whitespace
        stripped = code.rstrip()

        # Check for backslash at end
        if stripped.endswith('\\'):
            result['has_issue'] = True
            result['reason'] = "ends with backslash"
            result['details']['ends_with_backslash'] = True

        # Check for colon at end
        elif stripped.endswith(':'):
            lines = code.split('\n')
            last_line = stripped.split('\n')[-1]

            if not re.match(r'^\s*(def|class)\s+', last_line):
                result['has_issue'] = True
                result['reason'] = "ends with colon"
                result['details']['ends_with_colon'] = True

        # Check for final newline
        if not code.endswith('\n'):
            result['details']['missing_final_newline'] = True

        return result

    def _check_incomplete_lines(self, code: str) -> Dict:
        """
        Check for incomplete lines

        Find lines that look truncated
        """
        result = {
            'has_issue': False,
            'count': 0,
            'details': {}
        }

        lines = code.split('\n')
        incomplete_lines = []

        for i, line in enumerate(lines, 1):
            stripped = line.strip()

            # Skip empty and comments
            if not stripped or stripped.startswith('#'):
                continue

            # Check for backslash (line continuation)
            if stripped.endswith('\\'):
                incomplete_lines.append(i)

            # REMOVED: Comma check - commas at end of line are valid Python syntax
            # They're used in function arguments, list/dict elements, etc.

            # Check for operators (suspicious if line ends with operator)
            if stripped.endswith(('+', '-', '*', '/', '|', '&', '=')):
                incomplete_lines.append(i)

        if incomplete_lines:
            result['has_issue'] = True
            result['count'] = len(incomplete_lines)
            result['details']['incomplete_line_numbers'] = incomplete_lines[-5:]

        return result

    def _remove_strings_and_comments(self, code: str) -> str:
        """
        Remove strings and comments for control flow analysis

        Simplified implementation, may not handle all edge cases
        """
        result = []
        in_string = False
        string_char = None
        in_comment = False

        i = 0
        while i < len(code):
            char = code[i]

            # Check for comment
            if not in_string and not in_comment:
                if char == '#':
                    in_comment = True
                    i += 1
                    while i < len(code) and code[i] != '\n':
                        i += 1
                    continue

            # Exit comment
            if in_comment and char == '\n':
                in_comment = False
                result.append(char)
                i += 1
                continue

            if in_comment:
                i += 1
                continue

            # Check for string
            if char in ('"', "'") and (i == 0 or code[i-1] != '\\'):
                if not in_string:
                    in_string = True
                    string_char = char
                elif char == string_char:
                    in_string = False
                    string_char = None

            if in_string:
                i += 1
                continue

            # Keep non-string, non-comment content
            result.append(char)
            i += 1

        return ''.join(result)

    def _calculate_confidence(
        self,
        details: Dict[str, bool],
        is_truncated: bool
    ) -> float:
        """
        Calculate detection confidence

        Based on signal count and types
        """
        if not is_truncated:
            return 1.0

        # High confidence signals
        high_conf = [
            'unmatched_parens',
            'unmatched_brackets',
            'unmatched_braces',
            'unterminated_string',
        ]

        # Medium confidence signals
        medium_conf = [
            'incomplete_control_flow',
            'suspicious_eof',
        ]

        # Low confidence signals
        low_conf = [
            'incomplete_lines',
        ]

        score = 0.0
        weight_sum = 0.0

        for signal in high_conf:
            if details.get(signal, False):
                score += 1.0 * 1.0
                weight_sum += 1.0

        for signal in medium_conf:
            if details.get(signal, False):
                score += 1.0 * 0.7
                weight_sum += 0.7

        for signal in low_conf:
            if details.get(signal, False):
                score += 1.0 * 0.5
                weight_sum += 0.5

        if weight_sum == 0:
            return 0.5

        return min(1.0, score / weight_sum + 0.3)

    def suggest_fix(self, truncation_info: TruncationInfo, code: str) -> List[str]:
        """
        Suggest fixes based on truncation info

        Args:
            truncation_info: Truncation detection result
            code: Original code

        Returns:
            List of fix suggestions
        """
        suggestions = []

        if not truncation_info.is_truncated:
            suggestions.append("No truncation detected.")
            return suggestions

        details = truncation_info.details

        # Unmatched brackets
        if details.get('unmatched_parens'):
            count = code.count('(') - code.count(')')
            if count > 0:
                suggestions.append(f"Add {count} closing parenthesis ')'")
            else:
                suggestions.append(f"Add {-count} opening parenthesis '('")

        if details.get('unmatched_brackets'):
            count = code.count('[') - code.count(']')
            if count > 0:
                suggestions.append(f"Add {count} closing bracket ']'")
            else:
                suggestions.append(f"Add {-count} opening bracket '['")

        if details.get('unmatched_braces'):
            count = code.count('{') - code.count('}')
            if count > 0:
                suggestions.append(f"Add {count} closing brace '}}'")
            else:
                suggestions.append(f"Add {-count} opening brace '{{'")

        # Unterminated strings
        if details.get('unterminated_string'):
            if details.get('unterminated_triple_single'):
                suggestions.append("Close triple-single string (\"''\")")
            if details.get('unterminated_triple_double'):
                suggestions.append("Close triple-double string (\"\"\")")

        # Incomplete control flow
        if details.get('missing_except_or_finally'):
            try_count = details.get('try_count', 0)
            except_count = details.get('except_count', 0)
            finally_count = details.get('finally_count', 0)
            needed = try_count - except_count - finally_count
            if needed > 0:
                suggestions.append(f"Add {needed} 'except' or 'finally' block(s)")

        # EOF issues
        if details.get('ends_with_backslash'):
            suggestions.append("Remove backslash or complete the line")

        if details.get('ends_with_colon'):
            suggestions.append("Add code block after colon")

        return suggestions

    def __repr__(self) -> str:
        return (
            f"TokenBudget(max={self.max_tokens}, "
            f"reserved={self.reserved_tokens}, "
            f"model={self.model})"
        )


# Convenience functions
def detect_code_truncation(code: str, max_tokens: int = 24000) -> TruncationInfo:
    """
    Convenience function: Detect code truncation

    Args:
        code: Code to check
        max_tokens: Max tokens

    Returns:
        TruncationInfo object
    """
    budget = TokenBudget(max_tokens=max_tokens)
    return budget.detect_truncation(code)


def estimate_code_tokens(code: str, model: str = "deepseek-chat") -> int:
    """
    Convenience function: Estimate code tokens

    Args:
        code: Code text
        model: Model name

    Returns:
        Estimated token count
    """
    budget = TokenBudget(model=model)
    return budget.estimate_tokens(code)


if __name__ == "__main__":
    import sys

    print("=" * 80)
    print("TokenBudget Module Test - V6.2 Enhanced (24000 tokens)")
    print("=" * 80)

    budget = TokenBudget(max_tokens=24000)

    # Test 1: Unmatched brackets
    print("\n[Test 1] Unmatched parentheses")
    code1 = "def foo():\n    return (1 + 2\n"
    result1 = budget.detect_truncation(code1)
    print(f"Code: {repr(code1)}")
    print(f"Truncated: {result1.is_truncated}")
    print(f"Confidence: {result1.confidence:.2f}")
    print(f"Details: {result1.details}")
    print(f"Suggestions: {result1.suggestions}")

    # Test 2: Normal code
    print("\n[Test 2] Normal code")
    code2 = "def foo():\n    return (1 + 2)\n"
    result2 = budget.detect_truncation(code2)
    print(f"Code: {repr(code2)}")
    print(f"Truncated: {result2.is_truncated}")
    print(f"Confidence: {result2.confidence:.2f}")

    # Test 3: Unterminated string
    print("\n[Test 3] Unterminated string")
    code3 = 'def foo():\n    return "hello\n'
    result3 = budget.detect_truncation(code3)
    print(f"Code: {repr(code3)}")
    print(f"Truncated: {result3.is_truncated}")
    print(f"Details: {result3.details}")
    print(f"Suggestions: {result3.suggestions}")

    # Test 4: Incomplete try-except
    print("\n[Test 4] Incomplete try-except")
    code4 = """
def foo():
    try:
        x = 1 / 0
    # Missing except
"""
    result4 = budget.detect_truncation(code4)
    print(f"Truncated: {result4.is_truncated}")
    print(f"Details: {result4.details}")
    print(f"Suggestions: {result4.suggestions}")

    # Test 5: Token estimation
    print("\n[Test 5] Token estimation")
    test_code = """
def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n - 1)
"""
    tokens = budget.estimate_tokens(test_code)
    print(f"Lines: {len(test_code.splitlines())}")
    print(f"Estimated tokens: {tokens}")

    # Test 6: Budget check
    print("\n[Test 6] Budget check")
    long_prompt = "Generate code" * 1000
    sufficient, prompt_tokens, available = budget.check_prompt_budget(long_prompt)
    print(f"Prompt tokens: {prompt_tokens}")
    print(f"Available: {available}")
    print(f"Sufficient: {sufficient}")

    print("\n" + "=" * 80)
    print("All tests completed")
    print("=" * 80)
