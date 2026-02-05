#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Error Classification and Handling System - V6.1.2 Phase 2 Component 3

Features:
1. Error pattern recognition
2. Categorized fix strategies
3. Fix effectiveness evaluation
4. Learning from history

Author: AGI System Enhancement
Date: 2026-02-05
"""

import logging
import re
import json
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
from datetime import datetime

from validators import CodeValidator, ValidationResult
from fixers import LLMSemanticFixer, FixResult

logger = logging.getLogger(__name__)


class ErrorCategory(Enum):
    """Error categories for classification"""
    SYNTAX_TRUNCATION = "syntax_truncation"  # Code truncated
    SYNTAX_STRUCTURE = "syntax_structure"    # Structural issues
    SEMANTIC_TYPE = "semantic_type"          # Type errors
    IMPORT_DEPENDENCY = "import_dependency"  # Import issues
    LOGIC_ERROR = "logic_error"            # Logic errors
    UNKNOWN = "unknown"                     # Unknown errors


class FixStrategy(Enum):
    """Fix strategy types"""
    LLM_SEMANTIC = "llm_semantic"         # Use LLM for semantic fixing
    HEURISTIC_RULE = "heuristic_rule"   # Use rule-based fixes
    TEMPLATE_BASED = "template_based"     # Use template replacements
    HYBRID = "hybrid"                     # Combine multiple strategies
    SKIP = "skip"                         # Skip and continue


@dataclass
class ErrorPattern:
    """Error pattern definition"""
    pattern_id: str
    category: ErrorCategory
    name: str
    signature_regex: str
    examples: List[str]
    suggested_strategies: List[FixStrategy]
    success_rate: float = 0.0
    fix_count: int = 0


@dataclass
class ClassifiedError:
    """Classified error with metadata"""
    original_error: str
    category: ErrorCategory
    pattern_id: Optional[str]
    confidence: float
    context: Dict[str, Any]
    suggested_strategies: List[FixStrategy]


@dataclass
class FixAttempt:
    """Record of a fix attempt"""
    error_pattern: str
    strategy_used: FixStrategy
    success: bool
    time_taken: float
    tokens_used: int
    timestamp: datetime
    feedback_score: Optional[float] = None


class ErrorClassifier:
    """
    Error Classifier - V6.1.2

    Classifies errors into patterns and suggests optimal fix strategies.
    Learns from fix history to improve classification and strategy selection.
    """

    def __init__(
        self,
        enable_learning: bool = True,
        history_file: Optional[str] = None
    ):
        """
        Initialize Error Classifier

        Args:
            enable_learning: Enable learning from history
            history_file: File to save/load learning history
        """
        self.enable_learning = enable_learning
        self.history_file = history_file

        # Initialize error patterns database
        self.patterns = self._initialize_patterns()

        # Fix history for learning
        self.fix_history: List[FixAttempt] = []

        # Pattern statistics
        self.pattern_stats: Dict[str, Dict] = defaultdict(lambda: {
            'attempts': 0,
            'successes': 0,
            'strategy_performance': defaultdict(lambda: {
                'attempts': 0,
                'successes': 0,
                'avg_time': 0.0,
                'avg_tokens': 0
            })
        })

        # Load history if enabled
        if enable_learning and history_file:
            self._load_history()

        logger.info(
            f"[ErrorClassifier] Initialized: "
            f"patterns={len(self.patterns)}, "
            f"learning={enable_learning}"
        )

    def _initialize_patterns(self) -> List[ErrorPattern]:
        """Initialize error pattern database"""
        patterns = []

        # Pattern 1: Unmatched parentheses
        patterns.append(ErrorPattern(
            pattern_id="unmatched_parens",
            category=ErrorCategory.SYNTAX_STRUCTURE,
            name="Unmatched Parentheses",
            signature_regex=r"\(.*[^)]*\s*$",
            examples=[
                "def foo():\n    return (1 + 2\n",
                "x = [1, 2, (3, 4]"
            ],
            suggested_strategies=[
                FixStrategy.HEURISTIC_RULE,
                FixStrategy.LLM_SEMANTIC
            ]
        ))

        # Pattern 2: Unterminated string
        patterns.append(ErrorPattern(
            pattern_id="unterminated_string",
            category=ErrorCategory.SYNTAX_TRUNCATION,
            name="Unterminated String",
            signature_regex=r'["\'].*[^"\']\s*$',
            examples=[
                'def foo():\n    return "hello\n',
                "x = 'test"
            ],
            suggested_strategies=[
                FixStrategy.HEURISTIC_RULE,
                FixStrategy.LLM_SEMANTIC
            ]
        ))

        # Pattern 3: Incomplete try-except
        patterns.append(ErrorPattern(
            pattern_id="incomplete_try_except",
            category=ErrorCategory.SYNTAX_STRUCTURE,
            name="Incomplete Try-Except",
            signature_regex=r"try:\s*\n\s*[^e]",
            examples=[
                "try:\n    x = 1\n# Missing except",
                "try:\n    pass\n# Missing except"
            ],
            suggested_strategies=[
                FixStrategy.TEMPLATE_BASED,
                FixStrategy.LLM_SEMANTIC
            ]
        ))

        # Pattern 4: Parameter order error
        patterns.append(ErrorPattern(
            pattern_id="parameter_order",
            category=ErrorCategory.SYNTAX_STRUCTURE,
            name="Parameter Order Error",
            signature_regex=r"def.*=\w+\s*\w+\s*:",
            examples=[
                "def foo(a, b=1, c):",
                "def bar(x=1, y):"
            ],
            suggested_strategies=[
                FixStrategy.HEURISTIC_RULE,
                FixStrategy.LLM_SEMANTIC
            ]
        ))

        # Pattern 5: Missing import
        patterns.append(ErrorPattern(
            pattern_id="missing_import",
            category=ErrorCategory.IMPORT_DEPENDENCY,
            name="Missing Import",
            signature_regex=r"NameError|not defined",
            examples=[
                "json.loads(data)  # Missing 'import json'",
                "pd.DataFrame()  # Missing 'import pandas as pd'"
            ],
            suggested_strategies=[
                FixStrategy.HEURISTIC_RULE,
                FixStrategy.LLM_SEMANTIC
            ]
        ))

        # Pattern 6: Indentation error
        patterns.append(ErrorPattern(
            pattern_id="indentation_error",
            category=ErrorCategory.SYNTAX_STRUCTURE,
            name="Indentation Error",
            signature_regex=r"\n\s+.{4}\n[^\s]",
            examples=[
                "def foo():\n    x = 1\n  y = 2",  # Mixed indent
                "if True:\n    if True:\n        pass"    # Deep indent
            ],
            suggested_strategies=[
                FixStrategy.HEURISTIC_RULE,
                FixStrategy.LLM_SEMANTIC
            ]
        ))

        return patterns

    def classify_error(
        self,
        validation_result: ValidationResult,
        code_context: Optional[str] = None
    ) -> ClassifiedError:
        """
        Classify error into pattern and suggest strategies

        Args:
            validation_result: Validation result from CodeValidator
            code_context: Original code (optional)

        Returns:
            ClassifiedError object
        """
        error_type = validation_result.error_type or "unknown"
        error_message = validation_result.error_message or ""

        logger.info(f"[Classify] Classifying error: {error_type}")

        # Try to match against known patterns
        best_match = None
        best_score = 0.0

        for pattern in self.patterns:
            # Calculate match score
            score = self._calculate_match_score(
                error_type,
                error_message,
                pattern,
                code_context
            )

            if score > best_score:
                best_score = score
                best_match = pattern

        if best_match and best_score > 0.3:  # Threshold for pattern match
            confidence = min(1.0, best_score + 0.2)  # Boost confidence slightly

            return ClassifiedError(
                original_error=error_type,
                category=best_match.category,
                pattern_id=best_match.pattern_id,
                confidence=confidence,
                context={
                    'error_message': error_message,
                    'error_line': validation_result.error_line
                },
                suggested_strategies=best_match.suggested_strategies
            )

        # No pattern match, classify based on error type
        category = self._categorize_by_type(error_type)

        return ClassifiedError(
            original_error=error_type,
            category=category,
            pattern_id=None,
            confidence=0.5,  # Lower confidence for unknown
            context={
                'error_message': error_message,
                'error_line': validation_result.error_line
            },
            suggested_strategies=self._get_default_strategies(category)
        )

    def _calculate_match_score(
        self,
        error_type: str,
        error_message: str,
        pattern: ErrorPattern,
        code_context: Optional[str]
    ) -> float:
        """
        Calculate how well an error matches a pattern

        Args:
            error_type: Type of error
            error_message: Error message
            pattern: Error pattern
            code_context: Code context

        Returns:
            Match score (0.0 to 1.0)
        """
        score = 0.0

        # Factor 1: Error type match
        if error_type == 'truncation_detected' and pattern.category == ErrorCategory.SYNTAX_TRUNCATION:
            score += 0.4
        elif error_type in ['unmatched_parens', 'unmatched_brackets', 'unmatched_braces']:
            if pattern.category == ErrorCategory.SYNTAX_STRUCTURE:
                score += 0.5

        # Factor 2: Error message keyword match
        message_lower = error_message.lower()
        pattern_name_lower = pattern.name.lower()

        if pattern_name_lower in message_lower:
            score += 0.3

        # Factor 3: Code context pattern matching
        if code_context:
            if re.search(pattern.signature_regex, code_context, re.MULTILINE):
                score += 0.3

        return min(1.0, score)

    def _categorize_by_type(self, error_type: str) -> ErrorCategory:
        """Categorize error by type when pattern doesn't match"""
        if 'truncation' in error_type.lower():
            return ErrorCategory.SYNTAX_TRUNCATION
        elif error_type in ['unmatched_parens', 'unmatched_brackets', 'unmatched_braces']:
            return ErrorCategory.SYNTAX_STRUCTURE
        elif 'import' in error_type.lower():
            return ErrorCategory.IMPORT_DEPENDENCY
        elif 'parameter' in error_type.lower():
            return ErrorCategory.SEMANTIC_TYPE
        else:
            return ErrorCategory.UNKNOWN

    def _get_default_strategies(self, category: ErrorCategory) -> List[FixStrategy]:
        """Get default strategies for a category"""
        strategies = {
            ErrorCategory.SYNTAX_TRUNCATION: [FixStrategy.LLM_SEMANTIC],
            ErrorCategory.SYNTAX_STRUCTURE: [FixStrategy.HEURISTIC_RULE, FixStrategy.LLM_SEMANTIC],
            ErrorCategory.SEMANTIC_TYPE: [FixStrategy.LLM_SEMANTIC],
            ErrorCategory.IMPORT_DEPENDENCY: [FixStrategy.HEURISTIC_RULE],
            ErrorCategory.LOGIC_ERROR: [FixStrategy.LLM_SEMANTIC],
            ErrorCategory.UNKNOWN: [FixStrategy.LLM_SEMANTIC]
        }
        return strategies.get(category, [FixStrategy.LLM_SEMANTIC])

    def select_optimal_strategy(
        self,
        classified_error: ClassifiedError,
        llm_fixer: Optional[LLMSemanticFixer] = None
    ) -> FixStrategy:
        """
        Select optimal fix strategy based on classification and history

        Args:
            classified_error: Classified error
            llm_fixer: LLM fixer (optional)

        Returns:
            Selected fix strategy
        """
        suggested_strategies = classified_error.suggested_strategies

        if not suggested_strategies:
            return FixStrategy.LLM_SEMANTIC

        # If learning is disabled, return first suggested
        if not self.enable_learning:
            return suggested_strategies[0]

        # Check historical performance for this pattern
        if classified_error.pattern_id:
            pattern_id = classified_error.pattern_id
            stats = self.pattern_stats[pattern_id]

            # Find best performing strategy
            best_strategy = None
            best_success_rate = 0.0

            for strategy in suggested_strategies:
                if strategy.value in stats['strategy_performance']:
                    perf = stats['strategy_performance'][strategy.value]
                    success_rate = perf['successes'] / max(1, perf['attempts'])

                    if success_rate > best_success_rate and perf['attempts'] >= 3:
                        best_strategy = strategy
                        best_success_rate = success_rate

            if best_strategy:
                logger.info(
                    f"[Strategy] Selected {best_strategy.value} "
                    f"(success_rate={best_success_rate:.2f}, "
                    f"attempts={stats['strategy_performance'][best_strategy.value]['attempts']})"
                )
                return best_strategy

        # Fall back to first suggested strategy
        return suggested_strategies[0]

    def record_fix_attempt(
        self,
        classified_error: ClassifiedError,
        strategy: FixStrategy,
        fix_result: FixResult
    ):
        """
        Record fix attempt for learning

        Args:
            classified_error: The classified error
            strategy: Strategy used
            fix_result: Result of fix attempt
        """
        pattern_id = classified_error.pattern_id or "unknown"

        attempt = FixAttempt(
            error_pattern=pattern_id,
            strategy_used=strategy,
            success=fix_result.success,
            time_taken=0.0,  # Would be populated in real usage
            tokens_used=0,
            timestamp=datetime.now()
        )

        self.fix_history.append(attempt)

        # Update statistics
        stats = self.pattern_stats[pattern_id]
        stats['attempts'] += 1

        if fix_result.success:
            stats['successes'] += 1

        # Update strategy-specific stats
        strategy_stats = stats['strategy_performance'][strategy.value]
        strategy_stats['attempts'] += 1

        if fix_result.success:
            strategy_stats['successes'] += 1

        logger.info(
            f"[Learning] Recorded fix attempt: "
            f"pattern={pattern_id}, "
            f"strategy={strategy.value}, "
            f"success={fix_result.success}"
        )

        # Save history if enabled
        if self.enable_learning and self.history_file:
            self._save_history()

    def get_pattern_statistics(self) -> Dict[str, Dict]:
        """
        Get statistics for all patterns

        Returns:
            Dictionary of pattern stats
        """
        summary = {}

        for pattern_id, stats in self.pattern_stats.items():
            total_attempts = stats['attempts']
            total_successes = stats['successes']

            summary[pattern_id] = {
                'total_attempts': total_attempts,
                'total_successes': total_successes,
                'overall_success_rate': total_successes / max(1, total_attempts),
                'strategy_breakdown': dict(
                    (k, {
                        'attempts': v['attempts'],
                        'successes': v['successes'],
                        'success_rate': v['successes'] / max(1, v['attempts'])
                    })
                    for k, v in stats['strategy_performance'].items()
                )
            }

        return summary

    def get_learning_insights(self) -> List[str]:
        """
        Get insights from learning history

        Returns:
            List of insight strings
        """
        insights = []

        for pattern_id, stats in self.pattern_stats.items():
            if stats['attempts'] >= 3:  # Only analyze patterns with sufficient data
                total_attempts = stats['attempts']
                total_successes = stats['successes']
                success_rate = total_successes / total_attempts

                # Find best strategy
                best_strategy = None
                best_rate = 0.0

                for strategy_name, strategy_stats in stats['strategy_performance'].items():
                    if strategy_stats['attempts'] >= 2:
                        rate = strategy_stats['successes'] / strategy_stats['attempts']
                        if rate > best_rate:
                            best_rate = rate
                            best_strategy = strategy_name

                if best_strategy:
                    insights.append(
                        f"Pattern '{pattern_id}': {success_rate:.1%} success rate "
                        f"({total_successes}/{total_attempts} attempts), "
                        f"best strategy: {best_strategy}"
                    )

        return insights

    def _load_history(self):
        """Load learning history from file"""
        if not self.history_file:
            return

        try:
            history_path = Path(self.history_file)
            if history_path.exists():
                with open(history_path, 'r') as f:
                    data = json.load(f)

                # Reconstruct fix history
                for attempt_data in data.get('fix_history', []):
                    attempt = FixAttempt(
                        error_pattern=attempt_data['error_pattern'],
                        strategy_used=FixStrategy(attempt_data['strategy_used']),
                        success=attempt_data['success'],
                        time_taken=attempt_data['time_taken'],
                        tokens_used=attempt_data['tokens_used'],
                        timestamp=datetime.fromisoformat(attempt_data['timestamp'])
                    )
                    self.fix_history.append(attempt)

                # Restore stats
                self.pattern_stats = defaultdict(lambda: {
                    'attempts': 0,
                    'successes': 0,
                    'strategy_performance': defaultdict(lambda: {
                        'attempts': 0,
                        'successes': 0,
                        'avg_time': 0.0,
                        'avg_tokens': 0
                    })
                })

                for pattern_id, pattern_stats in data.get('pattern_stats', {}).items():
                    self.pattern_stats[pattern_id].update(pattern_stats)

                logger.info(f"[Learning] Loaded {len(self.fix_history)} fix attempts from history")

        except Exception as e:
            logger.error(f"[Learning] Failed to load history: {e}")

    def _save_history(self):
        """Save learning history to file"""
        if not self.history_file:
            return

        try:
            history_path = Path(self.history_file)

            # Prepare data for serialization
            data = {
                'saved_at': datetime.now().isoformat(),
                'total_attempts': len(self.fix_history),
                'fix_history': [
                    {
                        'error_pattern': attempt.error_pattern,
                        'strategy_used': attempt.strategy_used.value,
                        'success': attempt.success,
                        'time_taken': attempt.time_taken,
                        'tokens_used': attempt.tokens_used,
                        'timestamp': attempt.timestamp.isoformat()
                    }
                    for attempt in self.fix_history
                ],
                'pattern_stats': dict(self.pattern_stats)
            }

            # Save to file
            with open(history_path, 'w') as f:
                json.dump(data, f, indent=2)

            logger.debug(f"[Learning] Saved {len(self.fix_history)} fix attempts to history")

        except Exception as e:
            logger.error(f"[Learning] Failed to save history: {e}")

    def suggest_fix_with_strategy(
        self,
        code: str,
        validation_result: ValidationResult,
        llm_fixer: Optional[LLMSemanticFixer] = None
    ) -> Dict[str, Any]:
        """
        Suggest fix with optimal strategy

        Args:
            code: Code with error
            validation_result: Validation result
            llm_fixer: LLM fixer (optional)

        Returns:
            Suggestion dict with strategy and fix
        """
        # Classify error
        classified = self.classify_error(validation_result, code)

        # Select optimal strategy
        strategy = self.select_optimal_strategy(classified, llm_fixer)

        # Get strategy-specific instructions
        instructions = self._get_strategy_instructions(
            classified,
            strategy,
            code
        )

        return {
            'classified_error': classified,
            'selected_strategy': strategy,
            'instructions': instructions,
            'confidence': classified.confidence
        }

    def _get_strategy_instructions(
        self,
        classified_error: ClassifiedError,
        strategy: FixStrategy,
        code: str
    ) -> str:
        """Get detailed instructions for a strategy"""
        instructions = {
            FixStrategy.HEURISTIC_RULE: f"""
Use rule-based heuristic fix for {classified_error.original_error}:

Apply specific fix rules based on the error pattern.
This is fast and deterministic.
""",
            FixStrategy.LLM_SEMANTIC: f"""
Use LLM semantic fix for {classified_error.original_error}:

Let the LLM understand the code context and apply semantic fixes.
This is more flexible but slower.
""",
            FixStrategy.TEMPLATE_BASED: f"""
Use template-based fix for {classified_error.original_error}:

Apply predefined code templates to fix this error pattern.
Fast and predictable.
""",
            FixStrategy.HYBRID: f"""
Use hybrid approach for {classified_error.original_error}:

Combine heuristic rules with LLM semantic understanding.
Balance of speed and accuracy.
""",
            FixStrategy.SKIP: f"""
Skip this error and continue:

This error cannot be automatically fixed or is not critical.
Mark it for manual review.
"""
        }

        return instructions.get(strategy, "Unknown strategy")

    def reset_learning(self):
        """Reset learning history"""
        self.fix_history = []
        self.pattern_stats = defaultdict(lambda: {
            'attempts': 0,
            'successes': 0,
            'strategy_performance': defaultdict(lambda: {
                'attempts': 0,
                'successes': 0,
                'avg_time': 0.0,
                'avg_tokens': 0
            })
        })

        logger.info("[Learning] Reset learning history")


# Convenience functions
def create_error_classifier(**kwargs) -> ErrorClassifier:
    """
    Convenience function to create ErrorClassifier

    Args:
        **kwargs: Arguments to ErrorClassifier

    Returns:
        ErrorClassifier instance
    """
    return ErrorClassifier(**kwargs)


if __name__ == "__main__":
    print("=" * 80)
    print("Error Classification System Test")
    print("=" * 80)

    # Create classifier
    classifier = create_error_classifier(
        enable_learning=True,
        history_file=".error_learning_history.json"
    )

    # Test 1: Classify errors
    print("\n[Test 1] Error Classification")

    from validators import CodeValidator
    validator = CodeValidator()

    # Error 1: Unmatched parentheses
    code1 = "def foo():\n    return (1 + 2\n"
    result1 = validator.validate_code(code1)
    classified1 = classifier.classify_error(result1, code1)
    print(f"Error: {result1.error_type}")
    print(f"Category: {classified1.category.value}")
    print(f"Pattern: {classified1.pattern_id}")
    print(f"Confidence: {classified1.confidence:.2f}")
    print(f"Suggested strategies: {[s.value for s in classified1.suggested_strategies]}")

    # Error 2: Unterminated string
    code2 = 'def bar():\n    return "hello\n'
    result2 = validator.validate_code(code2)
    classified2 = classifier.classify_error(result2, code2)
    print(f"\nError: {result2.error_type}")
    print(f"Category: {classified2.category.value}")
    print(f"Suggested strategies: {[s.value for s in classified2.suggested_strategies]}")

    # Error 3: Parameter order
    code3 = "def baz(a, b=1, c):\n    pass\n"
    result3 = validator.validate_code(code3)
    classified3 = classifier.classify_error(result3, code3)
    print(f"\nError: {result3.error_type}")
    print(f"Category: {classified3.category.value}")
    print(f"Pattern: {classified3.pattern_id}")
    print(f"Suggested strategies: {[s.value for s in classified3.suggested_strategies]}")

    # Test 2: Strategy selection
    print("\n[Test 2] Strategy Selection")
    strategy1 = classifier.select_optimal_strategy(classified1)
    print(f"Selected strategy for unmatched_parens: {strategy1.value}")

    # Test 3: Pattern statistics
    print("\n[Test 3] Pattern Statistics (before learning)")
    stats = classifier.get_pattern_statistics()
    for pattern_id, pattern_stats in stats.items():
        print(f"\nPattern: {pattern_id}")
        print(f"  Attempts: {pattern_stats['total_attempts']}")
        print(f"  Successes: {pattern_stats['total_successes']}")
        print(f"  Success rate: {pattern_stats['overall_success_rate']:.2f}")

    # Test 4: Learning insights
    print("\n[Test 4] Learning Insights")
    insights = classifier.get_learning_insights()
    if insights:
        print("Insights from learning history:")
        for insight in insights:
            print(f"  - {insight}")
    else:
        print("No insights yet (no learning history)")

    # Cleanup
    import os
    if os.path.exists(".error_learning_history.json"):
        os.remove(".error_learning_history.json")

    print("\n" + "=" * 80)
    print("All tests completed")
    print("=" * 80)
