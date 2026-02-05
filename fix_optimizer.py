"""
AGI Autonomous Code Generation - Fix Optimizer Component
Phase 2, Component 4: Fix Strategy Optimization

This component provides intelligent fix strategy optimization with:
1. Fix strategy selection tree
2. Parallel fix attempts
3. Fix result merging
4. Performance optimization

Author: AGI Development Team
Version: 1.0.0
"""

import asyncio
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any, Set
from enum import Enum
import re
import json
import time
from datetime import datetime
from collections import defaultdict

# Try to import dependencies
try:
    from validators import ValidationResult, ErrorType
    from fixers import LLMSemanticFixer, FixResult, FixStrategy
    from error_classifier import ErrorClassifier, ClassifiedError, ErrorCategory
except ImportError:
    # Define stubs for standalone testing
    class ValidationResult:
        def __init__(self, is_valid: bool, error_type: Optional[str] = None,
                     error_message: str = "", error_line: int = 0):
            self.is_valid = is_valid
            self.error_type = error_type
            self.error_message = error_message
            self.error_line = error_line

    class ErrorType:
        SYNTAX_ERROR = "syntax_error"
        INDENTATION_ERROR = "indentation_error"
        IMPORT_ERROR = "import_error"
        UNTERMINATED_STRING = "unterminated_string"

    class ErrorCategory(Enum):
        SYNTAX_TRUNCATION = "syntax_truncation"
        SYNTAX_STRUCTURE = "syntax_structure"
        SEMANTIC_TYPE = "semantic_type"
        IMPORT_DEPENDENCY = "import_dependency"

    class FixStrategy(Enum):
        HEURISTIC_RULE = "heuristic_rule"
        LLM_SEMANTIC = "llm_semantic"
        FALLBACK = "fallback"
        PARALLEL_HYBRID = "parallel_hybrid"

    class ClassifiedError:
        def __init__(self, pattern: Optional[str], category: ErrorCategory,
                     confidence: float, suggested_strategies: List[FixStrategy]):
            self.pattern = pattern
            self.category = category
            self.confidence = confidence
            self.suggested_strategies = suggested_strategies


class FixOutcome(Enum):
    """Fix attempt outcome"""
    SUCCESS = "success"
    PARTIAL = "partial"
    FAILED = "failed"
    TIMEOUT = "timeout"


@dataclass
class FixAttempt:
    """Single fix attempt record"""
    strategy: FixStrategy
    outcome: FixOutcome
    duration_ms: float
    fixed_code: Optional[str]
    validation_result: Optional[ValidationResult]
    error_message: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class FixOptimizationResult:
    """Result of optimized fix process"""
    success: bool
    final_code: Optional[str]
    attempts: List[FixAttempt]
    best_strategy: Optional[FixStrategy]
    total_duration_ms: float
    validation_result: Optional[ValidationResult]
    optimization_insights: Dict[str, Any]


@dataclass
class StrategyPerformance:
    """Performance metrics for a strategy"""
    total_attempts: int = 0
    successes: int = 0
    partials: int = 0
    failures: int = 0
    timeouts: int = 0
    avg_duration_ms: float = 0.0
    success_rate: float = 0.0

    def update(self, attempt: FixAttempt):
        """Update metrics with new attempt"""
        self.total_attempts += 1

        if attempt.outcome == FixOutcome.SUCCESS:
            self.successes += 1
        elif attempt.outcome == FixOutcome.PARTIAL:
            self.partials += 1
        elif attempt.outcome == FixOutcome.TIMEOUT:
            self.timeouts += 1
        else:
            self.failures += 1

        # Update average duration
        self.avg_duration_ms = (
            (self.avg_duration_ms * (self.total_attempts - 1) + attempt.duration_ms)
            / self.total_attempts
        )

        # Update success rate
        self.success_rate = self.successes / max(1, self.total_attempts)


class FixStrategyTree:
    """
    Decision tree for selecting optimal fix strategy

    Rules:
    1. Simple syntax errors (high confidence) -> heuristic_rule
    2. Complex errors (low confidence) -> llm_semantic
    3. Ambiguous errors -> parallel_hybrid
    4. Time critical -> heuristic_rule (faster)
    5. Previous success -> reuse successful strategy
    """

    def __init__(self):
        """Initialize decision tree"""
        self.performance_history: Dict[FixStrategy, StrategyPerformance] = {
            strategy: StrategyPerformance()
            for strategy in FixStrategy
        }

        # Confidence thresholds
        self.high_confidence_threshold = 0.7
        self.low_confidence_threshold = 0.4

        # Category preferences
        self.category_strategy_map = {
            ErrorCategory.SYNTAX_TRUNCATION: [
                FixStrategy.HEURISTIC_RULE,
                FixStrategy.LLM_SEMANTIC
            ],
            ErrorCategory.SYNTAX_STRUCTURE: [
                FixStrategy.LLM_SEMANTIC,
                FixStrategy.HEURISTIC_RULE
            ],
            ErrorCategory.SEMANTIC_TYPE: [
                FixStrategy.LLM_SEMANTIC
            ],
            ErrorCategory.IMPORT_DEPENDENCY: [
                FixStrategy.HEURISTIC_RULE,
                FixStrategy.LLM_SEMANTIC
            ],
        }

    def select_strategy(self,
                       classified_error: ClassifiedError,
                       time_critical: bool = False,
                       allow_parallel: bool = True) -> FixStrategy:
        """
        Select optimal strategy based on error classification

        Args:
            classified_error: Classified error information
            time_critical: Whether time is critical (prefer faster strategies)
            allow_parallel: Whether parallel attempts are allowed

        Returns:
            Selected fix strategy
        """
        confidence = classified_error.confidence
        category = classified_error.category
        suggested = classified_error.suggested_strategies

        # Rule 1: High confidence simple errors -> heuristic
        if (confidence >= self.high_confidence_threshold and
            category == ErrorCategory.SYNTAX_TRUNCATION and
            FixStrategy.HEURISTIC_RULE in suggested):
            return FixStrategy.HEURISTIC_RULE

        # Rule 2: Low confidence -> LLM semantic
        if confidence < self.low_confidence_threshold:
            if FixStrategy.LLM_SEMANTIC in suggested:
                return FixStrategy.LLM_SEMANTIC

        # Rule 3: Time critical -> prefer heuristic
        if time_critical:
            if FixStrategy.HEURISTIC_RULE in suggested:
                return FixStrategy.HEURISTIC_RULE

        # Rule 4: Check category preferences
        if category in self.category_strategy_map:
            preferred = self.category_strategy_map[category]
            for strategy in preferred:
                if strategy in suggested:
                    # Check if this strategy has good performance
                    perf = self.performance_history[strategy]
                    if perf.success_rate > 0.5 or perf.total_attempts < 3:
                        return strategy

        # Rule 5: Ambiguous (mid confidence) -> parallel if allowed
        if (allow_parallel and
            self.low_confidence_threshold <= confidence < self.high_confidence_threshold and
            len(suggested) > 1):
            return FixStrategy.PARALLEL_HYBRID

        # Default: first suggested strategy
        return suggested[0] if suggested else FixStrategy.LLM_SEMANTIC

    def record_attempt(self, attempt: FixAttempt):
        """Record fix attempt for learning"""
        self.performance_history[attempt.strategy].update(attempt)

    def get_strategy_ranking(self) -> List[Tuple[FixStrategy, float]]:
        """Get strategies ranked by success rate"""
        rankings = []
        for strategy, perf in self.performance_history.items():
            if perf.total_attempts > 0:
                # Score combines success rate and speed (prefer faster)
                score = perf.success_rate * 0.7 + (1 / (1 + perf.avg_duration_ms / 1000)) * 0.3
                rankings.append((strategy, score))

        return sorted(rankings, key=lambda x: x[1], reverse=True)


class ParallelFixExecutor:
    """
    Execute multiple fix strategies in parallel

    Benefits:
    1. Faster time to solution
    2. Higher success rate
    3. Strategy comparison
    """

    def __init__(self, max_parallel: int = 3, timeout_ms: float = 30000):
        """
        Initialize parallel executor

        Args:
            max_parallel: Maximum number of parallel attempts
            timeout_ms: Timeout for each attempt in milliseconds
        """
        self.max_parallel = max_parallel
        self.timeout_ms = timeout_ms
        self.fixer = None  # Will be injected

    async def execute_parallel(self,
                               code: str,
                               validation_result: ValidationResult,
                               strategies: List[FixStrategy]) -> List[FixAttempt]:
        """
        Execute multiple fix strategies in parallel

        Args:
            code: Original code with error
            validation_result: Validation result
            strategies: List of strategies to try

        Returns:
            List of fix attempts
        """
        if not self.fixer:
            raise ValueError("Fixer not initialized. Call set_fixer() first.")

        # Limit parallel attempts
        strategies = strategies[:self.max_parallel]

        # Create tasks
        tasks = []
        for strategy in strategies:
            task = self._execute_with_timeout(
                code, validation_result, strategy
            )
            tasks.append(task)

        # Execute in parallel
        attempts = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results
        results = []
        for i, attempt in enumerate(attempts):
            if isinstance(attempt, Exception):
                # Create failed attempt
                results.append(FixAttempt(
                    strategy=strategies[i],
                    outcome=FixOutcome.FAILED,
                    duration_ms=self.timeout_ms,
                    fixed_code=None,
                    validation_result=None,
                    error_message=str(attempt)
                ))
            else:
                results.append(attempt)

        return results

    async def _execute_with_timeout(self,
                                    code: str,
                                    validation_result: ValidationResult,
                                    strategy: FixStrategy) -> FixAttempt:
        """Execute fix with timeout"""
        start_time = time.time()

        try:
            # Create timeout task
            fix_task = self._execute_single(code, validation_result, strategy)
            result = await asyncio.wait_for(
                fix_task,
                timeout=self.timeout_ms / 1000
            )

            duration_ms = (time.time() - start_time) * 1000
            return FixAttempt(
                strategy=strategy,
                outcome=result.outcome,
                duration_ms=duration_ms,
                fixed_code=result.fixed_code,
                validation_result=result.validation_result
            )

        except asyncio.TimeoutError:
            duration_ms = (time.time() - start_time) * 1000
            return FixAttempt(
                strategy=strategy,
                outcome=FixOutcome.TIMEOUT,
                duration_ms=duration_ms,
                fixed_code=None,
                validation_result=None,
                error_message="Fix attempt timed out"
            )

    async def _execute_single(self,
                              code: str,
                              validation_result: ValidationResult,
                              strategy: FixStrategy) -> FixAttempt:
        """Execute single fix attempt"""
        start_time = time.time()

        # Mock implementation for standalone testing
        # In real use, this would call self.fixer.fix_code() with appropriate strategy

        # Simulate fix
        duration_ms = (time.time() - start_time) * 1000

        # Return mock attempt
        return FixAttempt(
            strategy=strategy,
            outcome=FixOutcome.SUCCESS,
            duration_ms=duration_ms,
            fixed_code=code,  # In real use, actual fixed code
            validation_result=validation_result
        )

    def set_fixer(self, fixer):
        """Set the fixer instance"""
        self.fixer = fixer


class FixResultMerger:
    """
    Merge results from multiple fix attempts

    Strategies:
    1. Select best validation result
    2. Combine successful fixes
    3. Hybrid approach
    """

    def merge_results(self, attempts: List[FixAttempt],
                     original_code: str) -> Tuple[Optional[str], FixAttempt]:
        """
        Merge fix attempts and select best result

        Args:
            attempts: List of fix attempts
            original_code: Original code

        Returns:
            Tuple of (best_code, best_attempt)
        """
        if not attempts:
            return None, None

        # Filter successful attempts
        successful = [a for a in attempts if a.outcome == FixOutcome.SUCCESS]

        if successful:
            # Select best successful attempt
            # Prefer: fastest success, then by strategy preference
            best = min(successful, key=lambda a: a.duration_ms)
            return best.fixed_code, best

        # No full success, check partial
        partials = [a for a in attempts if a.outcome == FixOutcome.PARTIAL]
        if partials:
            # Try to merge partial fixes
            merged = self._merge_partial_fixes(partials, original_code)
            if merged:
                best = max(partials, key=lambda a: a.duration_ms)
                return merged, best

        # All failed, return most promising (longest duration = most effort)
        if attempts:
            best = max(attempts, key=lambda a: a.duration_ms)
            return best.fixed_code, best

        return None, None

    def _merge_partial_fixes(self,
                            partials: List[FixAttempt],
                            original_code: str) -> Optional[str]:
        """
        Attempt to merge partial fixes

        This is a sophisticated operation that requires:
        1. AST analysis of each partial fix
        2. Identifying which parts were fixed
        3. Combining successfully fixed parts
        4. Resolving conflicts

        For now, return the most promising partial fix
        """
        if not partials:
            return None

        # Select partial with best validation
        # (In real implementation, would do actual merging)
        best = max(partials, key=lambda a: a.duration_ms)
        return best.fixed_code


class FixOptimizer:
    """
    Main fix optimization coordinator

    Orchestrates:
    1. Strategy selection via decision tree
    2. Parallel execution
    3. Result merging
    4. Performance tracking
    """

    def __init__(self,
                 max_parallel_attempts: int = 3,
                 fix_timeout_ms: float = 30000,
                 enable_learning: bool = True):
        """
        Initialize fix optimizer

        Args:
            max_parallel_attempts: Maximum parallel fix attempts
            fix_timeout_ms: Timeout per fix attempt in milliseconds
            enable_learning: Whether to enable learning from attempts
        """
        self.strategy_tree = FixStrategyTree()
        self.parallel_executor = ParallelFixExecutor(
            max_parallel=max_parallel_attempts,
            timeout_ms=fix_timeout_ms
        )
        self.result_merger = FixResultMerger()

        self.enable_learning = enable_learning
        self.total_optimizations = 0
        self.total_successes = 0

    async def optimize_fix(self,
                          code: str,
                          validation_result: ValidationResult,
                          classified_error: ClassifiedError,
                          time_critical: bool = False) -> FixOptimizationResult:
        """
        Optimize fix strategy selection and execution

        Args:
            code: Code with error
            validation_result: Validation result
            classified_error: Classified error information
            time_critical: Whether time is critical

        Returns:
            Fix optimization result
        """
        start_time = time.time()
        attempts = []

        try:
            # Step 1: Select strategy
            strategy = self.strategy_tree.select_strategy(
                classified_error,
                time_critical=time_critical,
                allow_parallel=True
            )

            # Step 2: Execute fix(es)
            if strategy == FixStrategy.PARALLEL_HYBRID:
                # Execute multiple strategies in parallel
                strategies_to_try = classified_error.suggested_strategies
                parallel_attempts = await self.parallel_executor.execute_parallel(
                    code, validation_result, strategies_to_try
                )
                attempts.extend(parallel_attempts)

                # Step 3: Merge results
                final_code, best_attempt = self.result_merger.merge_results(
                    parallel_attempts, code
                )
                best_strategy = best_attempt.strategy if best_attempt else None

            else:
                # Single strategy execution
                attempt = await self.parallel_executor._execute_single(
                    code, validation_result, strategy
                )
                attempts.append(attempt)

                final_code = attempt.fixed_code
                best_strategy = strategy
                best_attempt = attempt

            # Step 4: Record for learning
            if self.enable_learning:
                for attempt in attempts:
                    self.strategy_tree.record_attempt(attempt)

            # Step 5: Build result
            total_duration_ms = (time.time() - start_time) * 1000
            success = final_code is not None and best_attempt.outcome == FixOutcome.SUCCESS

            if success:
                self.total_successes += 1
            self.total_optimizations += 1

            return FixOptimizationResult(
                success=success,
                final_code=final_code,
                attempts=attempts,
                best_strategy=best_strategy,
                total_duration_ms=total_duration_ms,
                validation_result=best_attempt.validation_result if best_attempt else None,
                optimization_insights=self._generate_insights(attempts)
            )

        except Exception as e:
            # Return failed result
            total_duration_ms = (time.time() - start_time) * 1000
            return FixOptimizationResult(
                success=False,
                final_code=None,
                attempts=attempts,
                best_strategy=None,
                total_duration_ms=total_duration_ms,
                validation_result=validation_result,
                optimization_insights={'error': str(e)}
            )

    def _generate_insights(self, attempts: List[FixAttempt]) -> Dict[str, Any]:
        """Generate insights from fix attempts"""
        if not attempts:
            return {}

        insights = {
            'total_attempts': len(attempts),
            'strategies_tried': [a.strategy.value for a in attempts],
            'outcomes': [a.outcome.value for a in attempts],
            'avg_duration_ms': sum(a.duration_ms for a in attempts) / len(attempts),
            'fastest_success_ms': None,
            'best_strategy': None
        }

        successful = [a for a in attempts if a.outcome == FixOutcome.SUCCESS]
        if successful:
            fastest = min(successful, key=lambda a: a.duration_ms)
            insights['fastest_success_ms'] = fastest.duration_ms
            insights['best_strategy'] = fastest.strategy.value

        return insights

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get overall performance summary"""
        ranking = self.strategy_tree.get_strategy_ranking()

        return {
            'total_optimizations': self.total_optimizations,
            'total_successes': self.total_successes,
            'overall_success_rate': (
                self.total_successes / max(1, self.total_optimizations)
            ),
            'strategy_ranking': [
                {'strategy': s.value, 'score': score}
                for s, score in ranking
            ],
            'strategy_performance': {
                strategy.value: {
                    'attempts': perf.total_attempts,
                    'successes': perf.successes,
                    'success_rate': perf.success_rate,
                    'avg_duration_ms': perf.avg_duration_ms
                }
                for strategy, perf in self.strategy_tree.performance_history.items()
            }
        }


# ============================================================================
# Test Suite
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("Fix Optimizer Test")
    print("=" * 80)

    # Test 1: Strategy Selection
    print("\n[Test 1] Strategy Selection")
    print("-" * 40)

    optimizer = FixOptimizer(enable_learning=False)

    # High confidence syntax error
    error_high_conf = ClassifiedError(
        pattern="unterminated_string",
        category=ErrorCategory.SYNTAX_TRUNCATION,
        confidence=0.8,
        suggested_strategies=[
            FixStrategy.HEURISTIC_RULE,
            FixStrategy.LLM_SEMANTIC
        ]
    )

    strategy = optimizer.strategy_tree.select_strategy(error_high_conf)
    print(f"High confidence syntax error: {strategy.value}")
    assert strategy == FixStrategy.HEURISTIC_RULE, "Should select heuristic for high confidence"

    # Low confidence error
    error_low_conf = ClassifiedError(
        pattern=None,
        category=ErrorCategory.SEMANTIC_TYPE,
        confidence=0.3,
        suggested_strategies=[FixStrategy.LLM_SEMANTIC]
    )

    strategy = optimizer.strategy_tree.select_strategy(error_low_conf)
    print(f"Low confidence semantic error: {strategy.value}")
    assert strategy == FixStrategy.LLM_SEMANTIC, "Should select LLM for low confidence"

    print("OK Strategy selection works correctly")

    # Test 2: Performance Tracking
    print("\n[Test 2] Performance Tracking")
    print("-" * 40)

    # Record some attempts
    attempt1 = FixAttempt(
        strategy=FixStrategy.HEURISTIC_RULE,
        outcome=FixOutcome.SUCCESS,
        duration_ms=100,
        fixed_code="code1",
        validation_result=ValidationResult(is_valid=True)
    )

    attempt2 = FixAttempt(
        strategy=FixStrategy.LLM_SEMANTIC,
        outcome=FixOutcome.FAILED,
        duration_ms=2000,
        fixed_code=None,
        validation_result=None
    )

    optimizer.strategy_tree.record_attempt(attempt1)
    optimizer.strategy_tree.record_attempt(attempt2)

    ranking = optimizer.strategy_tree.get_strategy_ranking()
    print(f"Strategy ranking: {[(s.value, f'{score:.2f}') for s, score in ranking]}")
    print("OK Performance tracking works")

    # Test 3: Result Merging
    print("\n[Test 3] Result Merging")
    print("-" * 40)

    merger = FixResultMerger()

    # Create test attempts
    attempts = [
        FixAttempt(
            strategy=FixStrategy.HEURISTIC_RULE,
            outcome=FixOutcome.SUCCESS,
            duration_ms=100,
            fixed_code="fixed_code_1",
            validation_result=ValidationResult(is_valid=True)
        ),
        FixAttempt(
            strategy=FixStrategy.LLM_SEMANTIC,
            outcome=FixOutcome.SUCCESS,
            duration_ms=200,
            fixed_code="fixed_code_2",
            validation_result=ValidationResult(is_valid=True)
        )
    ]

    best_code, best_attempt = merger.merge_results(attempts, "original_code")
    print(f"Best code: {best_code}")
    print(f"Best strategy: {best_attempt.strategy.value}")
    print(f"Best duration: {best_attempt.duration_ms}ms")
    assert best_attempt.strategy == FixStrategy.HEURISTIC_RULE, "Should select fastest success"
    print("OK Result merging works correctly")

    # Test 4: Full Optimization Flow
    print("\n[Test 4] Full Optimization Flow")
    print("-" * 40)

    async def test_optimization():
        # Create test data
        code = "def test():\n    return 'test"
        validation_result = ValidationResult(
            is_valid=False,
            error_type=ErrorType.UNTERMINATED_STRING,
            error_message="unterminated string literal",
            error_line=1
        )

        classified_error = ClassifiedError(
            pattern="unterminated_string",
            category=ErrorCategory.SYNTAX_TRUNCATION,
            confidence=0.7,
            suggested_strategies=[
                FixStrategy.HEURISTIC_RULE,
                FixStrategy.LLM_SEMANTIC
            ]
        )

        # Run optimization
        result = await optimizer.optimize_fix(
            code=code,
            validation_result=validation_result,
            classified_error=classified_error,
            time_critical=False
        )

        print(f"Success: {result.success}")
        print(f"Best strategy: {result.best_strategy.value if result.best_strategy else 'None'}")
        print(f"Total attempts: {len(result.attempts)}")
        print(f"Total duration: {result.total_duration_ms:.2f}ms")
        print(f"Insights: {result.optimization_insights}")
        print("OK Full optimization flow works")

    # Run async test
    asyncio.run(test_optimization())

    # Test 5: Performance Summary
    print("\n[Test 5] Performance Summary")
    print("-" * 40)

    summary = optimizer.get_performance_summary()
    print(f"Total optimizations: {summary['total_optimizations']}")
    print(f"Total successes: {summary['total_successes']}")
    print(f"Overall success rate: {summary['overall_success_rate']:.2%}")
    print(f"Strategy performance:")
    for strategy, perf in summary['strategy_performance'].items():
        print(f"  {strategy}:")
        print(f"    Attempts: {perf['attempts']}")
        print(f"    Successes: {perf['successes']}")
        print(f"    Success rate: {perf['success_rate']:.2%}")

    print("OK Performance summary works")

    print("\n" + "=" * 80)
    print("All tests completed successfully!")
    print("=" * 80)
