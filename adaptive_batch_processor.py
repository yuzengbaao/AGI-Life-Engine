#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Adaptive Batch Processor - V6.1.2 Phase 2

Features:
1. Dynamic batch size calculation
2. Method complexity estimation
3. Token usage monitoring
4. Success rate tracking
5. Adaptive adjustment

Author: AGI System Enhancement
Date: 2026-02-05
"""

import logging
import re
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from collections import deque
from datetime import datetime

from token_budget import TokenBudget

logger = logging.getLogger(__name__)


@dataclass
class MethodComplexity:
    """Method complexity metrics"""
    name: str
    line_count: int
    cyclomatic_complexity: int  # McCabe complexity
    nesting_depth: int
    parameter_count: int
    has_loops: bool
    has_exceptions: bool
    is_async: bool
    estimated_tokens: int
    overall_score: int  # 1-10


@dataclass
class BatchResult:
    """Result of a batch generation"""
    batch_number: int
    batch_size: int
    methods_in_batch: List[str]
    success: bool
    tokens_used: int
    time_taken: float
    error_message: Optional[str]
    truncation_detected: bool


@dataclass
class ProcessorState:
    """Current processor state"""
    current_batch_size: int
    success_rate: float  # Exponential moving average
    avg_token_usage: float
    total_batches_processed: int
    recent_results: deque  # Last N batch results
    adjustment_history: List[Dict]


class AdaptiveBatchProcessor:
    """
    Adaptive Batch Processor - V6.1.2

    Dynamically adjusts batch size based on:
    1. Method complexity
    2. Token usage history
    3. Success rate
    4. Performance metrics

    Goals:
    - Maximize throughput (larger batches when possible)
    - Minimize failures (smaller batches for complex methods)
    - Optimize token usage (avoid truncation)
    """

    def __init__(
        self,
        token_budget: TokenBudget,
        initial_batch_size: int = 3,
        min_batch_size: int = 1,
        max_batch_size: int = 5,
        complexity_threshold: int = 5,
        token_usage_threshold: float = 0.8,
        success_rate_threshold: float = 0.7,
        window_size: int = 10,
        alpha: float = 0.3  # EMA smoothing factor
    ):
        """
        Initialize Adaptive Batch Processor

        Args:
            token_budget: TokenBudget instance
            initial_batch_size: Starting batch size
            min_batch_size: Minimum batch size
            max_batch_size: Maximum batch size
            complexity_threshold: Complexity threshold (1-10)
            token_usage_threshold: Token usage warning threshold (0.0-1.0)
            success_rate_threshold: Success rate threshold (0.0-1.0)
            window_size: Size of sliding window for metrics
            alpha: Exponential moving average smoothing factor
        """
        self.token_budget = token_budget

        # Configuration
        self.min_batch_size = min_batch_size
        self.max_batch_size = max_batch_size
        self.complexity_threshold = complexity_threshold
        self.token_usage_threshold = token_usage_threshold
        self.success_rate_threshold = success_rate_threshold
        self.alpha = alpha

        # State
        self.state = ProcessorState(
            current_batch_size=initial_batch_size,
            success_rate=1.0,
            avg_token_usage=0.5,
            total_batches_processed=0,
            recent_results=deque(maxlen=window_size),
            adjustment_history=[]
        )

        logger.info(
            f"[AdaptiveBatchProcessor] Initialized: "
            f"batch_size={initial_batch_size} (range: {min_batch_size}-{max_batch_size}), "
            f"complexity_threshold={complexity_threshold}"
        )

    def calculate_optimal_batch_size(
        self,
        methods: List[str],
        context: Optional[Dict] = None
    ) -> int:
        """
        Calculate optimal batch size for given methods

        Considers:
        1. Method complexity
        2. Token usage history
        3. Recent success rate
        4. Available token budget

        Args:
            methods: List of method names to implement
            context: Optional context (code so far, etc.)

        Returns:
            Optimal batch size (1 to max_batch_size)
        """
        if not methods:
            return self.state.current_batch_size

        # Estimate complexity for each method
        complexities = []
        for method_name in methods:
            complexity = self._estimate_method_complexity(method_name, context)
            complexities.append(complexity)

        # Calculate average complexity
        avg_complexity = sum(c.overall_score for c in complexities) / len(complexities)

        # Factor 1: Complexity-based adjustment
        if avg_complexity > 8:  # Very complex
            complexity_factor = 0.3
            recommended_size = 1
        elif avg_complexity > 5:  # Moderately complex
            complexity_factor = 0.6
            recommended_size = max(1, self.state.current_batch_size - 1)
        elif avg_complexity < 3:  # Simple
            complexity_factor = 1.0
            recommended_size = min(self.max_batch_size, self.state.current_batch_size + 1)
        else:  # Average
            complexity_factor = 0.8
            recommended_size = self.state.current_batch_size

        # Factor 2: Token usage history
        if self.state.total_batches_processed > 0:
            if self.state.avg_token_usage > self.token_usage_threshold:
                # Token usage too high, reduce batch size
                token_factor = 0.7
                recommended_size = max(self.min_batch_size, recommended_size - 1)
            elif self.state.avg_token_usage < 0.6:
                # Plenty of token room, can increase
                token_factor = 1.0
                recommended_size = min(self.max_batch_size, recommended_size + 1)
            else:
                token_factor = 0.9
        else:
            token_factor = 1.0

        # Factor 3: Success rate
        if self.state.total_batches_processed > 0:
            if self.state.success_rate < self.success_rate_threshold:
                # Low success rate, be conservative
                success_factor = 0.7
                recommended_size = max(self.min_batch_size, recommended_size - 1)
            elif self.state.success_rate > 0.9:
                # High success rate, can be aggressive
                success_factor = 1.0
                recommended_size = min(self.max_batch_size, recommended_size + 1)
            else:
                success_factor = 0.9
        else:
            success_factor = 1.0

        # Combine factors and apply
        final_size = int(recommended_size)

        # Ensure within bounds
        final_size = max(self.min_batch_size, min(self.max_batch_size, final_size))

        logger.info(
            f"[BatchSize] Calculated: {final_size} "
            f"(avg_complexity={avg_complexity:.1f}, "
            f"token_usage={self.state.avg_token_usage:.2f}, "
            f"success_rate={self.state.success_rate:.2f})"
        )

        return final_size

    def _estimate_method_complexity(
        self,
        method_name: str,
        context: Optional[Dict]
    ) -> MethodComplexity:
        """
        Estimate method complexity

        Heuristics:
        1. Name-based estimation (if no code)
        2. Code-based estimation (if code available)
        3. Context-based estimation

        Args:
            method_name: Name of the method
            context: Optional context with code

        Returns:
            MethodComplexity object
        """
        # Default values
        complexity = MethodComplexity(
            name=method_name,
            line_count=10,
            cyclomatic_complexity=1,
            nesting_depth=1,
            parameter_count=2,
            has_loops=False,
            has_exceptions=False,
            is_async=False,
            estimated_tokens=150,
            overall_score=3
        )

        # If code is available, analyze it
        if context and 'code' in context:
            code = context['code']
            complexity = self._analyze_code_complexity(method_name, code)
        else:
            # Name-based heuristics
            complexity = self._estimate_from_name(method_name)

        return complexity

    def _analyze_code_complexity(
        self,
        method_name: str,
        code: str
    ) -> MethodComplexity:
        """
        Analyze code complexity from actual code

        Args:
            method_name: Method name
            code: Method code

        Returns:
            MethodComplexity object
        """
        lines = code.split('\n')
        line_count = len(lines)

        # McCabe complexity (approximate)
        cyclomatic = 1
        cyclomatic += code.count('if ')
        cyclomatic += code.count('elif ')
        cyclomatic += code.count('for ')
        cyclomatic += code.count('while ')
        cyclomatic += code.count('except ')
        cyclomatic += code.count('and ') + code.count(' or ')

        # Nesting depth
        nesting_depth = 0
        for line in lines:
            stripped = line.lstrip()
            indent = len(line) - len(stripped)
            current_depth = indent // 4
            nesting_depth = max(nesting_depth, current_depth)

        # Parameter count
        param_match = re.search(r'def\s+\w+\s*\((.*?)\):', code)
        if param_match:
            params = param_match.group(1)
            parameter_count = len([p.strip() for p in params.split(',') if p.strip()])
        else:
            parameter_count = 2

        # Feature detection
        has_loops = 'for ' in code or 'while ' in code
        has_exceptions = 'try:' in code or 'except ' in code
        is_async = 'async def ' in code

        # Estimate tokens (rough estimate: ~5 tokens per line)
        estimated_tokens = line_count * 5

        # Calculate overall score (1-10)
        score = 1  # Base score

        # Line count contribution
        score += min(3, line_count // 10)

        # Complexity contribution
        score += min(2, cyclomatic // 2)

        # Nesting contribution
        score += min(2, nesting_depth // 2)

        # Feature contributions
        if has_loops:
            score += 1
        if has_exceptions:
            score += 1
        if is_async:
            score += 2

        score = min(10, score)

        return MethodComplexity(
            name=method_name,
            line_count=line_count,
            cyclomatic_complexity=cyclomatic,
            nesting_depth=nesting_depth,
            parameter_count=parameter_count,
            has_loops=has_loops,
            has_exceptions=has_exceptions,
            is_async=is_async,
            estimated_tokens=estimated_tokens,
            overall_score=score
        )

    def _estimate_from_name(self, method_name: str) -> MethodComplexity:
        """
        Estimate complexity from method name only

        Heuristics based on common patterns:
        - get_* methods: simple
        - set_* methods: simple
        - is_* methods: simple
        - calculate_* methods: moderate
        - process_* methods: moderate
        - analyze_* methods: complex

        Args:
            method_name: Method name

        Returns:
            MethodComplexity object
        """
        score = 3  # Default moderate

        # Simple patterns
        if method_name.startswith(('get_', 'fetch_', 'read_', 'is_', 'has_')):
            score = 2
        elif method_name.startswith(('set_', 'update_', 'write_', 'save_')):
            score = 2
        elif method_name.startswith(('add_', 'remove_', 'delete_', 'create_')):
            score = 3
        elif method_name.startswith(('calculate_', 'compute_', 'validate_')):
            score = 5
        elif method_name.startswith(('process_', 'handle_', 'transform_')):
            score = 6
        elif method_name.startswith(('analyze_', 'optimize_', 'generate_')):
            score = 7
        elif method_name.endswith(('_async', '_asyncronous')):
            score = 8

        return MethodComplexity(
            name=method_name,
            line_count=10,
            cyclomatic_complexity=max(1, score),
            nesting_depth=1,
            parameter_count=2,
            has_loops=score > 5,
            has_exceptions=score > 6,
            is_async='async' in method_name.lower(),
            estimated_tokens=score * 50,
            overall_score=score
        )

    def record_batch_result(self, result: BatchResult):
        """
        Record batch result and update state

        Args:
            result: BatchResult to record
        """
        # Add to recent results
        self.state.recent_results.append(result)
        self.state.total_batches_processed += 1

        # Update success rate (EMA)
        if result.success:
            new_success = 1.0
        else:
            new_success = 0.0

        self.state.success_rate = (
            self.alpha * new_success +
            (1 - self.alpha) * self.state.success_rate
        )

        # Update average token usage (EMA)
        if result.tokens_used > 0:
            # Normalize token usage (assuming 8000 max)
            normalized_usage = result.tokens_used / 8000.0

            self.state.avg_token_usage = (
                self.alpha * normalized_usage +
                (1 - self.alpha) * self.state.avg_token_usage
            )

        # Record adjustment if size changed
        old_size = self.state.current_batch_size
        new_size = self._calculate_next_batch_size()
        self.state.current_batch_size = new_size

        if old_size != new_size:
            adjustment = {
                'timestamp': datetime.now().isoformat(),
                'batch_number': result.batch_number,
                'old_size': old_size,
                'new_size': new_size,
                'reason': self._get_adjustment_reason(result),
                'metrics': {
                    'success_rate': self.state.success_rate,
                    'token_usage': self.state.avg_token_usage
                }
            }
            self.state.adjustment_history.append(adjustment)

            logger.info(
                f"[Adjustment] Batch size: {old_size} → {new_size} "
                f"({adjustment['reason']})"
            )

    def _calculate_next_batch_size(self) -> int:
        """
        Calculate next batch size based on current state

        Returns:
            Recommended batch size
        """
        current_size = self.state.current_batch_size

        # If no batches processed yet, keep current size
        if self.state.total_batches_processed == 0:
            return current_size

        # Get recent success rate
        recent_success = self.state.success_rate
        recent_token_usage = self.state.avg_token_usage

        # Determine direction
        if recent_success < self.success_rate_threshold:
            # Low success, decrease
            new_size = max(self.min_batch_size, current_size - 1)
        elif recent_success > 0.9 and recent_token_usage < 0.7:
            # High success and low token usage, increase
            new_size = min(self.max_batch_size, current_size + 1)
        elif recent_token_usage > self.token_usage_threshold:
            # High token usage, decrease
            new_size = max(self.min_batch_size, current_size - 1)
        else:
            # Keep current
            new_size = current_size

        return new_size

    def _get_adjustment_reason(self, result: BatchResult) -> str:
        """Get human-readable reason for batch size adjustment"""
        reasons = []

        if not result.success:
            reasons.append("batch_failed")

        if result.tokens_used / 8000.0 > self.token_usage_threshold:
            reasons.append("high_token_usage")

        if self.state.success_rate < self.success_rate_threshold:
            reasons.append("low_success_rate")

        return " + ".join(reasons) if reasons else "optimal"

    def get_performance_summary(self) -> Dict[str, Any]:
        """
        Get performance summary statistics

        Returns:
            Performance summary dict
        """
        if self.state.total_batches_processed == 0:
            return {
                'status': 'no_data',
                'message': 'No batches processed yet'
            }

        # Calculate statistics from recent results
        total_batches = len(self.state.recent_results)
        successful_batches = sum(1 for r in self.state.recent_results if r.success)
        failed_batches = total_batches - successful_batches

        with_truncation = sum(1 for r in self.state.recent_results if r.truncation_detected)

        avg_time = sum(r.time_taken for r in self.state.recent_results) / total_batches
        avg_tokens = sum(r.tokens_used for r in self.state.recent_results) / total_batches

        return {
            'status': 'active',
            'total_batches_processed': self.state.total_batches_processed,
            'recent_batches': total_batches,
            'success_rate': self.state.success_rate,
            'successful_batches': successful_batches,
            'failed_batches': failed_batches,
            'batches_with_truncation': with_truncation,
            'avg_time_per_batch': avg_time,
            'avg_tokens_per_batch': avg_tokens,
            'avg_token_usage_ratio': avg_tokens / 8000.0,
            'current_batch_size': self.state.current_batch_size,
            'adjustments_made': len(self.state.adjustment_history),
            'recommendation': self._get_recommendation()
        }

    def _get_recommendation(self) -> str:
        """Get recommendation based on current state"""
        if self.state.success_rate < 0.5:
            return "Consider reducing batch size due to low success rate"
        elif self.state.avg_token_usage > 0.9:
            return "Consider reducing batch size due to high token usage"
        elif self.state.success_rate > 0.9 and self.state.avg_token_usage < 0.6:
            return "Consider increasing batch size (high success rate and low token usage)"
        else:
            return "Current batch size is optimal"

    def reset_state(self):
        """Reset processor state (for new project)"""
        self.state = ProcessorState(
            current_batch_size=3,
            success_rate=1.0,
            avg_token_usage=0.5,
            total_batches_processed=0,
            recent_results=deque(maxlen=10),
            adjustment_history=[]
        )
        logger.info("[AdaptiveBatchProcessor] State reset")


class BatchScheduler:
    """
    Batch Scheduler - Plans method implementation in batches

    Works with AdaptiveBatchProcessor to determine:
    1. Which methods to implement in each batch
    2. Optimal batch ordering
    3. Dependencies between methods
    """

    def __init__(
        self,
        processor: AdaptiveBatchProcessor,
        context: Optional[Dict] = None
    ):
        """
        Initialize Batch Scheduler

        Args:
            processor: AdaptiveBatchProcessor instance
            context: Optional context information
        """
        self.processor = processor
        self.context = context or {}
        logger.info("[BatchScheduler] Initialized")

    def schedule_methods(
        self,
        methods: List[str],
        available_context: Optional[Dict] = None
    ) -> List[List[str]]:
        """
        Schedule methods into batches

        Args:
            methods: List of method names to implement
            available_context: Code generated so far

        Returns:
            List of batches, where each batch is a list of method names
        """
        if not methods:
            return []

        # Update context
        if available_context:
            self.context.update(available_context)

        batches = []
        remaining_methods = methods.copy()

        batch_number = 1

        while remaining_methods:
            # Calculate optimal batch size
            batch_size = self.processor.calculate_optimal_batch_size(
                remaining_methods,
                self.context
            )

            # Take next batch
            current_batch = remaining_methods[:batch_size]
            remaining_methods = remaining_methods[batch_size:]

            batches.append(current_batch)

            logger.info(
                f"[Schedule] Batch {batch_number}: {len(current_batch)} methods - {current_batch}"
            )

            batch_number += 1

        logger.info(f"[Schedule] Total batches created: {len(batches)}")

        return batches


# Convenience functions
def create_adaptive_processor(**kwargs) -> AdaptiveBatchProcessor:
    """
    Convenience function to create AdaptiveBatchProcessor

    Args:
        **kwargs: Arguments to AdaptiveBatchProcessor

    Returns:
        AdaptiveBatchProcessor instance
    """
    token_budget = TokenBudget()
    return AdaptiveBatchProcessor(token_budget, **kwargs)


if __name__ == "__main__":
    import asyncio

    print("=" * 80)
    print("Adaptive Batch Processor Test")
    print("=" * 80)

    # Create processor
    processor = create_adaptive_processor(
        initial_batch_size=3,
        min_batch_size=1,
        max_batch_size=5
    )

    # Test 1: Calculate batch size for simple methods
    print("\n[Test 1] Simple methods")
    simple_methods = ['get_data', 'set_value', 'is_valid', 'has_permission']
    batch_size_1 = processor.calculate_optimal_batch_size(simple_methods)
    print(f"Methods: {simple_methods}")
    print(f"Calculated batch size: {batch_size_1}")
    print(f"Current batch size: {processor.state.current_batch_size}")

    # Test 2: Calculate batch size for complex methods
    print("\n[Test 2] Complex methods")
    complex_methods = ['analyze_data_patterns', 'optimize_performance', 'generate_report_from_data']
    batch_size_2 = processor.calculate_optimal_batch_size(complex_methods)
    print(f"Methods: {complex_methods}")
    print(f"Calculated batch size: {batch_size_2}")

    # Test 3: Record batch results
    print("\n[Test 3] Recording batch results")

    # Simulate successful batch
    result1 = BatchResult(
        batch_number=1,
        batch_size=3,
        methods_in_batch=simple_methods[:3],
        success=True,
        tokens_used=5000,
        time_taken=2.5,
        error_message=None,
        truncation_detected=False
    )
    processor.record_batch_result(result1)
    print(f"Recorded successful batch")
    print(f"Success rate: {processor.state.success_rate:.2f}")
    print(f"Token usage: {processor.state.avg_token_usage:.2f}")

    # Simulate failed batch
    result2 = BatchResult(
        batch_number=2,
        batch_size=3,
        methods_in_batch=complex_methods[:3],
        success=False,
        tokens_used=7800,
        time_taken=3.0,
        error_message="Token limit exceeded",
        truncation_detected=True
    )
    processor.record_batch_result(result2)
    print(f"Recorded failed batch")
    print(f"Success rate: {processor.state.success_rate:.2f}")
    print(f"New batch size: {processor.state.current_batch_size}")

    # Test 4: Performance summary
    print("\n[Test 4] Performance summary")
    summary = processor.get_performance_summary()
    print(f"Summary: {summary}")

    # Test 5: Scheduling
    print("\n[Test 5] Batch scheduling")
    scheduler = BatchScheduler(processor)
    methods_to_schedule = ['get_' + str(i) for i in range(10)]
    batches = scheduler.schedule_methods(methods_to_schedule)
    print(f"Methods to schedule: {len(methods_to_schedule)}")
    print(f"Batches created: {len(batches)}")
    for i, batch in enumerate(batches, 1):
        print(f"  Batch {i}: {len(batch)} methods - {batch[:3]}...")

    print("\n" + "=" * 80)
    print("All tests completed")
    print("=" * 80)
