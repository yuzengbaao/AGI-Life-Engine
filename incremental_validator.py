#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Incremental Validation System - V6.1.2 Phase 2 Component 2

Features:
1. Validate after each batch (not just at the end)
2. Rollback mechanism for failed batches
3. Intermediate state saving
4. Incremental code construction
5. Progress tracking and recovery

Author: AGI System Enhancement
Date: 2026-02-05
"""

import ast
import logging
import os
import json
import hashlib
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from enum import Enum

from validators import CodeValidator, ValidationResult
from adaptive_batch_processor import BatchResult

logger = logging.getLogger(__name__)


class ValidationCheckpoint(Enum):
    """Checkpoint types"""
    BEFORE_BATCH = "before_batch"
    AFTER_BATCH = "after_batch"
    ON_ERROR = "on_error"
    ON_RECOVERY = "on_recovery"


@dataclass
class IncrementalState:
    """State of incremental generation"""
    current_code: str
    completed_batches: int
    total_batches: int
    methods_completed: List[str]
    methods_pending: List[str]
    last_checkpoint: Optional[str]
    checkpoint_history: List[Dict] = field(default_factory=list)
    errors_encountered: List[Dict] = field(default_factory=list)
    rollback_count: int = 0


@dataclass
class BatchGenerationResult:
    """Result of batch generation"""
    batch_number: int
    success: bool
    code_before: str
    code_after: str
    methods_added: List[str]
    validation_result: Optional[ValidationResult]
    was_rolled_back: bool = False
    error_message: Optional[str] = None
    time_taken: float = 0.0


@dataclass
class IncrementalBuildResult:
    """Final result of incremental build"""
    success: bool
    final_code: str
    total_batches: int
    completed_batches: int
    methods_completed: List[str]
    validation_result: ValidationResult
    checkpoints_created: int
    rollbacks_performed: int
    build_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class IncrementalValidator:
    """
    Incremental Validator - V6.1.2

    Performs validation after each batch with automatic rollback on failure.

    Key features:
    1. Validate after each batch
    2. Automatic rollback on validation failure
    3. Checkpoint system for recovery
    4. Progress tracking
    5. State persistence
    """

    def __init__(
        self,
        validator: CodeValidator,
        enable_checkpoints: bool = True,
        max_retries_per_batch: int = 2,
        checkpoint_dir: Optional[str] = None,
        auto_rollback: bool = True
    ):
        """
        Initialize Incremental Validator

        Args:
            validator: CodeValidator instance
            enable_checkpoints: Enable checkpoint system
            max_retries_per_batch: Max retries for failed batches
            checkpoint_dir: Directory for checkpoints
            auto_rollback: Auto rollback on validation failure
        """
        self.validator = validator
        self.enable_checkpoints = enable_checkpoints
        self.max_retries = max_retries_per_batch
        self.auto_rollback = auto_rollback

        # Setup checkpoint directory
        if checkpoint_dir:
            self.checkpoint_dir = Path(checkpoint_dir)
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.checkpoint_dir = None

        logger.info(
            f"[IncrementalValidator] Initialized: "
            f"checkpoints={enable_checkpoints}, "
            f"max_retries={max_retries_per_batch}, "
            f"auto_rollback={auto_rollback}"
        )

    def validate_incremental_build(
        self,
        initial_code: str,
        batch_generator,
        methods: List[str]
    ) -> IncrementalBuildResult:
        """
        Perform incremental build with validation after each batch

        Args:
            initial_code: Initial skeleton code
            batch_generator: Generator that yields batches of methods
            methods: All methods to implement

        Returns:
            IncrementalBuildResult
        """
        start_time = datetime.now()

        # Initialize state
        state = IncrementalState(
            current_code=initial_code,
            completed_batches=0,
            total_batches=0,
            methods_completed=[],
            methods_pending=methods.copy(),
            last_checkpoint=None
        )

        logger.info(f"[IncrementalBuild] Starting with {len(methods)} methods")

        try:
            # Create initial checkpoint
            if self.enable_checkpoints:
                self._create_checkpoint(state, ValidationCheckpoint.BEFORE_BATCH, 0)

            # Process batches
            for batch_num, batch_methods in enumerate(batch_generator, 1):
                logger.info(f"\n[Batch {batch_num}] Processing {len(batch_methods)} methods: {batch_methods}")

                # Validate before batch
                pre_validation = self.validator.validate_code(state.current_code)
                if not pre_validation.is_valid:
                    logger.error(f"[Batch {batch_num}] Pre-batch validation failed: {pre_validation.error_type}")
                    return self._build_error_result(state, pre_validation, start_time)

                # Record state before batch
                code_before = state.current_code

                # Generate batch (this would call LLM in real system)
                try:
                    batch_result = self._generate_batch(
                        state.current_code,
                        batch_methods,
                        batch_num
                    )

                    if not batch_result.success:
                        logger.warning(f"[Batch {batch_num}] Generation failed")
                        # Handle batch generation failure
                        batch_result = self._handle_generation_failure(
                            state,
                            batch_methods,
                            batch_num,
                            batch_result
                        )
                        if not batch_result.success:
                            # Critical failure
                            return self._build_error_result(state, None, start_time)

                    state.current_code = batch_result.code_after
                    state.methods_completed.extend(batch_result.methods_added)

                except Exception as e:
                    logger.error(f"[Batch {batch_num}] Exception during generation: {e}")
                    # Attempt recovery
                    recovery_result = self._attempt_recovery(
                        state,
                        batch_methods,
                        batch_num,
                        code_before,
                        e
                    )
                    if not recovery_result.success:
                        return self._build_error_result(state, None, start_time)

                    state.current_code = recovery_result.code_after
                    state.methods_completed.extend(recovery_result.methods_added)

                # Validate after batch
                logger.info(f"[Batch {batch_num}] Validating after batch...")
                post_validation = self.validator.validate_code(state.current_code)

                if not post_validation.is_valid:
                    logger.warning(f"[Batch {batch_num}] Post-batch validation failed: {post_validation.error_type}")

                    if self.auto_rollback:
                        # Rollback to before this batch
                        logger.info(f"[Batch {batch_num}] Rolling back...")
                        state.current_code = code_before
                        state.methods_completed = [
                            m for m in state.methods_completed
                            if m not in batch_methods
                        ]
                        state.rollback_count += 1

                        # Create error checkpoint
                        if self.enable_checkpoints:
                            self._create_checkpoint(
                                state,
                                ValidationCheckpoint.ON_ERROR,
                                batch_num,
                                post_validation
                            )

                        # Try to recover with smaller batch
                        recovery_result = self._recover_from_validation_error(
                            state,
                            batch_methods,
                            batch_num
                        )

                        if recovery_result.success:
                            # Recovery successful
                            logger.info(f"[Batch {batch_num}] Recovery successful")
                            state.current_code = recovery_result.code_after
                            state.methods_completed.extend(recovery_result.methods_added)
                        else:
                            # Recovery failed, try to skip this batch
                            logger.warning(f"[Batch {batch_num}] Recovery failed, skipping batch")
                            if self.enable_checkpoints:
                                self._create_checkpoint(
                                    state,
                                    ValidationCheckpoint.ON_RECOVERY,
                                    batch_num
                                )
                            # Continue with next batch

                else:
                    # Validation successful
                    logger.info(f"[Batch {batch_num}] Validation successful ✓")
                    state.completed_batches += 1

                    # Create checkpoint
                    if self.enable_checkpoints:
                        self._create_checkpoint(
                            state,
                            ValidationCheckpoint.AFTER_BATCH,
                            batch_num,
                            post_validation
                        )

                    # Update pending methods
                    state.methods_pending = [
                        m for m in methods
                        if m not in state.methods_completed
                    ]

                state.total_batches += 1

            # All batches processed successfully
            logger.info(f"\n[IncrementalBuild] All {state.completed_batches} batches completed successfully")

            # Final validation
            final_validation = self.validator.validate_code(state.current_code)

            build_time = (datetime.now() - start_time).total_seconds()

            return IncrementalBuildResult(
                success=True,
                final_code=state.current_code,
                total_batches=state.total_batches,
                completed_batches=state.completed_batches,
                methods_completed=state.methods_completed,
                validation_result=final_validation,
                checkpoints_created=len(state.checkpoint_history),
                rollbacks_performed=state.rollback_count,
                build_time=build_time,
                metadata={
                    'total_methods': len(methods),
                    'methods_pending': state.methods_pending,
                    'error_count': len(state.errors_encountered)
                }
            )

        except Exception as e:
            logger.error(f"[IncrementalBuild] Critical error: {e}")
            return self._build_error_result(state, None, start_time)

    def _generate_batch(
        self,
        current_code: str,
        methods: List[str],
        batch_num: int
    ) -> BatchGenerationResult:
        """
        Generate a batch of methods (would use LLM in production)

        This is a placeholder for the actual generation logic.
        In production, this would call the LLM to implement methods.

        Args:
            current_code: Code so far
            methods: Methods to implement in this batch
            batch_num: Batch number

        Returns:
            BatchGenerationResult
        """
        # Placeholder: In production, this would call LLM
        # For now, simulate by adding placeholder implementations

        logger.info(f"[Generate] Implementing batch {batch_num}: {methods}")

        # Simulate code generation
        new_code = current_code
        methods_added = []

        for method_name in methods:
            # Add placeholder implementation
            placeholder = f"\n    def {method_name}(self):\n        pass  # TODO: implement\n"

            # Find the class or appropriate location
            if "class " in new_code:
                # Add to class
                new_code = new_code.rstrip() + placeholder + "\n"
            else:
                # Add at module level
                new_code += f"\ndef {method_name}(self):\n    pass  # TODO: implement\n"

            methods_added.append(method_name)

        return BatchGenerationResult(
            batch_number=batch_num,
            success=True,
            code_before=current_code,
            code_after=new_code,
            methods_added=methods_added,
            validation_result=None,
            was_rolled_back=False,
            time_taken=1.0
        )

    def _handle_generation_failure(
        self,
        state: IncrementalState,
        methods: List[str],
        batch_num: int,
        error_result: BatchGenerationResult
    ) -> BatchGenerationResult:
        """
        Handle batch generation failure

        Args:
            state: Current state
            methods: Methods in failed batch
            batch_num: Batch number
            error_result: Error result

        Returns:
            Recovery attempt result
        """
        logger.warning(f"[Handle Failure] Batch {batch_num} generation failed")

        # Record error
        state.errors_encountered.append({
            'batch': batch_num,
            'methods': methods,
            'error': error_result.error_message,
            'timestamp': datetime.now().isoformat()
        })

        # Try to recover by implementing methods individually
        logger.info(f"[Handle Failure] Attempting individual method implementation")

        recovered_code = state.current_code
        recovered_methods = []

        for method in methods:
            try:
                # Try to implement just this method
                single_method_code = self._generate_single_method(
                    state.current_code,
                    method,
                    batch_num
                )

                if single_method_code:
                    recovered_code = single_method_code
                    recovered_methods.append(method)
                    logger.info(f"[Handle Failure] Recovered method: {method}")
                else:
                    logger.warning(f"[Handle Failure] Could not recover: {method}")

            except Exception as e:
                logger.error(f"[Handle Failure] Error recovering {method}: {e}")
                continue

        if recovered_methods:
            return BatchGenerationResult(
                batch_number=batch_num,
                success=True,
                code_before=state.current_code,
                code_after=recovered_code,
                methods_added=recovered_methods,
                validation_result=None,
                was_rolled_back=False,
                error_message=None,
                time_taken=0.0
            )
        else:
            return error_result

    def _generate_single_method(
        self,
        current_code: str,
        method_name: str,
        batch_num: int
    ) -> Optional[str]:
        """
        Generate a single method (simplified recovery)

        Args:
            current_code: Current code state
            method_name: Method to generate
            batch_num: Batch number

        Returns:
            New code with method added, or None
        """
        # Simplified: add basic implementation
        new_method = f"\n    def {method_name}(self):\n        pass  # Recovery implementation\n"

        # Find insertion point
        if "class " in current_code:
            # Add to last class
            lines = current_code.split('\n')
            # Find last class definition
            for i in range(len(lines) - 1, -1, -1):
                if lines[i].strip().startswith('class '):
                    # Insert after this class
                    lines.insert(i + 1, new_method)
                    break
            return '\n'.join(lines)
        else:
            # Add at end
            return current_code + new_method

    def _attempt_recovery(
        self,
        state: IncrementalState,
        methods: List[str],
        batch_num: int,
        code_before: str,
        exception: Exception
    ) -> BatchGenerationResult:
        """
        Attempt recovery from exception

        Args:
            state: Current state
            methods: Methods being processed
            batch_num: Batch number
            code_before: Code before error
            exception: Exception that occurred

        Returns:
            Recovery result
        """
        logger.info(f"[Recovery] Attempting recovery from exception")

        try:
            # Try to implement methods one by one
            recovered_code = code_before
            recovered_methods = []

            for method in methods:
                single_code = self._generate_single_method(
                    recovered_code,
                    method,
                    batch_num
                )

                if single_code:
                    recovered_code = single_code
                    recovered_methods.append(method)

            if recovered_methods:
                logger.info(f"[Recovery] Recovered {len(recovered_methods)} methods")
                return BatchGenerationResult(
                    batch_number=batch_num,
                    success=True,
                    code_before=code_before,
                    code_after=recovered_code,
                    methods_added=recovered_methods,
                    validation_result=None,
                    was_rolled_back=False,
                    error_message=None,
                    time_taken=0.0
                )

        except Exception as e:
            logger.error(f"[Recovery] Recovery failed: {e}")

        return BatchGenerationResult(
            batch_number=batch_num,
            success=False,
            code_before=code_before,
            code_after=code_before,
            methods_added=[],
            validation_result=None,
            was_rolled_back=False,
            error_message=str(exception),
            time_taken=0.0
        )

    def _recover_from_validation_error(
        self,
        state: IncrementalState,
        methods: List[str],
        batch_num: int
    ) -> BatchGenerationResult:
        """
        Recover from validation error

        Try to implement the batch with different approach:
        1. Reduce batch size (implement methods one by one)
        2. Use simpler implementations
        3. Skip problematic methods

        Args:
            state: Current state
            methods: Methods in failed batch
            batch_num: Batch number

        Returns:
            Recovery result
        """
        logger.info(f"[Recovery] Recovering from validation error, batch {batch_num}")

        # Strategy 1: Implement methods one by one
        recovered_code = state.current_code
        recovered_methods = []

        for method in methods:
            try:
                single_code = self._generate_single_method(
                    recovered_code,
                    method,
                    batch_num
                )

                # Validate after each method
                if single_code:
                    validation = self.validator.validate_code(single_code)

                    if validation.is_valid:
                        recovered_code = single_code
                        recovered_methods.append(method)
                        logger.info(f"[Recovery] Method {method} recovered individually")
                    else:
                        logger.warning(f"[Recovery] Method {method} validation failed, skipping")
                        continue

            except Exception as e:
                logger.error(f"[Recovery] Error with {method}: {e}")
                continue

        if recovered_methods:
            logger.info(f"[Recovery] Recovered {len(recovered_methods)}/{len(methods)} methods")
            return BatchGenerationResult(
                batch_number=batch_num,
                success=True,
                code_before=state.current_code,
                code_after=recovered_code,
                methods_added=recovered_methods,
                validation_result=None,
                was_rolled_back=False,
                error_message=None,
                time_taken=0.0
            )

        # Recovery failed
        logger.error(f"[Recovery] Could not recover batch {batch_num}")
        return BatchGenerationResult(
            batch_number=batch_num,
            success=False,
            code_before=state.current_code,
            code_after=state.current_code,
            methods_added=[],
            validation_result=None,
            was_rolled_back=False,
            error_message="Recovery failed",
            time_taken=0.0
        )

    def _create_checkpoint(
        self,
        state: IncrementalState,
        checkpoint_type: ValidationCheckpoint,
        batch_num: int,
        validation_result: Optional[ValidationResult] = None
    ):
        """
        Create a checkpoint for recovery

        Args:
            state: Current state to save
            checkpoint_type: Type of checkpoint
            batch_num: Batch number
            validation_result: Optional validation result
        """
        if not self.checkpoint_dir:
            return

        checkpoint_id = f"batch_{batch_num}_{checkpoint_type.value}"

        checkpoint_data = {
            'checkpoint_id': checkpoint_id,
            'checkpoint_type': checkpoint_type.value,
            'batch_number': batch_num,
            'timestamp': datetime.now().isoformat(),
            'state': {
                'current_code_hash': self._hash_code(state.current_code),
                'completed_batches': state.completed_batches,
                'methods_completed': state.methods_completed,
                'methods_pending': state.methods_pending
            }
        }

        # Save validation result if provided
        if validation_result:
            checkpoint_data['validation'] = {
                'is_valid': validation_result.is_valid,
                'error_type': validation_result.error_type,
                'error_message': validation_result.error_message
            }

        # Save checkpoint to file
        checkpoint_file = self.checkpoint_dir / f"{checkpoint_id}.json"
        with open(checkpoint_file, 'w') as f:
            json.dump(checkpoint_data, f, indent=2)

        state.last_checkpoint = checkpoint_id
        state.checkpoint_history.append(checkpoint_data)

        logger.debug(f"[Checkpoint] Created: {checkpoint_id}")

    def _hash_code(self, code: str) -> str:
        """Create hash of code for change detection"""
        return hashlib.md5(code.encode()).hexdigest()

    def _build_error_result(
        self,
        state: IncrementalState,
        validation_result: Optional[ValidationResult],
        start_time: datetime
    ) -> IncrementalBuildResult:
        """Build error result"""
        build_time = (datetime.now() - start_time).total_seconds()

        return IncrementalBuildResult(
            success=False,
            final_code=state.current_code,
            total_batches=state.total_batches,
            completed_batches=state.completed_batches,
            methods_completed=state.methods_completed,
            validation_result=validation_result or ValidationResult(
                is_valid=False,
                error_type="build_failed",
                error_message="Incremental build failed",
                error_line=None,
                suggestions=["Check logs for details"],
                metadata={}
            ),
            checkpoints_created=len(state.checkpoint_history),
            rollbacks_performed=state.rollback_count,
            build_time=build_time,
            metadata={
                'total_methods': len(state.methods_completed) + len(state.methods_pending),
                'methods_completed': state.methods_completed,
                'methods_pending': state.methods_pending,
                'error_count': len(state.errors_encountered)
            }
        )

    def restore_from_checkpoint(self, checkpoint_id: str) -> Optional[IncrementalState]:
        """
        Restore state from checkpoint

        Args:
            checkpoint_id: Checkpoint ID to restore

        Returns:
            Restored IncrementalState or None
        """
        if not self.checkpoint_dir:
            logger.error("[Restore] Checkpoint directory not configured")
            return None

        checkpoint_file = self.checkpoint_dir / f"{checkpoint_id}.json"

        if not checkpoint_file.exists():
            logger.error(f"[Restore] Checkpoint not found: {checkpoint_id}")
            return None

        try:
            with open(checkpoint_file, 'r') as f:
                checkpoint_data = json.load(f)

            logger.info(f"[Restore] Restored from checkpoint: {checkpoint_id}")
            # Note: Full state restoration would require saving/loading full code
            # This is a simplified version

            return checkpoint_data

        except Exception as e:
            logger.error(f"[Restore] Failed to restore checkpoint: {e}")
            return None

    def get_progress_summary(self, state: IncrementalState) -> Dict[str, Any]:
        """
        Get progress summary

        Args:
            state: Current state

        Returns:
            Progress summary dict
        """
        total_methods = len(state.methods_completed) + len(state.methods_pending)
        completed = len(state.methods_completed)
        progress_percent = (completed / total_methods * 100) if total_methods > 0 else 0

        return {
            'total_methods': total_methods,
            'completed_methods': completed,
            'pending_methods': len(state.methods_pending),
            'progress_percent': progress_percent,
            'completed_batches': state.completed_batches,
            'rollbacks': state.rollback_count,
            'errors': len(state.errors_encountered),
            'checkpoints': len(state.checkpoint_history)
        }


# Convenience functions
def validate_incrementally(
    initial_code: str,
    batch_generator,
    methods: List[str],
    validator: CodeValidator,
    **kwargs
) -> IncrementalBuildResult:
    """
    Convenience function for incremental validation

    Args:
        initial_code: Initial code
        batch_generator: Batch generator
        methods: Methods to implement
        validator: CodeValidator instance
        **kwargs: Args for IncrementalValidator

    Returns:
        IncrementalBuildResult
    """
    incremental_validator = IncrementalValidator(validator, **kwargs)
    return incremental_validator.validate_incremental_build(
        initial_code,
        batch_generator,
        methods
    )


if __name__ == "__main__":
    import asyncio

    print("=" * 80)
    print("Incremental Validation System Test")
    print("=" * 80)

    # Create validator
    from validators import CodeValidator
    validator = CodeValidator()

    # Create incremental validator
    inc_validator = IncrementalValidator(
        validator,
        enable_checkpoints=True,
        checkpoint_dir=".checkpoints_temp"
    )

    # Test 1: Simple incremental build
    print("\n[Test 1] Simple incremental build")

    def batch_gen_1():
        """Yield batches of methods"""
        yield ['get_name', 'set_name']
        yield ['get_age', 'set_age']
        yield ['is_valid']

    code_1 = "class Person:\n    pass\n"
    methods_1 = ['get_name', 'set_name', 'get_age', 'set_age', 'is_valid']

    result_1 = inc_validator.validate_incremental_build(
        code_1,
        batch_gen_1(),
        methods_1
    )

    print(f"Success: {result_1.success}")
    print(f"Batches completed: {result_1.completed_batches}/{result_1.total_batches}")
    print(f"Methods completed: {result_1.methods_completed}")
    print(f"Progress: {len(result_1.methods_completed)}/{result_1.metadata['total_methods']}")
    print(f"Rollbacks: {result_1.rollbacks_performed}")
    print(f"Checkpoints: {result_1.checkpoints_created}")

    # Test 2: Build with validation error
    print("\n[Test 2] Build with validation error recovery")

    code_2 = "class Calculator:\n"
    methods_2 = ['add', 'subtract', 'multiply', 'divide']

    def batch_gen_2():
        yield ['add', 'subtract']
        yield ['multiply']  # This might cause issues
        yield ['divide']

    result_2 = inc_validator.validate_incremental_build(
        code_2,
        batch_gen_2(),
        methods_2
    )

    print(f"Success: {result_2.success}")
    if result_2.success:
        print(f"Final code length: {len(result_2.final_code)}")

    # Cleanup
    import shutil
    if os.path.exists(".checkpoints_temp"):
        shutil.rmtree(".checkpoints_temp")

    print("\n" + "=" * 80)
    print("All tests completed")
    print("=" * 80)
