#\!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""AGI AUTONOMOUS CORE V6.2 - Full Integration"""

import asyncio
import os
import sys
import logging
from typing import Optional, List
from pathlib import Path

# Import Phase 1 & 2
try:
    from token_budget import TokenBudget
    from validators import CodeValidator
    from fixers import LLMSemanticFixer
    PHASE1 = True
except:
    PHASE1 = False

try:
    from adaptive_batch_processor import AdaptiveBatchProcessor
    from incremental_validator import IncrementalValidator
    from error_classifier import ErrorClassifier
    from fix_optimizer import FixOptimizer
    PHASE2 = True
except:
    PHASE2 = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DeepSeekLLM:
    """DeepSeek LLM client"""
    def __init__(self):
        self.client = None
        self.model = None
        try:
            import openai
            api_key = os.getenv('DEEPSEEK_API_KEY')
            if api_key:
                self.client = openai.AsyncOpenAI(
                    api_key=api_key,
                    base_url='https://api.deepseek.com/v1'
                )
                self.model = os.getenv('DEEPSEEK_MODEL', 'deepseek-chat')
                logger.info(f'[LLM] Initialized: {self.model}')
        except Exception as e:
            logger.error(f'[LLM] Init failed: {e}')
    
    async def generate(self, prompt: str, max_tokens: int = 4000) -> str:
        if not self.client:
            raise ValueError('LLM not initialized')
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[{'role': 'user', 'content': prompt}],
            max_tokens=max_tokens
        )
        return response.choices[0].message.content


class V62Generator:
    """V6.2 Code Generator - Phase 1 + Phase 2 Integration"""
    
    def __init__(self, llm: DeepSeekLLM):
        self.llm = llm

        # Phase 1 components (initialize first, needed by Phase 2)
        self.validator = CodeValidator() if PHASE1 else None
        self.fixer = LLMSemanticFixer(llm) if PHASE1 else None

        # Phase 2 components (with proper dependencies)
        if PHASE2:
            # Create token budget for batch processor
            from token_budget import TokenBudget
            self.token_budget = TokenBudget() if PHASE1 else None

            self.batch_processor = AdaptiveBatchProcessor(
                token_budget=self.token_budget
            ) if self.token_budget else None

            self.incremental_validator = IncrementalValidator(
                validator=self.validator
            ) if self.validator else None

            self.error_classifier = ErrorClassifier()
            self.fix_optimizer = FixOptimizer()
        else:
            self.batch_processor = None
            self.incremental_validator = None
            self.error_classifier = None
            self.fix_optimizer = None

        # Configure fix optimizer with semantic fixer
        if self.fix_optimizer and self.fixer:
            self.fix_optimizer.parallel_executor.set_fixer(self.fixer)
    
    async def generate(self, project_desc: str, methods: List[str], filename: str):
        """Generate code with V6.2 optimization"""
        import time
        start = time.time()
        
        logger.info(f'[V6.2] Starting: {filename}')
        logger.info(f'[V6.2] Phase 1: {PHASE1}, Phase 2: {PHASE2}')
        
        # Calculate batch size
        if self.batch_processor:
            batch_size = self.batch_processor.calculate_optimal_batch_size(methods)
            logger.info(f'[V6.2] Adaptive batch size: {batch_size}')
        else:
            batch_size = 3
        
        # Create batches
        batches = [methods[i:i+batch_size] for i in range(0, len(methods), batch_size)]
        logger.info(f'[V6.2] Created {len(batches)} batches')
        
        # Process batches
        code = ''
        for i, batch in enumerate(batches, 1):
            logger.info(f'[V6.2] Batch {i}/{len(batches)}')
            
            # Generate
            batch_code = await self._generate_batch(project_desc, batch, code)
            if not batch_code:
                continue
            
            # Validate & Fix
            batch_code = await self._validate_and_fix(batch_code, i)
            if batch_code:
                code = batch_code
        
        # Save
        if code:
            Path(filename).parent.mkdir(parents=True, exist_ok=True)
            Path(filename).write_text(code, encoding='utf-8')
            
            duration = (time.time() - start) * 1000
            logger.info(f'[V6.2] Saved to {filename} ({duration:.2f}ms)')
            
            return {
                'success': True,
                'filename': filename,
                'batches': len(batches),
                'duration_ms': duration
            }
        
        return {'success': False}
    
    async def _generate_batch(self, project_desc: str, batch: List[str], existing: str) -> str:
        """Generate batch code"""
        prompt = f'Project: {project_desc}\n\nMethods:\n'
        prompt += '\n'.join(f'- {m}' for m in batch)
        if existing:
            prompt += f'\n\nExisting:\n```python\n{existing}\n```'
        prompt += '\n\nGenerate Python code with type hints.'
        
        try:
            response = await self.llm.generate(prompt)
            return self._extract_code(response)
        except:
            return ''
    
    async def _validate_and_fix(self, code: str, batch_idx: int) -> str:
        """Validate and fix using Phase 1 & 2"""
        # Phase 1 validation
        if self.validator:
            result = self.validator.validate_code(code, f'batch_{batch_idx}.py')
            
            if result.is_valid:
                return code
            
            logger.warning(f'[V6.2] Validation failed: {result.error_type}')
            
            # Phase 2 classification
            if self.error_classifier:
                classified = self.error_classifier.classify_error(result, code)
                logger.info(f'[V6.2] Error: {classified.category.value}')
                
                # Phase 2 optimization
                if self.fix_optimizer:
                    fix_result = await self.fix_optimizer.optimize_fix(
                        code=code,
                        validation_result=result,
                        classified_error=classified,
                        time_critical=False
                    )
                    
                    if fix_result.success:
                        logger.info(f'[V6.2] Fixed with {fix_result.best_strategy.value}')
                        return fix_result.final_code
            
            # Phase 1 fix
            if self.fixer:
                fix_result = await self.fixer.fix_code(code, result)
                if fix_result.success:
                    return fix_result.fixed_code
        
        return code
    
    def _extract_code(self, response: str) -> str:
        """Extract code from response"""
        if '```python' in response:
            start = response.find('```python') + 9
            end = response.find('```', start)
            if end > start:
                return response[start:end].strip()
        
        if '```' in response:
            start = response.find('```') + 3
            end = response.find('```', start)
            if end > start:
                code = response[start:end].strip()
                if code.startswith('python'):
                    code = code[6:].strip()
                return code
        
        return response.strip()


async def main():
    """Main entry point"""
    print('=' * 80)
    print('AGI AUTONOMOUS CORE V6.2 - Intelligent Optimization')
    print('=' * 80)
    print()
    print(f'[V6.2] Phase 1: {"OK" if PHASE1 else "SKIP"}')
    print(f'[V6.2] Phase 2: {"OK" if PHASE2 else "SKIP"}')
    print()
    
    if not PHASE1 and not PHASE2:
        print('[V6.2] ERROR: No components available!')
        return
    
    llm = DeepSeekLLM()
    generator = V62Generator(llm)
    
    # Example
    methods = [
        'def add(self, a: float, b: float) -> float:',
        'def subtract(self, a: float, b: float) -> float:',
        'def multiply(self, a: float, b: float) -> float:',
        'def divide(self, a: float, b: float) -> float:',
    ]
    
    result = await generator.generate(
        project_desc='Calculator class',
        methods=methods,
        filename='output/test_v62.py'
    )
    
    print()
    print('[V6.2] Result:', result)
    print()
    print('=' * 80)


if __name__ == '__main__':
    asyncio.run(main())
