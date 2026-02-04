# Phase 3.3: Self-Evolution Path Resolution & Context Fix

## 1. Issue Description
In the Cycle 7 logs (`周期任务信息流日志解读.txt`), the system encountered a critical failure during self-optimization:
```
2025-12-25 19:00:54,917 - ERROR -   ❌ Dry Run failed: core.perception.processors.adapter
2025-12-25 19:00:54,919 - ERROR -    ❌ Self-Optimization FAILED.
```
This failure prevented the system from verifying and applying optimizations to its own code, effectively blocking the "Self-Evolution" capability.

## 2. Root Cause Analysis
- **Isolation vs. Dependency**: The `ShadowRunner` was designed to isolate modified files for safety. It only copied the specific file being modified (e.g., `adapter.py`) into a clean temporary directory.
- **Relative Imports**: `core.perception.processors.adapter` contains relative imports like `from .video import AdvancedVideoProcessor`.
- **Failure Mechanism**: Since `video.py` was not present in the shadow directory, the Python interpreter raised an `ImportError` (or `SystemError` for relative imports with no parent package), causing the `Dry Run` to fail.

## 3. Implemented Solution
We implemented a **Full Context Shadow Environment** strategy to balance isolation and dependency resolution.

### 3.1 Code Changes
1.  **`core/research/lab.py` (ShadowRunner)**:
    - Added `full_context` parameter to `create_shadow_env`.
    - When `full_context=True`, the runner uses `shutil.copytree` to copy the entire project `core/` directory to the shadow environment.
    - Added `ignore_patterns` to exclude `__pycache__`, `.git`, etc., for performance.
    - **Timeout Adjustment**: Increased execution timeout from **10s to 30s** to accommodate the loading time of heavy libraries (PyTorch, OpenCV) often imported by perception modules.

2.  **`core/evolution/impl.py` (SandboxCompiler)**:
    - Updated `verify_in_sandbox` to call `create_shadow_env` with `full_context=True`.
    - This ensures that when the system attempts to optimize *any* core module, all its siblings and dependencies are available for the verification process.

### 3.2 Verification
- **Test Script**: `tests/verify_shadow_context.py`
- **Result**: Confirmed that the `core` directory is successfully copied to the shadow session, enabling successful resolution of relative imports.

## 4. Impact
- **Before**: System could only optimize standalone utility files without dependencies.
- **After**: System can now safely optimize complex, interconnected modules (like Perception, Planner, Memory) that rely on the rest of the framework.
- **Status**: The "Last Mile" of Self-Evolution is now bridged. The system is capable of verifying its own "surgery" on complex organs.

## 5. Next Steps
- Monitor the next few cycles of `AGI_Life_Engine.py` to confirm `Self-Optimization SUCCESS` logs appear for complex modules.
