"""
瓶颈系统启动钩子 (Bottleneck System Startup Hook)

这个文件在AGI_Life_Engine启动时自动加载，初始化瓶颈系统。
"""

import sys
from pathlib import Path

# 确保在项目根目录
BASE_DIR = Path(__file__).parent
sys.path.insert(0, str(BASE_DIR))

# 初始化瓶颈系统
try:
    from bottleneck_integration_patch import apply_bottleneck_integration
    success = apply_bottleneck_integration()
    if success:
        print("✅ 瓶颈系统启动钩子执行成功")
    else:
        print("❌ 瓶颈系统启动钩子执行失败")
except Exception as e:
    print(f"❌ 瓶颈系统启动钩子异常: {e}")
    import traceback
    traceback.print_exc()
