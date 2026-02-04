#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Execute system switch without interactive prompt"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Add scripts directory to path
scripts_dir = Path(__file__).parent
sys.path.insert(0, str(scripts_dir))

# Import the module directly
import switch_to_production

if __name__ == "__main__":
    print("[自动] 开始执行系统切换...")
    switcher = switch_to_production.SystemSwitcher()
    switcher.run_full_switch()
