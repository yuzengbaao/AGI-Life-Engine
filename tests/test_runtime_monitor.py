import sys
import os

# 将项目根目录加入路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.perception.runtime_monitor import RuntimeMonitor

class CriticalComponent:
    def __init__(self, name):
        self.name = name
        # 模拟在初始化时注册
        RuntimeMonitor.register(self, context_info=f"Component: {name}")

def create_component():
    return CriticalComponent("Planner")

def main():
    print("=== Testing RuntimeMonitor (Phase 2.3) ===")
    
    # 1. 创建对象
    print("[1] Creating object...")
    comp = create_component()
    
    # 2. 检查对象
    print("[2] Inspecting object...")
    info = RuntimeMonitor.inspect_object(comp)
    
    if info:
        print(f"✅ Object Found in Registry!")
        print(f"   Type: {info['type']}")
        print(f"   File: {info['file_path']}")
        print(f"   Line: {info['line_number']}")
        print(f"   Context: {info['context']}")
        
        # 验证行号是否正确 (CriticalComponent.__init__ 大约在第 11 行)
        if "test_runtime_monitor.py" in info['file_path']:
            print("✅ File path verification successful.")
        else:
            print("❌ File path mismatch.")
    else:
        print("❌ Object NOT found in registry.")

    # 3. 模拟未注册对象
    print("\n[3] Testing unregistered object...")
    x = [1, 2, 3]
    info_x = RuntimeMonitor.inspect_object(x)
    if info_x is None:
        print("✅ Unregistered object correctly returned None.")
    else:
        print("❌ Unexpected result for unregistered object.")

if __name__ == "__main__":
    main()