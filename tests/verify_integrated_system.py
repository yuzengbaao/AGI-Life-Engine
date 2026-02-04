import asyncio
import sys
import os
import logging

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

from agi_system_evolutionary import FullyIntegratedAGISystem
from agi_component_coordinator import Event

async def verify_system():
    print("\n" + "="*80)
    print("🚀 Verifying Fully Integrated AGI System & 34 Capabilities")
    print("="*80 + "\n")
    
    # 1. Initialize System
    agi = FullyIntegratedAGISystem(config_path="agi_integrated_config.json")
    
    # Subscribe to events to verify data flow
    event_log = []
    async def event_logger(event: Event):
        print(f"📨 [EventBus] {event.type} from {event.source}")
        event_log.append(event)
        
    # We need to initialize first to get the coordinator
    print("⏳ Initializing all modules (this may take a moment)...")
    await agi.initialize_all_modules()
    
    if agi.coordinator:
        agi.coordinator.subscribe("*", event_logger)
        print("✅ Event Bus Subscribed")
    else:
        print("❌ Coordinator not initialized!")
    
    # 2. Verify Capabilities
    print("\n📊 Capability Verification:")
    active_modules = agi.status.active_modules
    print(f"   Active Modules Count: {len(active_modules)}")
    
    expected_capabilities = [
        "LLM推理", "世界模型", "JEPA世界模型", "P1增强世界模型", 
        "哲学思辨", "持续学习", "自我进化", "自我优化", "创新方案", 
        "P1三层记忆", "P1层级规划", "自主学习守护进程", 
        "元认知层", "架构感知层", "RCE重构能力",
        "视觉感知", "听觉感知", 
        "系统监控", "任务队列", "负载均衡", "健康检查", "备份恢复", 
        "文件操作", "OpenHands助手", "自主文档创建", "组件协调器", 
        "安全管理", "系统评估", "权限管理固件"
    ]
    
    missing = []
    for cap in expected_capabilities:
        found = False
        for module in active_modules:
            if cap in module or module in cap:
                found = True
                break
        if found:
            print(f"   ✅ {cap}")
        else:
            # Some might be disabled by config, check if that's expected
            print(f"   ⚠️ {cap} (Not Active/Detected)")
            missing.append(cap)
            
    print(f"\n   Total Active: {len(active_modules)} / Target: ~34")
    
    # 3. Verify Data Flow (LLM Event)
    print("\n🔄 Verifying Data Flow (LLM -> EventBus)...")
    if hasattr(agi, 'local_llm_provider') and agi.local_llm_provider:
        # Trigger a simple generation
        try:
            response = agi.generate_response("Ping", system_msg="Test")
            print(f"   LLM Response: {response}")
            
            # Give a moment for async events to process
            await asyncio.sleep(0.5)
            
            # Check if we received LLM events
            llm_events = [e for e in event_log if e.source == "local_llm"]
            if llm_events:
                print(f"   ✅ Received {len(llm_events)} LLM events via Bus")
                for e in llm_events:
                    print(f"      - {e.type}: {e.data.keys()}")
            else:
                print("   ❌ No LLM events received via Bus")
        except Exception as e:
            print(f"   ❌ LLM Generation Failed: {e}")
    else:
        print("   ❌ Local LLM Provider not available")

    print("\n" + "="*80)
    print("🏁 Verification Complete")
    print("="*80 + "\n")

if __name__ == "__main__":
    asyncio.run(verify_system())
