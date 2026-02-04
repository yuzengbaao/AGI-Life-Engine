import sys
import os
import asyncio

# 添加项目根目录到路径，以便导入 AGI_Life_Engine
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from AGI_Life_Engine import LifeEngineEventBus
    print("✅ Successfully imported LifeEngineEventBus from AGI_Life_Engine.")
except ImportError as e:
    print(f"❌ Failed to import: {e}")
    sys.exit(1)

async def main():
    print("🔬 Testing LifeEngineEventBus...")
    
    # 1. Initialize Bus
    try:
        bus = LifeEngineEventBus(source="VerifyScript")
        print("   ✅ Bus initialized.")
    except Exception as e:
        print(f"   ❌ Bus initialization failed: {e}")
        return

    # 2. Define a subscriber to verify receipt
    received_events = []
    async def test_handler(event):
        print(f"   📨 Handler received event: {event.type} from {event.source}")
        received_events.append(event)

    # 3. Subscribe
    bus.subscribe("test_event", test_handler)
    print("   ✅ Subscribed to 'test_event'.")

    # 4. Publish
    print("   📢 Publishing 'test_event'...")
    await bus.publish("test_event", {"message": "Hello World", "status": "ok"})
    
    # Allow some time for async processing
    await asyncio.sleep(0.1)

    # 5. Verify
    if len(received_events) > 0:
        evt = received_events[0]
        if evt.data.get("message") == "Hello World":
            print("   ✅ SUCCESS: Event received correctly.")
        else:
            print("   ⚠️ Event received but data mismatch.")
    else:
        print("   ❌ FAILURE: No event received.")

if __name__ == "__main__":
    asyncio.run(main())
