import asyncio
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agi_component_coordinator import ComponentCoordinator, Event

def test_bus():
    print("🧪 Testing ComponentCoordinator EventBus...")
    
    # Mock AGI system
    class MockAGI:
        pass
    
    coord = ComponentCoordinator(MockAGI())
    
    # Define handler
    received_events = []
    async def handler(event: Event):
        print(f"📥 Received event: {event.type} from {event.source}")
        received_events.append(event)
        
    # Subscribe
    coord.subscribe("test.event", handler)
    
    async def _run():
        await coord.publish("test.event", "test_script", {"msg": "hello"})
        await asyncio.sleep(0.1)

    asyncio.run(_run())
    
    if len(received_events) == 1:
        print("✅ Event received successfully!")
        print(f"   Data: {received_events[0].data}")
    else:
        print(f"❌ Event not received. Count: {len(received_events)}")

if __name__ == "__main__":
    test_bus()
