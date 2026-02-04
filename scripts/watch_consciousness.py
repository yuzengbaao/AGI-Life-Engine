import time
import json
import os
import sys

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def watch():
    file_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "consciousness.json")
    
    print("🧠 Connecting to AGI Global Workspace...")
    last_ts = 0
    
    while True:
        try:
            if os.path.exists(file_path):
                with open(file_path, 'r', encoding='utf-8') as f:
                    try:
                        data = json.load(f)
                    except json.JSONDecodeError:
                        time.sleep(0.5)
                        continue
                        
                ts = data.get("timestamp", 0)
                if ts != last_ts:
                    clear_screen()
                    print(f"🧠 AGI CONSCIOUSNESS STREAM [T={time.strftime('%H:%M:%S', time.localtime(ts))}]")
                    print("="*60)
                    
                    print(f"🔥 DRIVE: {data.get('sensory_summary', {}).get('motivation', 'Unknown')}")
                    print(f"👀 ATTENTION: {data.get('attention')}")
                    print(f"🧘 STATE: {data.get('cognitive_state')}")
                    
                    print("\n🎯 GOAL STACK:")
                    if data.get('goals'):
                        for g in reversed(data['goals']):
                            print(f"  - [{g['priority'].upper()}] {g['goal']}")
                    else:
                        print("  (Empty)")
                        
                    print("\n💭 INNER MONOLOGUE (Last 5 thoughts):")
                    thoughts = data.get('thoughts', [])
                    for t in thoughts[-5:]:
                        print(f"  {t}")
                        
                    last_ts = ts
            else:
                print("Waiting for consciousness signal...", end='\r')
                
            time.sleep(1)
            
        except KeyboardInterrupt:
            print("\nDisconnected.")
            break
        except Exception as e:
            print(f"Error: {e}")
            time.sleep(1)

if __name__ == "__main__":
    watch()