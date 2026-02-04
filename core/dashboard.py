import os
import time
from typing import Dict, Any

class Dashboard:
    def __init__(self):
        self.start_time = time.time()
        
    def render(self, context: Dict[str, Any], stats: Dict[str, Any]):
        """
        Render a text-based dashboard.
        """
        # Clear screen (simulated with newlines for logging safety)
        output = []
        output.append("=" * 60)
        output.append(f"   AGI DIGITAL LIFEFORM - STATUS DASHBOARD   ")
        output.append("=" * 60)
        
        # 1. Vitals
        uptime = int(time.time() - self.start_time)
        metrics = context.get('latest_metrics', {})
        output.append(f"⏱️  Uptime: {uptime}s | 🔄 Iteration: {context.get('iteration', 0)}")
        output.append(f"❤️  Heartbeat: ACTIVE | 💾 Memory Usage: {metrics.get('process_memory_mb', 0):.1f} MB")
        output.append(f"🧠  CPU Usage: {metrics.get('cpu_percent', 0)}% | 🛡️  Health: {metrics.get('status', 'OK')}")
        output.append("-" * 60)
        
        # 2. Cognition
        state = context.get('current_state', 'UNKNOWN')
        strategy = context.get('current_strategy', 'Standard')
        output.append(f"🤔 Current State: {state}")
        output.append(f"🎯 Strategy: {strategy}")
        
        last_thought = context.get('last_thought', {})
        if last_thought:
            goal = last_thought.get('goal', 'None')
            output.append(f"💭 Current Goal: {goal}")
            
        output.append("-" * 60)
        
        # 3. Evolution & Knowledge
        graph_stats = stats.get('graph_stats', {'node_count': 0})
        output.append(f"📚 Knowledge Nodes: {graph_stats.get('node_count', 0)}")
        
        last_result = context.get('last_action_result', {})
        if last_result:
            score = last_result.get('success_score', 0.0)
            bar_len = int(score * 20)
            bar = "█" * bar_len + "░" * (20 - bar_len)
            output.append(f"📉 Last Performance: [{bar}] {score:.2f}")
            
        output.append("=" * 60)
        
        return "\n".join(output)

    def display(self, context: Dict[str, Any], stats: Dict[str, Any]):
        print(self.render(context, stats))

class VisualDashboardAdapter:
    """Adapter to provide visual dashboard functionality."""
    def __init__(self, system_instance):
        self.system = system_instance
        self.text_dashboard = Dashboard()
        self.running = False
        
    def start(self):
        """Start the visual dashboard (e.g. Flask server or just status logging)."""
        if self.running:
            return
        self.running = True
        print("📊 Visual Dashboard Adapter started.")
        # In a real implementation, this would start the web server thread
        # e.g., threading.Thread(target=run_flask_app).start()

def get_visual_dashboard(system_instance) -> VisualDashboardAdapter:
    """Factory function to get a visual dashboard instance."""
    return VisualDashboardAdapter(system_instance)
