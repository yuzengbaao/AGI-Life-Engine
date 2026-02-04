from typing import Dict, Any, List
import numpy as np

class MetricDrivenEvolver:
    def __init__(self):
        self.history_metrics: List[Dict[str, float]] = []
        
    def analyze_metrics(self, current_metrics: Dict[str, float]) -> Dict[str, Any]:
        """
        Analyze current metrics to decide if evolution/adaptation is needed.
        
        Metrics expected:
        - cpu_percent
        - memory_percent
        - task_success_rate (0.0 - 1.0)
        """
        self.history_metrics.append(current_metrics)
        # Keep last 50 data points
        if len(self.history_metrics) > 50:
            self.history_metrics.pop(0)
            
        analysis = {
            "status": "STABLE",
            "suggestion": None
        }
        
        # 1. Resource Pressure Check
        if current_metrics.get('memory_percent', 0) > 85:
            analysis['status'] = "STRESSED"
            analysis['suggestion'] = "TRIGGER_GC_AND_COMPRESSION"
            return analysis
            
        # 2. Performance Stagnation Check
        # Calculate trend of success rate over last 10 steps
        if len(self.history_metrics) >= 10:
            recent_success = [m.get('task_success_rate', 0) for m in self.history_metrics[-10:]]
            avg_success = sum(recent_success) / len(recent_success)
            
            if avg_success < 0.5:
                analysis['status'] = "STAGNANT"
                analysis['suggestion'] = "SWITCH_STRATEGY"
            elif avg_success > 0.9:
                analysis['status'] = "THRIVING"
                analysis['suggestion'] = "INCREASE_DIFFICULTY" # Or optimize for speed
                
        return analysis

    def suggest_mutation(self, context: str) -> str:
        """
        Suggest a specific mutation based on context.
        """
        if context == "SWITCH_STRATEGY":
            return "Adopt 'Conservative' Strategy" if np.random.random() > 0.5 else "Adopt 'Aggressive' Strategy"
        return "No mutation"
