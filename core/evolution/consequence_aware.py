"""
Consequence-Aware Evolution Module (后果感知进化模块)

This module provides risk assessment and consequence prediction for AGI evolution decisions.
It integrates G-value (Expected Free Energy) evaluation to ensure that self-modifications
are safe and beneficial before being applied.

Core Principle: "Think before you act, predict before you evolve"

Author: AGI Evolution System
Version: 1.0.0
Date: 2025-07-17
"""

import logging
import math
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import numpy as np

from core.evolution.dynamics import EvolutionaryDynamics

logger = logging.getLogger("ConsequenceAwareEvolution")


class RiskLevel(Enum):
    """Risk level classification for evolution actions"""
    MINIMAL = "minimal"      # G < 0.2: Almost no risk
    LOW = "low"              # G < 0.5: Low risk, auto-approvable
    MEDIUM = "medium"        # G < 0.8: Medium risk, shadow test required
    HIGH = "high"            # G < 1.0: High risk, human review required
    CRITICAL = "critical"    # G >= 1.0: Critical risk, blocked


@dataclass
class ConsequencePrediction:
    """Prediction of consequences for an evolution action"""
    action_type: str
    target_file: str
    g_value: float  # Expected Free Energy (lower is better)
    risk_level: RiskLevel
    impact_assessment: Dict[str, Any]
    mitigation_strategies: List[str]
    recommendation: str
    confidence: float  # 0-1, how confident the prediction is


class ConsequenceAwareEvolution:
    """
    A module that evaluates the consequences of evolution actions before they are applied.
    
    It uses the G-value (Expected Free Energy) framework to assess:
    1. Risk: How much the action deviates from stable states
    2. Ambiguity: How uncertain the outcome is
    3. Impact: What systems are affected
    
    The goal is to minimize negative consequences while allowing beneficial evolution.
    """
    
    def __init__(self, dynamics: EvolutionaryDynamics = None):
        self.dynamics = dynamics or EvolutionaryDynamics()
        
        # System criticality weights (higher = more critical)
        self.criticality_weights = {
            "impl.py": 1.0,              # Evolution core - highest criticality
            "permission": 0.95,           # Permission system
            "seed.py": 0.9,              # Core seed
            "motivation.py": 0.85,        # Motivation core
            "AGI_Life_Engine.py": 0.9,   # Main engine
            "memory": 0.7,               # Memory systems
            "llm": 0.6,                  # LLM interface
            "tool": 0.5,                 # Tool systems
            "default": 0.3               # Other files
        }
        
        # Historical evolution outcomes (for learning)
        self.evolution_history: List[Dict[str, Any]] = []
        
        logger.info("🔮 ConsequenceAwareEvolution initialized")
    
    def _get_file_criticality(self, file_path: str) -> float:
        """Get the criticality weight for a file based on its name/path"""
        normalized_path = file_path.lower().replace("\\", "/")
        
        for key, weight in self.criticality_weights.items():
            if key.lower() in normalized_path:
                return weight
        
        return self.criticality_weights["default"]
    
    def _calculate_change_magnitude(self, original_code: str, new_code: str) -> float:
        """
        Calculate the magnitude of change between original and new code.
        Uses a combination of:
        - Line count difference
        - Character count difference
        - Structural changes (function/class additions/removals)
        """
        if not original_code:
            # New file creation is moderate risk
            return 0.5
        
        if not new_code:
            # File deletion is high risk
            return 1.0
        
        # Line-based change ratio
        orig_lines = len(original_code.splitlines())
        new_lines = len(new_code.splitlines())
        line_ratio = abs(new_lines - orig_lines) / max(orig_lines, 1)
        
        # Character-based change ratio
        orig_chars = len(original_code)
        new_chars = len(new_code)
        char_ratio = abs(new_chars - orig_chars) / max(orig_chars, 1)
        
        # Structural analysis (simple heuristic)
        orig_defs = original_code.count("def ") + original_code.count("class ")
        new_defs = new_code.count("def ") + new_code.count("class ")
        struct_ratio = abs(new_defs - orig_defs) / max(orig_defs, 1)
        
        # Weighted combination
        magnitude = (line_ratio * 0.3 + char_ratio * 0.3 + struct_ratio * 0.4)
        
        # Normalize to 0-1
        return min(1.0, magnitude)
    
    def _estimate_uncertainty(self, action_context: Dict[str, Any]) -> float:
        """
        Estimate the uncertainty/ambiguity of an evolution action.
        Higher uncertainty = higher risk.
        """
        uncertainty = 0.3  # Base uncertainty
        
        # Increase uncertainty for complex actions
        if action_context.get("involves_async", False):
            uncertainty += 0.1
        
        if action_context.get("modifies_imports", False):
            uncertainty += 0.15
        
        if action_context.get("modifies_class_hierarchy", False):
            uncertainty += 0.2
        
        # Decrease uncertainty if tests are available
        if action_context.get("has_tests", False):
            uncertainty -= 0.2
        
        # Decrease uncertainty based on shadow test availability
        if action_context.get("shadow_test_passed", False):
            uncertainty -= 0.3
        
        return max(0.0, min(1.0, uncertainty))
    
    def calculate_g_value(
        self,
        target_file: str,
        original_code: str,
        new_code: str,
        action_context: Optional[Dict[str, Any]] = None
    ) -> float:
        """
        Calculate the G-value (Expected Free Energy) for an evolution action.
        
        G = Risk + Ambiguity - Information_Gain
        
        Lower G is better (less risky, less uncertain, more informative).
        """
        action_context = action_context or {}
        
        # 1. Calculate Risk component
        criticality = self._get_file_criticality(target_file)
        change_magnitude = self._calculate_change_magnitude(original_code, new_code)
        risk = criticality * change_magnitude
        
        # 2. Calculate Ambiguity component
        uncertainty = self._estimate_uncertainty(action_context)
        ambiguity = uncertainty * criticality
        
        # 3. Calculate Information Gain (reduces G)
        # If we have tests and they passed, we gain information
        info_gain = 0.0
        if action_context.get("shadow_test_passed", False):
            info_gain += 0.3
        if action_context.get("performance_improvement", 0) > 0:
            info_gain += min(0.2, action_context["performance_improvement"] / 100)
        
        # G = Risk + Ambiguity - Information_Gain
        g_value = risk + ambiguity - info_gain
        
        return max(0.0, g_value)
    
    def classify_risk_level(self, g_value: float) -> RiskLevel:
        """Classify the risk level based on G-value"""
        if g_value < 0.2:
            return RiskLevel.MINIMAL
        elif g_value < 0.5:
            return RiskLevel.LOW
        elif g_value < 0.8:
            return RiskLevel.MEDIUM
        elif g_value < 1.0:
            return RiskLevel.HIGH
        else:
            return RiskLevel.CRITICAL
    
    def generate_mitigation_strategies(
        self,
        risk_level: RiskLevel,
        target_file: str,
        action_context: Dict[str, Any]
    ) -> List[str]:
        """Generate mitigation strategies based on risk level"""
        strategies = []
        
        if risk_level in [RiskLevel.MINIMAL, RiskLevel.LOW]:
            strategies.append("Standard shadow test verification")
        
        if risk_level in [RiskLevel.MEDIUM, RiskLevel.HIGH]:
            strategies.append("Mandatory shadow test with full context")
            strategies.append("Create backup before modification")
            
        if risk_level == RiskLevel.HIGH:
            strategies.append("Human review required before application")
            strategies.append("Staged rollout: test in isolated environment first")
        
        if risk_level == RiskLevel.CRITICAL:
            strategies.append("BLOCKED: Risk too high for automatic evolution")
            strategies.append("Manual intervention required")
            strategies.append("Consider alternative approaches")
        
        # File-specific mitigations
        if "impl.py" in target_file.lower():
            strategies.append("Meta-evolution safety: Double verification required")
            strategies.append("Ensure ShadowRunner is available")
        
        if "permission" in target_file.lower():
            strategies.append("Permission system change: Requires human review")
        
        return strategies
    
    def predict_consequences(
        self,
        action_type: str,
        target_file: str,
        original_code: str,
        new_code: str,
        action_context: Optional[Dict[str, Any]] = None
    ) -> ConsequencePrediction:
        """
        Main method: Predict the consequences of an evolution action.
        
        Returns a ConsequencePrediction object with:
        - G-value assessment
        - Risk level classification
        - Impact assessment
        - Mitigation strategies
        - Recommendation (proceed/review/block)
        """
        action_context = action_context or {}
        
        # Calculate G-value
        g_value = self.calculate_g_value(target_file, original_code, new_code, action_context)
        
        # Classify risk
        risk_level = self.classify_risk_level(g_value)
        
        # Impact assessment
        criticality = self._get_file_criticality(target_file)
        change_magnitude = self._calculate_change_magnitude(original_code, new_code)
        
        impact_assessment = {
            "file_criticality": criticality,
            "change_magnitude": change_magnitude,
            "affected_systems": self._identify_affected_systems(target_file),
            "estimated_recovery_time": self._estimate_recovery_time(risk_level),
        }
        
        # Generate mitigation strategies
        mitigation_strategies = self.generate_mitigation_strategies(
            risk_level, target_file, action_context
        )
        
        # Generate recommendation
        if risk_level == RiskLevel.MINIMAL:
            recommendation = "PROCEED: Safe to apply automatically"
        elif risk_level == RiskLevel.LOW:
            recommendation = "PROCEED_WITH_SHADOW_TEST: Apply after shadow verification"
        elif risk_level == RiskLevel.MEDIUM:
            recommendation = "CAUTION: Apply with enhanced verification"
        elif risk_level == RiskLevel.HIGH:
            recommendation = "REVIEW_REQUIRED: Human review before application"
        else:
            recommendation = "BLOCKED: Risk too high, manual intervention required"
        
        # Confidence based on available information
        confidence = 0.7  # Base confidence
        if action_context.get("has_tests", False):
            confidence += 0.15
        if action_context.get("shadow_test_passed", False):
            confidence += 0.1
        confidence = min(0.95, confidence)
        
        return ConsequencePrediction(
            action_type=action_type,
            target_file=target_file,
            g_value=g_value,
            risk_level=risk_level,
            impact_assessment=impact_assessment,
            mitigation_strategies=mitigation_strategies,
            recommendation=recommendation,
            confidence=confidence
        )
    
    def _identify_affected_systems(self, target_file: str) -> List[str]:
        """Identify which systems are affected by changes to a file"""
        affected = []
        normalized = target_file.lower().replace("\\", "/")
        
        if "evolution" in normalized or "impl.py" in normalized:
            affected.extend(["evolution_system", "sandbox_compiler", "shadow_runner"])
        
        if "memory" in normalized:
            affected.extend(["memory_system", "topology_core", "knowledge_graph"])
        
        if "permission" in normalized:
            affected.extend(["permission_system", "security_layer", "audit_log"])
        
        if "llm" in normalized:
            affected.extend(["llm_service", "chat_interface", "code_generation"])
        
        if "motivation" in normalized or "seed" in normalized:
            affected.extend(["motivation_core", "drive_system", "active_inference"])
        
        if not affected:
            affected.append("general")
        
        return affected
    
    def _estimate_recovery_time(self, risk_level: RiskLevel) -> str:
        """Estimate recovery time if something goes wrong"""
        recovery_times = {
            RiskLevel.MINIMAL: "< 1 minute (automatic rollback)",
            RiskLevel.LOW: "1-5 minutes (backup restoration)",
            RiskLevel.MEDIUM: "5-15 minutes (manual review + rollback)",
            RiskLevel.HIGH: "15-60 minutes (debugging + restoration)",
            RiskLevel.CRITICAL: "> 1 hour (extensive recovery needed)"
        }
        return recovery_times.get(risk_level, "unknown")
    
    def should_proceed(self, prediction: ConsequencePrediction) -> Tuple[bool, str]:
        """
        Decide whether to proceed with an evolution action based on the prediction.
        
        Returns: (should_proceed: bool, reason: str)
        """
        if prediction.risk_level == RiskLevel.CRITICAL:
            return False, "BLOCKED: Critical risk level - manual intervention required"
        
        if prediction.risk_level == RiskLevel.HIGH:
            # High risk requires human review
            return False, "REVIEW_REQUIRED: High risk - awaiting human approval"
        
        if prediction.risk_level in [RiskLevel.MINIMAL, RiskLevel.LOW, RiskLevel.MEDIUM]:
            return True, f"APPROVED: {prediction.recommendation}"
        
        return False, "UNKNOWN: Unable to determine safety"
    
    def record_outcome(
        self,
        prediction: ConsequencePrediction,
        actual_outcome: str,
        success: bool
    ):
        """
        Record the actual outcome of an evolution action for learning.
        This helps improve future predictions.
        """
        record = {
            "target_file": prediction.target_file,
            "predicted_g_value": prediction.g_value,
            "predicted_risk": prediction.risk_level.value,
            "actual_outcome": actual_outcome,
            "success": success,
            "prediction_accurate": (
                (success and prediction.risk_level in [RiskLevel.MINIMAL, RiskLevel.LOW]) or
                (not success and prediction.risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL])
            )
        }
        
        self.evolution_history.append(record)
        
        # Log for learning
        accuracy_str = "✅" if record["prediction_accurate"] else "❌"
        logger.info(
            f"📊 Evolution outcome recorded: {prediction.target_file} "
            f"- G={prediction.g_value:.2f}, Risk={prediction.risk_level.value}, "
            f"Success={success}, Prediction {accuracy_str}"
        )


# Singleton instance for global access
_consequence_evaluator: Optional[ConsequenceAwareEvolution] = None

def get_consequence_evaluator() -> ConsequenceAwareEvolution:
    """Get or create the global ConsequenceAwareEvolution instance"""
    global _consequence_evaluator
    if _consequence_evaluator is None:
        _consequence_evaluator = ConsequenceAwareEvolution()
    return _consequence_evaluator
