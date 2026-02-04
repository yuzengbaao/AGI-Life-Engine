"""
Unit tests for P1 Enhanced World Model module.

Tests cover:
- Initialization and configuration
- State updates and history management
- State prediction
- Counterfactual reasoning
- Error handling
- Health monitoring
"""

import pytest
import time
from enhanced_world_model import (
    EnhancedWorldModel,
    WorldModelException,
    WorldModelOperationError,
    WorldState
)


class TestEnhancedWorldModel:
    """Test suite for EnhancedWorldModel."""
    
    def test_initialize_and_health(self):
        """Test basic initialization and health check."""
        model = EnhancedWorldModel()
        
        # Initialize with custom config
        config = {"max_history": 50, "warmup_ms": 5, "metadata": {"env": "test"}}
        ok = model.initialize(config)
        
        assert ok is True
        
        # Check health status
        health = model.health_status()
        assert health["initialized"] is True
        assert health["state_count"] == 0
        assert health["has_current_state"] is True
        assert health["uptime_sec"] >= 0
    
    def test_update_state_basic(self):
        """Test basic state update functionality."""
        model = EnhancedWorldModel()
        model.initialize()
        
        # Update with observation
        observation = {
            "entities": {"player": {"position": [0, 0], "health": 100}},
            "relations": [("player", "at", "start")],
            "metadata": {"step": 1}
        }
        
        ok = model.update_state(observation)
        assert ok is True
        
        health = model.health_status()
        assert health["state_count"] == 1  # Previous state saved to history
    
    def test_update_state_capacity_management(self):
        """Test state history capacity management."""
        model = EnhancedWorldModel()
        model.initialize({"max_history": 3})
        
        # Add 5 states (should keep only last 3)
        for i in range(5):
            model.update_state({"entities": {"step": i}})
        
        health = model.health_status()
        assert health["state_count"] == 3  # Max capacity enforced
    
    def test_predict_future_state(self):
        """Test state prediction capability."""
        model = EnhancedWorldModel()
        model.initialize()
        
        # Set initial state
        model.update_state({"entities": {"item_count": 5}})
        
        # Predict after adding item
        action = {"add_entity": {"item_count": 6}}
        prediction = model.predict(action, steps=1)
        
        assert "predicted_state" in prediction
        assert "confidence" in prediction
        assert prediction["confidence"] > 0
        assert prediction["steps"] == 1
        assert prediction["predicted_state"]["entities"]["item_count"] == 6
    
    def test_counterfactual_reasoning(self):
        """Test counterfactual query 'what if...'."""
        model = EnhancedWorldModel()
        model.initialize()
        
        model.update_state({"entities": {"temperature": 20}})
        
        # Query: "What if temperature was 100?"
        condition = {
            "premise": "temperature=100",
            "hypothetical_entities": {"temperature": 100}
        }
        
        result = model.query_counterfactual(condition)
        
        assert "result" in result
        assert "reasoning" in result
        assert "confidence" in result
        assert result["confidence"] > 0
    
    def test_operations_require_initialization(self):
        """Test that operations fail before initialization."""
        model = EnhancedWorldModel()
        
        with pytest.raises(WorldModelOperationError, match="not initialized"):
            model.update_state({"entities": {}})
        
        with pytest.raises(WorldModelOperationError, match="not initialized"):
            model.predict({"action": "test"})
        
        with pytest.raises(WorldModelOperationError, match="not initialized"):
            model.query_counterfactual({"premise": "test"})
    
    def test_invalid_inputs_raise_errors(self):
        """Test validation of input types."""
        model = EnhancedWorldModel()
        model.initialize()
        
        # Non-dict observation
        with pytest.raises(WorldModelOperationError, match="must be a dictionary"):
            model.update_state("invalid")
        
        # Non-dict action
        with pytest.raises(WorldModelOperationError, match="must be a dictionary"):
            model.predict("invalid")
        
        # Non-dict condition
        with pytest.raises(WorldModelOperationError, match="must be a dictionary"):
            model.query_counterfactual("invalid")
    
    def test_double_initialization_is_safe(self):
        """Test that double initialization is safe (idempotent)."""
        model = EnhancedWorldModel()
        
        ok1 = model.initialize()
        ok2 = model.initialize()
        
        assert ok1 is True
        assert ok2 is True  # Second init returns True (already initialized)
    
    def test_shutdown(self):
        """Test clean shutdown and resource cleanup."""
        model = EnhancedWorldModel()
        model.initialize()
        
        # Add some states
        model.update_state({"entities": {"data": 1}})
        model.update_state({"entities": {"data": 2}})
        
        # Shutdown
        model.shutdown()
        
        # Verify cleanup
        health = model.health_status()
        assert health["initialized"] is False
        assert health["state_count"] == 0
        assert health["has_current_state"] is False
    
    def test_worldstate_dataclass(self):
        """Test WorldState dataclass creation."""
        state = WorldState(
            timestamp=time.time(),
            entities={"agent": {"x": 10}},
            relations=[("agent", "near", "goal")],
            metadata={"level": 1}
        )
        
        assert state.timestamp > 0
        assert "agent" in state.entities
        assert len(state.relations) == 1
        assert state.metadata["level"] == 1
