from typing import Any, Optional, Dict


class HotSwapper:
    def __init__(self, agi_system: Any, coordinator: Optional[Any] = None):
        self.agi = agi_system
        self.coordinator = coordinator
        self._backup: Dict[str, Any] = {}

    def replace_attr(self, attr: str, new_obj: Any) -> Dict[str, Any]:
        previous = getattr(self.agi, attr, None)
        self._backup[attr] = previous
        setattr(self.agi, attr, new_obj)
        return {"success": True, "attr": attr}

    def rollback_attr(self, attr: str) -> Dict[str, Any]:
        if attr not in self._backup:
            return {"success": False, "error": "no backup for attr"}
        setattr(self.agi, attr, self._backup[attr])
        return {"success": True, "attr": attr}

    def register_component(self, key: str, instance: Any) -> Dict[str, Any]:
        if self.coordinator and hasattr(self.coordinator, "register_component"):
            self.coordinator.register_component(key, instance)
            return {"success": True, "via": "coordinator"}
        setattr(self.agi, key, instance)
        return {"success": True, "via": "attribute"}

