import unittest
from unittest.mock import patch

from agi_terminal_tool import TerminalTool


class MockFirmware:
    """Captures terminal audit events and returns predetermined decisions."""

    def __init__(self, decision_sequence):
        self.decision_sequence = list(decision_sequence)
        self.events = []
        self.contexts = []

    def evaluate_terminal_command(self, command, risk_level, safe_mode, context):
        self.contexts.append({
            "command": command,
            "risk_level": risk_level,
            "safe_mode": safe_mode,
            "context": context,
        })
        if not self.decision_sequence:
            return "allow"
        return self.decision_sequence.pop(0)

    def record_terminal_event(self, event_type, payload):
        self.events.append((event_type, payload))


class TerminalPermissionTests(unittest.TestCase):
    def _make_tool(self, decision):
        firmware = MockFirmware([decision])
        tool = TerminalTool(permission_firmware=firmware)
        tool.enable_persistent_sessions = False  # Avoid launching real shells in tests
        return tool, firmware

    def test_command_allowed_runs_normally(self):
        tool, firmware = self._make_tool("allow")
        with patch.object(TerminalTool, "_run_command_with_session", return_value=("ok", "", 0)) as mocked_run:
            result = tool.execute(command="echo allow")
        self.assertTrue(result.success)
        mocked_run.assert_called_once()
        event_types = [evt for evt, _ in firmware.events]
        self.assertIn("terminal-command-start", event_types)
        self.assertIn("terminal-command-approved", event_types)
        self.assertIn("terminal-command-finished", event_types)
        self.assertTrue(firmware.contexts)
        self.assertTrue(firmware.contexts[0]["safe_mode"])  # default safe_mode must propagate

    def test_warn_decision_logs_warning_but_runs(self):
        tool, firmware = self._make_tool("warn")
        with patch.object(TerminalTool, "_run_command_with_session", return_value=("warn", "", 0)) as mocked_run:
            result = tool.execute(command="echo warn")
        self.assertTrue(result.success)
        mocked_run.assert_called_once()
        event_types = [evt for evt, _ in firmware.events]
        self.assertIn("terminal-command-warn", event_types)
        self.assertIn("terminal-command-finished", event_types)

    def test_deny_decision_blocks_execution(self):
        tool, firmware = self._make_tool("deny")
        with patch.object(TerminalTool, "_run_command_with_session", return_value=("denied", "", 0)) as mocked_run:
            result = tool.execute(command="echo deny")
        self.assertFalse(result.success)
        mocked_run.assert_not_called()
        event_types = [evt for evt, _ in firmware.events]
        self.assertIn("terminal-command-denied", event_types)
        self.assertIn("terminal-command-permission-block", event_types)


if __name__ == "__main__":
    unittest.main()
