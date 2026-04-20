from __future__ import annotations

import tempfile
import unittest

from whesper.memory import MemoryService, MemoryStore, extract_memory_candidates


class MemoryTests(unittest.TestCase):
    def test_extract_memory_candidates_finds_preference_and_recent_event(self) -> None:
        candidates = extract_memory_candidates(
            "I like jasmine tea. Recently I have been sleeping badly."
        )

        contents = {candidate.content for candidate in candidates}
        memory_types = {candidate.memory_type for candidate in candidates}

        self.assertIn("The user likes jasmine tea.", contents)
        self.assertIn("preference_memory", memory_types)
        self.assertIn("episodic_memory", memory_types)

    def test_relevant_memories_prefers_keyword_matches(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = MemoryStore(tmpdir)
            service = MemoryService(store)
            store.remember(
                memory_type="preference_memory",
                title="Preference",
                content="The user likes jasmine tea.",
                source="user_message",
                confidence=0.8,
                session_id="main",
            )
            store.remember(
                memory_type="profile_memory",
                title="Location",
                content="The user lives in Shanghai.",
                source="user_message",
                confidence=0.8,
                session_id="main",
            )

            memories = service.relevant_memories("What tea should I drink?", limit=1)

        self.assertEqual(len(memories), 1)
        self.assertEqual(memories[0].content, "The user likes jasmine tea.")

    def test_capture_tool_result_creates_short_term_cup_device_memories(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = MemoryStore(tmpdir)
            service = MemoryService(store)

            saved = service.capture_tool_result(
                "main",
                "control_cup",
                (
                    '{"ok": true, "action": "set_led", "requested": {"color": "#0000ff"}, '
                    '"result": {"led_color": "#0000FF", "led_blink_hz": 1}}'
                ),
            )

            contents = {memory.content for memory in saved}
            memory_types = {memory.memory_type for memory in saved}

        self.assertIn("device_state_memory", memory_types)
        self.assertIn("device_preference_memory", memory_types)
        self.assertIn("The CUP LED is currently set to blue (蓝色) (#0000ff).", contents)
        self.assertIn(
            "The user recently chose blue (蓝色) (#0000ff) lighting for the CUP.",
            contents,
        )

    def test_relevant_memories_prefers_same_session_recent_device_context(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = MemoryStore(tmpdir)
            service = MemoryService(store)
            service.capture_tool_result(
                "main",
                "control_cup",
                (
                    '{"ok": true, "action": "apply_scene", "scene": "gentle", '
                    '"led": {"color": "#ffb36b"}, "motor": {"target_velocity": 40, "enabled": true}}'
                ),
            )
            service.capture_tool_result(
                "other",
                "control_cup",
                (
                    '{"ok": true, "action": "set_led", "requested": {"color": "#ff0000"}, '
                    '"result": {"led_color": "#FF0000"}}'
                ),
            )

            memories = service.relevant_memories(
                "再温柔一点",
                session_id="main",
                limit=2,
            )

        self.assertGreaterEqual(len(memories), 1)
        self.assertTrue(all(memory.session_id == "main" for memory in memories))


if __name__ == "__main__":
    unittest.main()
