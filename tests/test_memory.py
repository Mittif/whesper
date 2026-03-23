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


if __name__ == "__main__":
    unittest.main()
