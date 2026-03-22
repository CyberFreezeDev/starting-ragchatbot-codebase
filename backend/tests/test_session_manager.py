"""Tests for session_manager.py — pure Python, no mocking needed."""
import pytest
from session_manager import SessionManager


class TestCreateSession:
    def test_returns_string_id(self):
        sm = SessionManager()
        sid = sm.create_session()
        assert isinstance(sid, str)
        assert sid.startswith("session_")

    def test_unique_ids(self):
        sm = SessionManager()
        ids = {sm.create_session() for _ in range(5)}
        assert len(ids) == 5


class TestAddMessage:
    def test_add_to_existing_session(self):
        sm = SessionManager()
        sid = sm.create_session()
        sm.add_message(sid, "user", "hello")
        history = sm.get_conversation_history(sid)
        assert "hello" in history

    def test_auto_creates_session(self):
        sm = SessionManager()
        # Writing to an unknown session ID should not raise
        sm.add_message("phantom_session", "user", "hi")
        history = sm.get_conversation_history("phantom_session")
        assert history is not None

    def test_role_preserved(self):
        sm = SessionManager()
        sid = sm.create_session()
        sm.add_message(sid, "assistant", "I can help")
        history = sm.get_conversation_history(sid)
        assert "Assistant" in history


class TestHistoryTruncation:
    def test_respects_max_history(self):
        sm = SessionManager(max_history=2)
        sid = sm.create_session()
        # Add 3 exchanges = 6 messages; limit is 2*2=4
        for i in range(3):
            sm.add_exchange(sid, f"q{i}", f"a{i}")
        messages = sm.sessions[sid]
        assert len(messages) == 4  # only last 2 exchanges retained

    def test_oldest_messages_dropped(self):
        sm = SessionManager(max_history=1)
        sid = sm.create_session()
        sm.add_exchange(sid, "first question", "first answer")
        sm.add_exchange(sid, "second question", "second answer")
        history = sm.get_conversation_history(sid)
        assert "first question" not in history
        assert "second question" in history

    def test_max_history_zero_drops_all(self):
        sm = SessionManager(max_history=0)
        sid = sm.create_session()
        sm.add_exchange(sid, "q", "a")
        # With max_history=0, limit = 0*2=0 — all messages truncated
        messages = sm.sessions[sid]
        assert len(messages) == 0


class TestGetConversationHistory:
    def test_unknown_session_returns_none(self):
        sm = SessionManager()
        assert sm.get_conversation_history("nonexistent") is None

    def test_none_session_id_returns_none(self):
        sm = SessionManager()
        assert sm.get_conversation_history(None) is None

    def test_empty_session_returns_none(self):
        sm = SessionManager()
        sid = sm.create_session()
        assert sm.get_conversation_history(sid) is None

    def test_formatted_output_labels(self):
        sm = SessionManager()
        sid = sm.create_session()
        sm.add_exchange(sid, "What is Python?", "A programming language.")
        history = sm.get_conversation_history(sid)
        assert "User:" in history
        assert "Assistant:" in history

    def test_multiple_exchanges_in_history(self):
        sm = SessionManager(max_history=5)
        sid = sm.create_session()
        sm.add_exchange(sid, "q1", "a1")
        sm.add_exchange(sid, "q2", "a2")
        history = sm.get_conversation_history(sid)
        assert "q1" in history
        assert "q2" in history


class TestClearSession:
    def test_clear_empties_history(self):
        sm = SessionManager()
        sid = sm.create_session()
        sm.add_exchange(sid, "q", "a")
        sm.clear_session(sid)
        assert sm.get_conversation_history(sid) is None

    def test_clear_nonexistent_session_does_not_raise(self):
        sm = SessionManager()
        sm.clear_session("ghost_session")  # should not raise
