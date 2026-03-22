"""Tests for ai_generator.py — mocks openai.OpenAI."""
import json
import pytest
from unittest.mock import MagicMock, patch, PropertyMock
from ai_generator import AIGenerator


def _make_response(content="Answer text", tool_calls=None, finish_reason="stop"):
    """Build a mock openai ChatCompletion response."""
    message = MagicMock()
    message.content = content
    message.tool_calls = tool_calls or []
    choice = MagicMock()
    choice.message = message
    choice.finish_reason = finish_reason
    response = MagicMock()
    response.choices = [choice]
    return response


def _make_tool_call(name="search_course_content", arguments='{"query": "RAG"}', tc_id="call_1"):
    """Build a mock tool call object."""
    tc = MagicMock()
    tc.id = tc_id
    tc.function.name = name
    tc.function.arguments = arguments
    return tc


@pytest.fixture
def mock_client():
    """Patch openai.OpenAI so no real HTTP calls are made."""
    with patch("ai_generator.OpenAI") as MockOpenAI:
        client = MagicMock()
        MockOpenAI.return_value = client
        yield client


@pytest.fixture
def generator(mock_client):
    return AIGenerator(base_url="http://localhost:11434/v1", model="qwen2.5")


class TestGenerateResponseNoTools:
    def test_returns_content_string(self, generator, mock_client):
        mock_client.chat.completions.create.return_value = _make_response("Hello!")
        result = generator.generate_response("What is Python?")
        assert result == "Hello!"

    def test_no_history_uses_base_prompt(self, generator, mock_client):
        mock_client.chat.completions.create.return_value = _make_response("ok")
        generator.generate_response("test")
        call_args = mock_client.chat.completions.create.call_args
        messages = call_args.kwargs.get("messages") or call_args[1].get("messages") or call_args[0][0]
        system_msg = next(m for m in messages if m["role"] == "system")
        assert "Previous conversation" not in system_msg["content"]

    def test_history_prepended_to_system_prompt(self, generator, mock_client):
        mock_client.chat.completions.create.return_value = _make_response("ok")
        generator.generate_response("test", conversation_history="User: hi\nAssistant: hello")
        call_args = mock_client.chat.completions.create.call_args
        messages = call_args.kwargs.get("messages") or call_args[1].get("messages") or call_args[0][0]
        system_msg = next(m for m in messages if m["role"] == "system")
        assert "Previous conversation" in system_msg["content"]

    def test_tools_added_to_api_params_when_provided(self, generator, mock_client):
        mock_client.chat.completions.create.return_value = _make_response("ok")
        tools = [{"type": "function", "function": {"name": "search"}}]
        generator.generate_response("test", tools=tools)
        call_args = mock_client.chat.completions.create.call_args
        kwargs = call_args.kwargs if call_args.kwargs else call_args[1]
        assert "tools" in kwargs
        assert kwargs.get("tool_choice") == "auto"

    def test_no_tools_when_not_provided(self, generator, mock_client):
        mock_client.chat.completions.create.return_value = _make_response("ok")
        generator.generate_response("test")
        call_args = mock_client.chat.completions.create.call_args
        kwargs = call_args.kwargs if call_args.kwargs else call_args[1]
        assert "tools" not in kwargs


class TestToolCallDetection:
    def test_tool_call_triggers_handle_execution(self, generator, mock_client):
        tc = _make_tool_call()
        first_resp = _make_response(tool_calls=[tc], finish_reason="tool_calls")
        second_resp = _make_response("Final answer")
        mock_client.chat.completions.create.side_effect = [first_resp, second_resp]

        tool_manager = MagicMock()
        tool_manager.execute_tool.return_value = "search results"

        result = generator.generate_response("RAG query", tools=[{}], tool_manager=tool_manager)
        assert result == "Final answer"
        assert mock_client.chat.completions.create.call_count == 2

    def test_finish_reason_stop_with_tool_calls_still_detected(self, generator, mock_client):
        """Ollama quirk: finish_reason='stop' but tool_calls present."""
        tc = _make_tool_call()
        first_resp = _make_response(tool_calls=[tc], finish_reason="stop")
        second_resp = _make_response("Ollama final answer")
        mock_client.chat.completions.create.side_effect = [first_resp, second_resp]

        tool_manager = MagicMock()
        tool_manager.execute_tool.return_value = "results"

        result = generator.generate_response("query", tools=[{}], tool_manager=tool_manager)
        assert result == "Ollama final answer"

    def test_no_tool_calls_skips_second_call(self, generator, mock_client):
        mock_client.chat.completions.create.return_value = _make_response("direct")
        tool_manager = MagicMock()
        result = generator.generate_response("query", tools=[{}], tool_manager=tool_manager)
        assert mock_client.chat.completions.create.call_count == 1
        assert result == "direct"


class TestHandleToolExecution:
    def test_empty_args_fallback_to_user_query(self, generator, mock_client):
        """If model sends empty arguments, use the original query as fallback."""
        tc = _make_tool_call(arguments="{}")
        first_resp = _make_response(tool_calls=[tc], finish_reason="tool_calls")
        second_resp = _make_response("result")
        mock_client.chat.completions.create.side_effect = [first_resp, second_resp]

        tool_manager = MagicMock()
        tool_manager.execute_tool.return_value = "data"

        generator.generate_response("my user query", tools=[{}], tool_manager=tool_manager)

        # The fallback query should be the user's original message (wrapped in the prompt)
        call_kwargs = tool_manager.execute_tool.call_args
        assert "my user query" in str(call_kwargs)

    def test_tool_result_added_to_followup_messages(self, generator, mock_client):
        tc = _make_tool_call(arguments='{"query": "RAG"}')
        first_resp = _make_response(tool_calls=[tc], finish_reason="tool_calls")
        second_resp = _make_response("synthesized")
        mock_client.chat.completions.create.side_effect = [first_resp, second_resp]

        tool_manager = MagicMock()
        tool_manager.execute_tool.return_value = "tool output here"

        generator.generate_response("test", tools=[{}], tool_manager=tool_manager)

        # Second API call should include a "tool" role message
        second_call_args = mock_client.chat.completions.create.call_args
        messages = second_call_args.kwargs.get("messages") or second_call_args[1].get("messages") or second_call_args[0][0]
        roles = [m["role"] if isinstance(m, dict) else getattr(m, "role", None) for m in messages]
        assert "tool" in roles

    def test_malformed_json_args_raises(self, generator, mock_client):
        tc = _make_tool_call(arguments="not valid json {{{")
        first_resp = _make_response(tool_calls=[tc], finish_reason="tool_calls")
        mock_client.chat.completions.create.return_value = first_resp

        tool_manager = MagicMock()
        with pytest.raises(json.JSONDecodeError):
            generator.generate_response("query", tools=[{}], tool_manager=tool_manager)
