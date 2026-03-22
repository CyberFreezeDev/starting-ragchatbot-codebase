"""Tests for search_tools.py — mocks VectorStore."""
import pytest
from unittest.mock import MagicMock
from vector_store import SearchResults
from search_tools import CourseSearchTool, ToolManager, Tool


class TestCourseSearchToolDefinition:
    def test_definition_has_type_and_function(self, mock_vector_store):
        tool = CourseSearchTool(mock_vector_store)
        defn = tool.get_tool_definition()
        assert defn["type"] == "function"
        assert "function" in defn

    def test_definition_has_name(self, mock_vector_store):
        tool = CourseSearchTool(mock_vector_store)
        assert tool.get_tool_definition()["function"]["name"] == "search_course_content"

    def test_definition_has_parameters(self, mock_vector_store):
        tool = CourseSearchTool(mock_vector_store)
        params = tool.get_tool_definition()["function"]["parameters"]
        assert "query" in params["properties"]
        assert "query" in params["required"]

    def test_optional_params_present(self, mock_vector_store):
        tool = CourseSearchTool(mock_vector_store)
        params = tool.get_tool_definition()["function"]["parameters"]
        assert "course_name" in params["properties"]
        assert "lesson_number" in params["properties"]


class TestCourseSearchToolExecute:
    def test_execute_returns_non_empty_string(self, mock_vector_store):
        tool = CourseSearchTool(mock_vector_store)
        result = tool.execute(query="what is RAG")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_execute_calls_vector_store_search(self, mock_vector_store):
        tool = CourseSearchTool(mock_vector_store)
        tool.execute(query="RAG techniques", course_name="Advanced Retrieval")
        mock_vector_store.search.assert_called_once_with(
            query="RAG techniques",
            course_name="Advanced Retrieval",
            lesson_number=None,
        )

    def test_execute_empty_results_returns_message(self, mock_vector_store):
        mock_vector_store.search.return_value = SearchResults(
            documents=[], metadata=[], distances=[], error=None
        )
        tool = CourseSearchTool(mock_vector_store)
        result = tool.execute(query="xyz")
        assert "No relevant content found" in result

    def test_execute_error_propagates(self, mock_vector_store):
        mock_vector_store.search.return_value = SearchResults(
            documents=[], metadata=[], distances=[], error="ChromaDB unavailable"
        )
        tool = CourseSearchTool(mock_vector_store)
        result = tool.execute(query="anything")
        assert "ChromaDB unavailable" in result

    def test_sources_stored_after_execute(self, mock_vector_store):
        tool = CourseSearchTool(mock_vector_store)
        tool.execute(query="RAG")
        assert len(tool.last_sources) > 0

    def test_sources_deduplicated(self, mock_search_results_duplicates):
        store = MagicMock()
        store.search.return_value = mock_search_results_duplicates
        tool = CourseSearchTool(store)
        tool.execute(query="RAG")
        # 3 docs all same lesson → should collapse to 1 unique source
        assert len(tool.last_sources) == 1

    def test_source_includes_filename(self, mock_vector_store):
        tool = CourseSearchTool(mock_vector_store)
        tool.execute(query="RAG")
        assert any("course3_script.txt" in s for s in tool.last_sources)

    def test_source_includes_line_numbers(self, mock_vector_store):
        tool = CourseSearchTool(mock_vector_store)
        tool.execute(query="RAG")
        assert any("lines" in s for s in tool.last_sources)

    def test_source_includes_lesson_number(self, mock_vector_store):
        tool = CourseSearchTool(mock_vector_store)
        tool.execute(query="RAG")
        assert any("Lesson" in s for s in tool.last_sources)


class TestToolManager:
    def test_register_openai_format(self):
        manager = ToolManager()
        mock_tool = MagicMock(spec=Tool)
        mock_tool.get_tool_definition.return_value = {
            "type": "function",
            "function": {"name": "my_tool", "parameters": {}}
        }
        manager.register_tool(mock_tool)
        assert "my_tool" in manager.tools

    def test_register_flat_format(self):
        manager = ToolManager()
        mock_tool = MagicMock(spec=Tool)
        mock_tool.get_tool_definition.return_value = {
            "name": "flat_tool",
            "description": "A flat tool",
        }
        manager.register_tool(mock_tool)
        assert "flat_tool" in manager.tools

    def test_register_missing_name_raises(self):
        manager = ToolManager()
        mock_tool = MagicMock(spec=Tool)
        mock_tool.get_tool_definition.return_value = {"type": "function", "function": {}}
        with pytest.raises(ValueError, match="name"):
            manager.register_tool(mock_tool)

    def test_execute_unknown_tool_returns_error(self):
        manager = ToolManager()
        result = manager.execute_tool("nonexistent_tool", query="test")
        assert "not found" in result.lower() or "nonexistent_tool" in result

    def test_execute_calls_correct_tool(self, mock_vector_store):
        manager = ToolManager()
        tool = CourseSearchTool(mock_vector_store)
        manager.register_tool(tool)
        result = manager.execute_tool("search_course_content", query="test")
        assert isinstance(result, str)

    def test_get_last_sources_empty_before_search(self):
        manager = ToolManager()
        assert manager.get_last_sources() == []

    def test_get_last_sources_after_search(self, mock_vector_store):
        manager = ToolManager()
        tool = CourseSearchTool(mock_vector_store)
        manager.register_tool(tool)
        manager.execute_tool("search_course_content", query="RAG")
        assert len(manager.get_last_sources()) > 0

    def test_reset_sources_clears(self, mock_vector_store):
        manager = ToolManager()
        tool = CourseSearchTool(mock_vector_store)
        manager.register_tool(tool)
        manager.execute_tool("search_course_content", query="RAG")
        manager.reset_sources()
        assert manager.get_last_sources() == []

    def test_get_tool_definitions_returns_list(self, mock_vector_store):
        manager = ToolManager()
        tool = CourseSearchTool(mock_vector_store)
        manager.register_tool(tool)
        defs = manager.get_tool_definitions()
        assert isinstance(defs, list)
        assert len(defs) == 1
