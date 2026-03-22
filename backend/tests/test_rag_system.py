"""Tests for rag_system.py — integration-style with mocked components."""
import os
import pytest
from unittest.mock import MagicMock, patch
from models import Course, Lesson, CourseChunk


@pytest.fixture
def mock_config():
    config = MagicMock()
    config.CHUNK_SIZE = 800
    config.CHUNK_OVERLAP = 100
    config.CHROMA_PATH = "./chroma_db_test"
    config.EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    config.MAX_RESULTS = 5
    config.OLLAMA_BASE_URL = "http://localhost:11434/v1"
    config.OLLAMA_MODEL = "qwen2.5"
    config.MAX_HISTORY = 2
    return config


@pytest.fixture
def sample_course():
    course = Course(title="Test Course", instructor="Bob")
    course.lessons.append(Lesson(lesson_number=1, title="Intro"))
    return course


@pytest.fixture
def sample_chunks(sample_course):
    return [
        CourseChunk(
            content="chunk content",
            course_title=sample_course.title,
            chunk_index=0,
            lesson_number=1,
            source_file="test.txt",
            start_line=5,
            end_line=20,
        )
    ]


@pytest.fixture
def rag(mock_config, sample_course, sample_chunks):
    """RAGSystem with all heavy dependencies patched out."""
    with patch("rag_system.DocumentProcessor") as MockDP, \
         patch("rag_system.VectorStore") as MockVS, \
         patch("rag_system.AIGenerator") as MockAI:

        mock_dp = MockDP.return_value
        mock_vs = MockVS.return_value
        mock_ai = MockAI.return_value

        mock_dp.process_course_document.return_value = (sample_course, sample_chunks)
        mock_vs.get_existing_course_titles.return_value = []
        mock_vs.get_course_count.return_value = 1
        mock_ai.generate_response.return_value = "Here is the answer."

        from rag_system import RAGSystem
        system = RAGSystem(mock_config)

        # Expose mocks for assertions
        system._mock_dp = mock_dp
        system._mock_vs = mock_vs
        system._mock_ai = mock_ai

        yield system


class TestAddCourseDocument:
    def test_success_returns_course_and_chunk_count(self, rag):
        course, count = rag.add_course_document("fake/path/course.txt")
        assert course is not None
        assert count == 1

    def test_metadata_and_content_added_to_store(self, rag):
        rag.add_course_document("fake/path/course.txt")
        rag._mock_vs.add_course_metadata.assert_called_once()
        rag._mock_vs.add_course_content.assert_called_once()

    def test_bad_path_returns_none_zero(self, rag):
        rag._mock_dp.process_course_document.side_effect = FileNotFoundError("not found")
        course, count = rag.add_course_document("nonexistent.txt")
        assert course is None
        assert count == 0

    def test_processing_exception_returns_none_zero(self, rag):
        rag._mock_dp.process_course_document.side_effect = Exception("parse error")
        course, count = rag.add_course_document("bad.txt")
        assert course is None
        assert count == 0


class TestAddCourseFolder:
    def test_nonexistent_folder_returns_zeros(self, rag):
        courses, chunks = rag.add_course_folder("/nonexistent/path")
        assert courses == 0
        assert chunks == 0

    def test_loads_txt_files(self, rag, tmp_path, sample_course, sample_chunks):
        (tmp_path / "course1.txt").write_text("Course Title: A\nCourse Link:\nCourse Instructor:\n\nLesson 1: X\nContent.")
        rag._mock_dp.process_course_document.return_value = (sample_course, sample_chunks)
        courses, chunks = rag.add_course_folder(str(tmp_path))
        assert courses == 1

    def test_skips_duplicate_courses(self, rag, tmp_path, sample_course, sample_chunks):
        (tmp_path / "course1.txt").write_text("content")
        # Pretend this course already exists
        rag._mock_vs.get_existing_course_titles.return_value = [sample_course.title]
        rag._mock_dp.process_course_document.return_value = (sample_course, sample_chunks)
        courses, chunks = rag.add_course_folder(str(tmp_path))
        assert courses == 0

    def test_clear_existing_calls_clear_all_data(self, rag, tmp_path):
        courses, chunks = rag.add_course_folder(str(tmp_path), clear_existing=True)
        rag._mock_vs.clear_all_data.assert_called_once()

    def test_ignores_non_course_files(self, rag, tmp_path):
        (tmp_path / "readme.md").write_text("# readme")
        (tmp_path / "data.bin").write_bytes(b"\x00\x01\x02")
        courses, chunks = rag.add_course_folder(str(tmp_path))
        rag._mock_dp.process_course_document.assert_not_called()
        assert courses == 0


class TestQuery:
    def test_returns_answer_string(self, rag):
        sid = rag.session_manager.create_session()
        answer, sources = rag.query("What is RAG?", session_id=sid)
        assert isinstance(answer, str)
        assert len(answer) > 0

    def test_returns_sources_list(self, rag):
        sid = rag.session_manager.create_session()
        answer, sources = rag.query("What is RAG?", session_id=sid)
        assert isinstance(sources, list)

    def test_works_without_session_id(self, rag):
        answer, sources = rag.query("general question")
        assert answer == "Here is the answer."

    def test_creates_history_after_query(self, rag):
        sid = rag.session_manager.create_session()
        rag.query("First question", session_id=sid)
        history = rag.session_manager.get_conversation_history(sid)
        assert history is not None
        assert "First question" in history

    def test_history_passed_to_ai_on_second_query(self, rag):
        sid = rag.session_manager.create_session()
        rag.query("First question", session_id=sid)
        rag.query("Second question", session_id=sid)
        # On the second call, conversation_history should not be None
        second_call = rag._mock_ai.generate_response.call_args_list[1]
        kwargs = second_call.kwargs if second_call.kwargs else second_call[1]
        assert kwargs.get("conversation_history") is not None

    def test_sources_reset_after_query(self, rag):
        sid = rag.session_manager.create_session()
        rag.query("test", session_id=sid)
        # After query, tool sources should be reset
        assert rag.tool_manager.get_last_sources() == []


class TestGetCourseAnalytics:
    def test_returns_total_courses(self, rag):
        analytics = rag.get_course_analytics()
        assert "total_courses" in analytics
        assert analytics["total_courses"] == 1

    def test_returns_course_titles(self, rag):
        rag._mock_vs.get_existing_course_titles.return_value = ["Course A", "Course B"]
        analytics = rag.get_course_analytics()
        assert "course_titles" in analytics
        assert "Course A" in analytics["course_titles"]
