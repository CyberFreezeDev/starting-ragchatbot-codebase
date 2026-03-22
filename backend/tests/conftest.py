"""
Shared pytest fixtures for backend unit tests.
"""
import os
import sys
import tempfile
import pytest
from unittest.mock import MagicMock

# Ensure the backend directory is on the path so imports work
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from vector_store import SearchResults


@pytest.fixture
def sample_course_txt(tmp_path):
    """Create a temp .txt file in the expected course document format."""
    content = """Course Title: Test Python Course
Course Link: https://example.com/python
Course Instructor: Jane Doe

Lesson 0: Introduction
Lesson Link: https://example.com/python/lesson0
This is the intro content. It covers the basics of Python. Variables are fundamental.

Lesson 1: Data Types
Lesson Link: https://example.com/python/lesson1
Python has several built-in data types. Integers store whole numbers. Floats store decimals. Strings store text.

Lesson 2: Functions
Lesson Link: https://example.com/python/lesson2
Functions are reusable blocks of code. You define them with the def keyword. They can take parameters and return values.
"""
    doc = tmp_path / "test_course.txt"
    doc.write_text(content, encoding="utf-8")
    return str(doc)


@pytest.fixture
def mock_search_results():
    """A real SearchResults with 2 docs and metadata."""
    return SearchResults(
        documents=[
            "Lesson 0 content: RAG is a technique for grounding LLM responses.",
            "Lesson 1 content: Vector search finds semantically similar documents.",
        ],
        metadata=[
            {
                "course_title": "Advanced Retrieval for AI",
                "lesson_number": 0,
                "source_file": "course3_script.txt",
                "start_line": 7,
                "end_line": 10,
            },
            {
                "course_title": "Advanced Retrieval for AI",
                "lesson_number": 1,
                "source_file": "course3_script.txt",
                "start_line": 12,
                "end_line": 50,
            },
        ],
        distances=[0.12, 0.25],
        error=None,
    )


@pytest.fixture
def mock_search_results_duplicates():
    """SearchResults where all 3 docs map to the same lesson (duplicate sources)."""
    meta = {
        "course_title": "Advanced Retrieval for AI",
        "lesson_number": 0,
        "source_file": "course3_script.txt",
        "start_line": 7,
        "end_line": 10,
    }
    return SearchResults(
        documents=["chunk A", "chunk B", "chunk C"],
        metadata=[meta, meta, meta],
        distances=[0.1, 0.2, 0.3],
        error=None,
    )


@pytest.fixture
def mock_vector_store(mock_search_results):
    """MagicMock of VectorStore with sensible defaults."""
    store = MagicMock()
    store.search.return_value = mock_search_results
    store.get_existing_course_titles.return_value = []
    store.get_course_count.return_value = 2
    return store
