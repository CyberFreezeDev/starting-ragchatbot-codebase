"""Tests for models.py — Pydantic validation."""
import pytest
from pydantic import ValidationError
from models import Lesson, Course, CourseChunk


class TestLesson:
    def test_valid_lesson(self):
        lesson = Lesson(lesson_number=1, title="Intro")
        assert lesson.lesson_number == 1
        assert lesson.title == "Intro"
        assert lesson.lesson_link is None

    def test_lesson_with_link(self):
        lesson = Lesson(lesson_number=2, title="Deep Dive", lesson_link="https://example.com")
        assert lesson.lesson_link == "https://example.com"

    def test_lesson_missing_number_raises(self):
        with pytest.raises(ValidationError):
            Lesson(title="No Number")

    def test_lesson_missing_title_raises(self):
        with pytest.raises(ValidationError):
            Lesson(lesson_number=1)


class TestCourse:
    def test_valid_course_defaults(self):
        course = Course(title="My Course")
        assert course.title == "My Course"
        assert course.course_link is None
        assert course.instructor is None
        assert course.lessons == []

    def test_course_with_all_fields(self):
        course = Course(
            title="Full Course",
            course_link="https://example.com",
            instructor="Alice",
        )
        assert course.instructor == "Alice"

    def test_course_missing_title_raises(self):
        with pytest.raises(ValidationError):
            Course()

    def test_course_lessons_mutable(self):
        course = Course(title="X")
        lesson = Lesson(lesson_number=1, title="First")
        course.lessons.append(lesson)
        assert len(course.lessons) == 1


class TestCourseChunk:
    def test_valid_chunk_minimal(self):
        chunk = CourseChunk(content="hello", course_title="Python", chunk_index=0)
        assert chunk.content == "hello"
        assert chunk.lesson_number is None
        assert chunk.source_file is None
        assert chunk.start_line is None
        assert chunk.end_line is None

    def test_chunk_with_all_source_fields(self):
        chunk = CourseChunk(
            content="text",
            course_title="ML",
            chunk_index=3,
            lesson_number=2,
            source_file="ml_course.txt",
            start_line=10,
            end_line=50,
        )
        assert chunk.source_file == "ml_course.txt"
        assert chunk.start_line == 10
        assert chunk.end_line == 50

    def test_chunk_missing_content_raises(self):
        with pytest.raises(ValidationError):
            CourseChunk(course_title="X", chunk_index=0)

    def test_chunk_missing_course_title_raises(self):
        with pytest.raises(ValidationError):
            CourseChunk(content="text", chunk_index=0)
