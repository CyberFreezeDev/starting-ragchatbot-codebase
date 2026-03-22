"""Tests for document_processor.py — file I/O + chunking logic."""
import os
import pytest
from document_processor import DocumentProcessor


@pytest.fixture
def proc():
    return DocumentProcessor(chunk_size=200, chunk_overlap=50)


@pytest.fixture
def large_proc():
    """Processor with large chunk size so most docs produce 1 chunk."""
    return DocumentProcessor(chunk_size=5000, chunk_overlap=100)


class TestChunkText:
    def test_basic_produces_chunks(self, proc):
        text = "Python is great. It is easy to learn. Many people use it daily."
        chunks = proc.chunk_text(text)
        assert len(chunks) >= 1

    def test_no_chunk_exceeds_size(self, proc):
        # Generate enough text to force multiple chunks
        sentences = ["This is sentence number %d in the test." % i for i in range(50)]
        text = " ".join(sentences)
        chunks = proc.chunk_text(text)
        for chunk in chunks:
            assert len(chunk) <= proc.chunk_size + 50  # small tolerance for last sentence

    def test_empty_string_returns_empty_list(self, proc):
        assert proc.chunk_text("") == []

    def test_whitespace_only_returns_empty(self, proc):
        assert proc.chunk_text("   \n\t  ") == []

    def test_single_sentence(self, proc):
        chunks = proc.chunk_text("Hello world.")
        assert len(chunks) == 1
        assert "Hello world" in chunks[0]

    def test_abbreviation_not_split(self, proc):
        # "Mr." should not be treated as a sentence boundary
        text = "Mr. Smith went to school. He learned a lot."
        chunks = proc.chunk_text(text)
        # Both sentences in one chunk (they're short enough)
        combined = " ".join(chunks)
        assert "Mr. Smith" in combined

    def test_overlap_shares_content(self):
        """Consecutive chunks should share content when overlap > 0."""
        proc = DocumentProcessor(chunk_size=100, chunk_overlap=60)
        sentences = [
            "First sentence here is long enough to matter.",
            "Second sentence also adds content to the chunk.",
            "Third sentence pushes us past the chunk boundary now.",
            "Fourth sentence starts the next chunk with overlap.",
        ]
        text = " ".join(sentences)
        chunks = proc.chunk_text(text)
        if len(chunks) >= 2:
            # Last sentence(s) of chunk 1 should appear at start of chunk 2
            last_words_of_first = chunks[0].split()[-3:]
            combined = " ".join(chunks[1].split()[:10])
            # At least one word from the end of chunk 1 appears in chunk 2
            assert any(w in combined for w in last_words_of_first)

    def test_no_overlap_no_repeated_content(self):
        proc = DocumentProcessor(chunk_size=100, chunk_overlap=0)
        sentences = ["Sentence %d is here for testing." % i for i in range(20)]
        text = " ".join(sentences)
        chunks = proc.chunk_text(text)
        # With 0 overlap, total character count should roughly equal input
        total_chars = sum(len(c) for c in chunks)
        assert total_chars <= len(text) + len(chunks) * 5  # small join overhead


class TestProcessCourseDocument:
    def test_metadata_extracted(self, large_proc, sample_course_txt):
        course, chunks = large_proc.process_course_document(sample_course_txt)
        assert course.title == "Test Python Course"
        assert course.course_link == "https://example.com/python"
        assert course.instructor == "Jane Doe"

    def test_lessons_parsed(self, large_proc, sample_course_txt):
        course, chunks = large_proc.process_course_document(sample_course_txt)
        assert len(course.lessons) == 3
        lesson_titles = [l.title for l in course.lessons]
        assert "Introduction" in lesson_titles
        assert "Data Types" in lesson_titles
        assert "Functions" in lesson_titles

    def test_chunks_created(self, large_proc, sample_course_txt):
        course, chunks = large_proc.process_course_document(sample_course_txt)
        assert len(chunks) > 0

    def test_source_file_set_on_chunks(self, large_proc, sample_course_txt):
        course, chunks = large_proc.process_course_document(sample_course_txt)
        for chunk in chunks:
            assert chunk.source_file == "test_course.txt"

    def test_line_numbers_set(self, large_proc, sample_course_txt):
        course, chunks = large_proc.process_course_document(sample_course_txt)
        for chunk in chunks:
            assert chunk.start_line is not None
            assert chunk.end_line is not None
            assert chunk.start_line <= chunk.end_line

    def test_line_numbers_positive(self, large_proc, sample_course_txt):
        course, chunks = large_proc.process_course_document(sample_course_txt)
        for chunk in chunks:
            assert chunk.start_line >= 1

    def test_no_lessons_falls_back_to_single_chunk(self, large_proc, tmp_path):
        content = """Course Title: No Lessons Course
Course Link: https://example.com
Course Instructor: Bob

This is just a big block of content with no lesson markers.
It should be treated as one document.
"""
        doc = tmp_path / "no_lessons.txt"
        doc.write_text(content, encoding="utf-8")
        course, chunks = large_proc.process_course_document(str(doc))
        assert len(chunks) >= 1
        assert course.title == "No Lessons Course"

    def test_missing_instructor_defaults(self, large_proc, tmp_path):
        content = """Course Title: Minimal Course
Course Link: https://example.com

Lesson 1: Only Lesson
Some content here.
"""
        doc = tmp_path / "minimal.txt"
        doc.write_text(content, encoding="utf-8")
        course, chunks = large_proc.process_course_document(str(doc))
        assert course.instructor is None

    def test_chunks_have_course_title(self, large_proc, sample_course_txt):
        course, chunks = large_proc.process_course_document(sample_course_txt)
        for chunk in chunks:
            assert chunk.course_title == "Test Python Course"

    def test_chunks_have_lesson_numbers(self, large_proc, sample_course_txt):
        course, chunks = large_proc.process_course_document(sample_course_txt)
        for chunk in chunks:
            assert chunk.lesson_number is not None

    def test_utf8_fallback_reads_file(self, large_proc, tmp_path):
        """File with invalid UTF-8 bytes should still be readable."""
        doc = tmp_path / "bad_encoding.txt"
        # Write valid content then patch in a bad byte via binary mode
        doc.write_bytes(
            b"Course Title: Encoding Test\nCourse Link: \nCourse Instructor: X\n\n"
            b"Lesson 1: Test\nContent with bad byte \x80 here.\n"
        )
        # Should not raise
        course, chunks = large_proc.process_course_document(str(doc))
        assert course.title == "Encoding Test"
