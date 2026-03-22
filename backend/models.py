from typing import List, Dict, Optional
from datetime import datetime
from pydantic import BaseModel

class Lesson(BaseModel):
    """Represents a lesson within a course"""
    lesson_number: int  # Sequential lesson number (1, 2, 3, etc.)
    title: str         # Lesson title
    lesson_link: Optional[str] = None  # URL link to the lesson

class Course(BaseModel):
    """Represents a complete course with its lessons"""
    title: str                 # Full course title (used as unique identifier)
    course_link: Optional[str] = None  # URL link to the course
    instructor: Optional[str] = None  # Course instructor name (optional metadata)
    lessons: List[Lesson] = [] # List of lessons in this course

class CourseChunk(BaseModel):
    """Represents a text chunk from a course for vector storage"""
    content: str                        # The actual text content
    course_title: str                   # Which course this chunk belongs to
    lesson_number: Optional[int] = None # Which lesson this chunk is from
    chunk_index: int                    # Position of this chunk in the document
    source_file: Optional[str] = None  # Original filename (e.g. course1_script.txt)
    start_line: Optional[int] = None   # First line of the lesson in the source file
    end_line: Optional[int] = None     # Last line of the lesson in the source file

class NewsArticle(BaseModel):
    """Represents a news article fetched from an RSS feed"""
    url: str                            # Canonical article URL (used as unique ID)
    title: str                          # Article headline
    summary: str                        # Article summary / description from RSS
    published_at: Optional[str] = None  # ISO-8601 publish datetime string
    fetched_at: str = ""                # ISO-8601 datetime when we fetched it
    source: str = "BBC News"            # Feed source name
    section: str = "general"            # News section (e.g. technology, world)

class NewsChunk(BaseModel):
    """Represents a text chunk from a news article for vector storage"""
    content: str                        # The chunk text
    url: str                            # Article URL (links chunk back to article)
    title: str                          # Article headline
    source: str                         # Feed source name
    section: str                        # News section
    published_at: Optional[str] = None  # ISO-8601 publish datetime
    fetched_at: str = ""                # ISO-8601 fetch datetime
    chunk_index: int = 0                # Position within the article