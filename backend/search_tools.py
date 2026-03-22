from typing import Dict, Any, Optional, Protocol
from abc import ABC, abstractmethod
from vector_store import VectorStore, SearchResults


class Tool(ABC):
    """Abstract base class for all tools"""
    
    @abstractmethod
    def get_tool_definition(self) -> Dict[str, Any]:
        """Return Anthropic tool definition for this tool"""
        pass
    
    @abstractmethod
    def execute(self, **kwargs) -> str:
        """Execute the tool with given parameters"""
        pass


class CourseSearchTool(Tool):
    """Tool for searching course content with semantic course name matching"""
    
    def __init__(self, vector_store: VectorStore):
        self.store = vector_store
        self.last_sources = []  # Track sources from last search
    
    def get_tool_definition(self) -> Dict[str, Any]:
        """Return OpenAI-compatible tool definition for this tool"""
        return {
            "type": "function",
            "function": {
                "name": "search_course_content",
                "description": "Search course materials with smart course name matching and lesson filtering",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "What to search for in the course content"
                        },
                        "course_name": {
                            "type": "string",
                            "description": "Course title (partial matches work, e.g. 'MCP', 'Introduction')"
                        },
                        "lesson_number": {
                            "type": "integer",
                            "description": "Specific lesson number to search within (e.g. 1, 2, 3)"
                        }
                    },
                    "required": ["query"]
                }
            }
        }
    
    def execute(self, query: str, course_name: Optional[str] = None, lesson_number: Optional[int] = None) -> str:
        """
        Execute the search tool with given parameters.
        
        Args:
            query: What to search for
            course_name: Optional course filter
            lesson_number: Optional lesson filter
            
        Returns:
            Formatted search results or error message
        """
        
        # Use the vector store's unified search interface
        results = self.store.search(
            query=query,
            course_name=course_name,
            lesson_number=lesson_number
        )
        
        # Handle errors
        if results.error:
            return results.error
        
        # Handle empty results
        if results.is_empty():
            filter_info = ""
            if course_name:
                filter_info += f" in course '{course_name}'"
            if lesson_number:
                filter_info += f" in lesson {lesson_number}"
            return f"No relevant content found{filter_info}."
        
        # Format and return results
        return self._format_results(results)
    
    def _format_results(self, results: SearchResults) -> str:
        """Format search results with course and lesson context"""
        formatted = []
        sources = []  # Track sources for the UI
        
        for doc, meta in zip(results.documents, results.metadata):
            course_title = meta.get('course_title', 'unknown')
            lesson_num = meta.get('lesson_number')
            source_file = meta.get('source_file', '')
            start_line = meta.get('start_line', 0)
            end_line = meta.get('end_line', 0)

            # Build context header
            header = f"[{course_title}"
            if lesson_num is not None:
                header += f" - Lesson {lesson_num}"
            header += "]"

            # Build source reference with file + line range
            source = course_title
            if lesson_num is not None:
                source += f" - Lesson {lesson_num}"
            if source_file and start_line and end_line:
                source += f" ({source_file}, lines {start_line}–{end_line})"
            elif source_file:
                source += f" ({source_file})"
            sources.append(source)
            
            formatted.append(f"{header}\n{doc}")
        
        # Deduplicate sources while preserving order
        seen = set()
        unique_sources = []
        for s in sources:
            if s not in seen:
                seen.add(s)
                unique_sources.append(s)

        # Store sources for retrieval
        self.last_sources = unique_sources
        
        return "\n\n".join(formatted)

class SearchNewsTool(Tool):
    """Tool for searching live news articles indexed from RSS feeds."""

    def __init__(self, vector_store: VectorStore):
        self.store = vector_store
        self.last_sources = []

    def get_tool_definition(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "search_news",
                "description": (
                    "Search today's live news headlines and articles. "
                    "Use this for questions about current events, breaking news, or recent developments. "
                    "Do NOT use this for course content questions."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "What to search for in the news"
                        },
                        "section": {
                            "type": "string",
                            "description": "Optional news section, e.g. 'technology', 'world', 'business'"
                        },
                        "max_hours_old": {
                            "type": "integer",
                            "description": "Only return articles fetched within this many hours (default: 72)"
                        }
                    },
                    "required": ["query"]
                }
            }
        }

    def execute(self, query: str, section: Optional[str] = None, max_hours_old: Optional[int] = None) -> str:
        results = self.store.search_news(
            query=query,
            section=section,
            max_hours_old=max_hours_old,
        )

        if results.error:
            return results.error

        if results.is_empty():
            return "No relevant news found. The news index may be empty — try refreshing via /api/news/refresh."

        return self._format_results(results)

    def _format_results(self, results) -> str:
        formatted = []
        sources = []

        for doc, meta in zip(results.documents, results.metadata):
            title = meta.get("title", "")
            url = meta.get("url", "")
            source = meta.get("source", "News")
            published_at = meta.get("published_at", "")

            header = f"[{source}] {title}"
            if published_at:
                try:
                    from datetime import datetime, timezone
                    dt = datetime.fromisoformat(published_at)
                    header += f" ({dt.strftime('%b %d, %Y')})"
                except Exception:
                    pass

            formatted.append(f"{header}\n{doc}")

            source_label = f"{title} — {url}" if url else title
            if source_label not in sources:
                sources.append(source_label)

        self.last_sources = sources
        return "\n\n".join(formatted)


class SearchWebTool(Tool):
    """Tool for live web search via headless Chromium (Google→Bing→DuckDuckGo fallback)."""

    def __init__(self, web_searcher):
        self.web_searcher = web_searcher
        self.last_sources = []

    def get_tool_definition(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "search_web",
                "description": (
                    "Search the internet for any general knowledge, technical topics, or information "
                    "not covered by course materials or news. Use for 'how does X work', 'latest X', "
                    "definitions, tutorials, comparisons, etc. Do NOT use for course content or today's news."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "What to search for on the web"
                        }
                    },
                    "required": ["query"]
                }
            }
        }

    def execute(self, query: str, **kwargs) -> str:
        results = self.web_searcher.search(query)

        if not results:
            return "Web search returned no results. All search engines may be temporarily unavailable."

        self.last_sources = [r["url"] for r in results if r.get("url")]
        return self._format_results(results)

    def _format_results(self, results: list) -> str:
        parts = []
        for r in results:
            engine = r.get("engine", "Web")
            title = r.get("title", "")
            snippet = r.get("snippet", "")
            url = r.get("url", "")
            block = f"[{engine}] {title}"
            if snippet:
                block += f"\n{snippet}"
            if url:
                block += f"\nURL: {url}"
            parts.append(block)
        return "\n\n".join(parts)


class ToolManager:
    """Manages available tools for the AI"""
    
    def __init__(self):
        self.tools = {}
    
    def register_tool(self, tool: Tool):
        """Register any tool that implements the Tool interface"""
        tool_def = tool.get_tool_definition()
        # Support both OpenAI format {"type":"function","function":{"name":...}}
        # and flat format {"name":...}
        if "function" in tool_def:
            tool_name = tool_def["function"].get("name")
        else:
            tool_name = tool_def.get("name")
        if not tool_name:
            raise ValueError("Tool must have a 'name' in its definition")
        self.tools[tool_name] = tool

    
    def get_tool_definitions(self) -> list:
        """Get all tool definitions for Anthropic tool calling"""
        return [tool.get_tool_definition() for tool in self.tools.values()]
    
    def execute_tool(self, tool_name: str, **kwargs) -> str:
        """Execute a tool by name with given parameters"""
        if tool_name not in self.tools:
            return f"Tool '{tool_name}' not found"
        
        return self.tools[tool_name].execute(**kwargs)
    
    def get_last_sources(self) -> list:
        """Get sources from the last search operation"""
        # Check all tools for last_sources attribute
        for tool in self.tools.values():
            if hasattr(tool, 'last_sources') and tool.last_sources:
                return tool.last_sources
        return []

    def reset_sources(self):
        """Reset sources from all tools that track sources"""
        for tool in self.tools.values():
            if hasattr(tool, 'last_sources'):
                tool.last_sources = []