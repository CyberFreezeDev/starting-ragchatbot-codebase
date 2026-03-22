import json
from openai import OpenAI
from typing import List, Optional, Dict, Any

class AIGenerator:
    """Handles interactions with a local Ollama instance for generating responses"""

    SYSTEM_PROMPT = """You are an AI assistant with three search tools:

1. **search_course_content** — searches enrolled course materials and lessons
2. **search_news** — searches today's live news index (BBC RSS, updated hourly)
3. **search_web** — performs a live internet search via a real browser (Google → Bing → DuckDuckGo)

Tool Routing Rules — pick EXACTLY ONE per query:
- "What does lesson X cover?" / "Explain concept from the course" → **search_course_content**
- "What's in the news?" / "What happened today?" / "Latest headlines" → **search_news**
- "How does X work?" / "What is X?" / "Latest version of X?" / anything factual not in courses → **search_web**
- Simple greetings or truly general knowledge → answer directly without searching
- **One search per query maximum — never call multiple tools**

Synthesize search results into accurate, fact-based responses.
If a search yields no results, say so clearly.

Response Protocol:
- Direct answers only — no meta-commentary, no "I searched for...", no "based on results..."
- Brief and focused — get to the point
- Include examples when they aid understanding
"""

    def __init__(self, base_url: str, model: str):
        self.client = OpenAI(api_key="ollama", base_url=base_url)
        self.model = model

        self.base_params = {
            "model": self.model,
            "temperature": 0,
            "max_tokens": 800
        }

    def generate_response(self, query: str,
                         conversation_history: Optional[str] = None,
                         tools: Optional[List] = None,
                         tool_manager=None) -> str:
        """
        Generate AI response with optional tool usage and conversation context.
        """
        system_content = (
            f"{self.SYSTEM_PROMPT}\n\nPrevious conversation:\n{conversation_history}"
            if conversation_history
            else self.SYSTEM_PROMPT
        )

        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": query}
        ]

        api_params = {
            **self.base_params,
            "messages": messages
        }

        if tools:
            api_params["tools"] = tools
            api_params["tool_choice"] = "auto"

        response = self.client.chat.completions.create(**api_params)

        # Ollama sometimes returns finish_reason="stop" even when tool_calls are present
        message = response.choices[0].message
        has_tool_calls = bool(message.tool_calls) or response.choices[0].finish_reason == "tool_calls"
        if has_tool_calls and tool_manager:
            return self._handle_tool_execution(response, messages, system_content, tool_manager)

        return message.content

    def _handle_tool_execution(self, initial_response, messages: List, system_content: str, tool_manager) -> str:
        """
        Handle execution of tool calls and get follow-up response.
        """
        assistant_message = initial_response.choices[0].message
        tool_results = []

        # Extract the original user query for fallback when model sends empty args
        original_query = next(
            (m["content"] for m in messages if m["role"] == "user"), ""
        )

        for tc in assistant_message.tool_calls:
            args = json.loads(tc.function.arguments) if tc.function.arguments else {}
            # Fallback: if model called tool with no query arg, use the user's message
            if not args.get("query"):
                args["query"] = original_query
            result = tool_manager.execute_tool(tc.function.name, **args)
            tool_results.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result
            })

        followup_messages = [
            *messages,
            assistant_message,
            *tool_results
        ]

        final_params = {
            **self.base_params,
            "messages": followup_messages
        }

        final_response = self.client.chat.completions.create(**final_params)
        return final_response.choices[0].message.content
