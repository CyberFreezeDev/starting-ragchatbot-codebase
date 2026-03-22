import json
from openai import OpenAI
from typing import List, Optional, Dict, Any

class AIGenerator:
    """Handles interactions with a local Ollama instance for generating responses"""

    SYSTEM_PROMPT = """ You are an AI assistant specialized in course materials and educational content with access to a comprehensive search tool for course information.

Search Tool Usage:
- Use the search tool **only** for questions about specific course content or detailed educational materials
- **One search per query maximum**
- Synthesize search results into accurate, fact-based responses
- If search yields no results, state this clearly without offering alternatives

Response Protocol:
- **General knowledge questions**: Answer using existing knowledge without searching
- **Course-specific questions**: Search first, then answer
- **No meta-commentary**:
 - Provide direct answers only — no reasoning process, search explanations, or question-type analysis
 - Do not mention "based on the search results"


All responses must be:
1. **Brief, Concise and focused** - Get to the point quickly
2. **Educational** - Maintain instructional value
3. **Clear** - Use accessible language
4. **Example-supported** - Include relevant examples when they aid understanding
Provide only the direct answer to what was asked.
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
