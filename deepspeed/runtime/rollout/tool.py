# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""Tools the model can call during multi-turn rollout.

Users subclass ``Tool``: its spec tells the model what the tool does (it is
passed to ``tokenizer.apply_chat_template(tools=...)``) and ``execute`` runs
a call. ``ToolSet`` groups the tools of a rollout, and
``parse_hermes_tool_calls`` reads the calls a model emits.
"""

import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence


class Tool(ABC):
    """A tool the model can call.

    Subclasses set ``name``, ``description`` and ``parameters`` (a JSON
    schema of the arguments) and implement ``execute``.
    """

    name: str
    description: str
    parameters: Dict[str, Any]

    def spec(self) -> Dict[str, Any]:
        """The tool in OpenAI function format, as chat templates expect it."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    @abstractmethod
    async def execute(self, arguments: Dict[str, Any]) -> str:
        """Run one call. The returned string becomes the tool message's content."""


class ToolSet:
    """The tools available in a rollout, looked up by name.

    Prompts must be rendered with ``specs`` so the model is shown exactly the
    tools the rollout can run.
    """

    def __init__(self, tools: Sequence[Tool]):
        self.tools = {}
        for tool in tools:
            if tool.name in self.tools:
                raise ValueError(f"duplicate tool name {tool.name!r}")
            self.tools[tool.name] = tool

    @property
    def specs(self) -> List[Dict[str, Any]]:
        return [tool.spec() for tool in self.tools.values()]

    def get(self, name: str) -> Optional[Tool]:
        return self.tools.get(name)

    async def run(self, call: "ToolCall") -> str:
        """Execute one call and return the text the model reads next.

        Parse errors, unknown tools and exceptions are reported as text, so
        one bad call never fails the rollout.
        """
        if call.error is not None:
            return f"Error: {call.error}"
        tool = self.get(call.name)
        if tool is None:
            return f"Error: unknown tool {call.name!r}; available tools: {', '.join(self.tools) or 'none'}"
        try:
            return str(await tool.execute(call.arguments))
        except Exception as error:  # noqa: BLE001 - any tool failure is reported to the model
            return f"Error: {type(error).__name__}: {error}"

    def __len__(self) -> int:
        return len(self.tools)


@dataclass
class ToolCall:
    """One call parsed from model output; ``error`` is set when it could not be parsed."""

    name: Optional[str]
    arguments: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None


_TOOL_CALL_BLOCK = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)


def parse_hermes_tool_calls(text: str) -> List[ToolCall]:
    """Parse ``<tool_call>{"name": ..., "arguments": {...}}</tool_call>`` blocks.

    This is the format the Hermes, Qwen2.5 and Qwen3 chat templates teach. A
    block that cannot be parsed still yields a ``ToolCall`` with ``error``
    set, so the model can be told what went wrong instead of the call
    silently disappearing.
    """
    calls = []
    for match in _TOOL_CALL_BLOCK.finditer(text):
        try:
            payload = json.loads(match.group(1).strip())
        except json.JSONDecodeError as error:
            calls.append(ToolCall(name=None, error=f"the tool call is not valid JSON ({error.msg})"))
            continue
        if not isinstance(payload, dict) or not isinstance(payload.get("name"), str):
            calls.append(ToolCall(name=None, error='a tool call must be a JSON object with a string "name"'))
            continue

        name = payload["name"]
        arguments = payload.get("arguments", {})
        # Some models emit the arguments as a JSON-encoded string.
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments) if arguments.strip() else {}
            except json.JSONDecodeError as error:
                calls.append(ToolCall(name=name, error=f"the tool arguments are not valid JSON ({error.msg})"))
                continue
        if not isinstance(arguments, dict):
            calls.append(ToolCall(name=name, error='"arguments" must be a JSON object'))
            continue
        calls.append(ToolCall(name=name, arguments=arguments))

    # An opening tag without a closing one means the call was cut off.
    if text.count("<tool_call>") > text.count("</tool_call>"):
        calls.append(ToolCall(name=None, error="the <tool_call> block is not closed with </tool_call>"))
    return calls
