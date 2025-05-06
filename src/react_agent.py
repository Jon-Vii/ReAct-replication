
from __future__ import annotations

import os
import re
from typing import Dict, List

import openai
from .tools import Tool


__all__ = ["ReactAgent"]


class ReactAgent:
    """Reason + Act (ReAct) agent that orchestrates an LLM and Python tools."""

    action_re = re.compile(r"^Action: (\w+)\[(.*)\]")

    def __init__(
        self,
        llm_model: str,
        tools: List[Tool],
        temperature: float = 0.0,
        max_turns: int = 10,
    ) -> None:
        self.model = llm_model
        self.tools: Dict[str, Tool] = {tool.name: tool for tool in tools}
        self.temperature = temperature
        self.max_turns = max_turns

        # Configure OpenAI client
        self.client = openai.OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key="KEY",
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def run(self, prompt: str) -> str:
        i = 0
        if i == 0:
            print(prompt)
        next_prompt = prompt
        action_re = self.__class__.action_re
        while i < self.max_turns:
            i += 1
            result = self._chat(next_prompt)
            print(result)
            actions = [action_re.match(a) for a in result.split('\n') if action_re.match(a)]
            if actions:
                groups = actions[0].groups()
                action = groups[0]
                action_input = groups[1] if len(groups) > 1 else None
                if action not in self.tools:
                    raise Exception(f"Unknown action: {action}: {action_input}")
                if action_input:
                    observation = self.tools[action](action_input)
                else:
                    observation = self.tools[action]()
                next_prompt = f"Observation: {observation}"
            else:
                return

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _chat(self, prompt_string: str) -> str:
        """One OpenAI ChatCompletion call (separated for mocking in tests)."""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt_string}],
            temperature=self.temperature,
            max_tokens=512,
        )
        return response.choices[0].message.content.strip()
