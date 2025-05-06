from __future__ import annotations

import os
import re
from typing import Dict, List

import openai
from .tools import Tool
from .prompts import PromptBuilder


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
    ):
        self.model = llm_model
        self.tools: Dict[str, Tool] = {tool.name: tool for tool in tools}
        self.temperature = temperature
        self.max_turns = max_turns

        # Configure OpenAI client
        self.client = openai.OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key="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",
            # Ensure OPENAI_API_KEY env var is set
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def run(self, initial_task: str) -> str:
        print(f"Task: {initial_task}")

        # Pass self.tools (Dict[str, Tool]) directly to PromptBuilder
        messages: List[Dict[str, str]] = [
            {"role": "user", "content": PromptBuilder.build_base_prompt(initial_task, self.tools)}
        ]
        
        print("--- Initial Prompt to LLM ---")
        print(messages[0]['content'])
        print("-----------------------------")

        i = 0
        action_re = self.__class__.action_re
        answer_re = re.compile(r"^Answer: (.*)", re.MULTILINE)

        while i < self.max_turns:
            i += 1
            result = self._chat(messages)
            print(result) # Print LLM's raw output
            messages.append({"role": "assistant", "content": result})

            actions = [action_re.match(a) for a in result.split('\n') if action_re.match(a)]
            if actions:
                groups = actions[0].groups()
                action = groups[0]
                action_input = groups[1] if len(groups) > 1 else None
                
                # ---- DEBUG PRINT ----
                print(f"DEBUG: Extracted Action: '{action}', Input: '{action_input}'")
                # ---- END DEBUG PRINT ----

                if action not in self.tools:
                    observation = f"Error: Unknown action {action} with input {action_input}"
                else:
                    try:
                        if action_input:
                            observation = self.tools[action](action_input)
                        else:
                            observation = self.tools[action]()
                    except Exception as e:
                        observation = f"Error executing action {action}: {str(e)}"
                
                observation_msg_content = f"Observation: {observation}"
                messages.append({"role": "user", "content": observation_msg_content})
            else:
                match = answer_re.search(result)
                if match:
                    final_answer = match.group(1).strip()
                    return final_answer
                else:
                    return "Agent finished without a formal answer or action."

        return "Agent reached max_turns without a final answer."

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _chat(self, messages: List[Dict[str, str]]) -> str:
        """One OpenAI ChatCompletion call (separated for mocking in tests)."""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=2048, # Increased
            )
            print(f"--- Raw API Response ---\n{response}\n------------------------")

            if response.choices:
                return response.choices[0].message.content.strip()
            else:
                print("Error: API response.choices is None or empty.")
                # Potentially raise an error or return a specific error message string
                # For now, to avoid crashing the agent, let's return an error observation string
                return "Error: No response choices received from LLM."

        except Exception as e:
            print(f"Error during API call: {e}")
            # Return an error observation string so the agent can attempt to recover or report
            return f"Error: API call failed with exception: {str(e)}"
