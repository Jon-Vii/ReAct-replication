"""Prompt construction utilities for the ReAct agent."""
from typing import Dict
from .tools import Tool

# ---------------------------------------------------------------------
# Prompt helpers
# ---------------------------------------------------------------------
THOUGHT_TAG = "[Thinking]" 
class PromptBuilder:
    def __init__(self):
        pass
    def _tool_descriptions(tools: Dict[str, Tool]) -> str:
        """Return a newline‑separated list of tool descriptions."""
        lines = []
        for name, tool_obj in tools.items(): 
            description = (tool_obj.description or "No description available.").strip().replace("\n", " ")
            lines.append(f"{name}: {description}")
        return "\n".join(lines)

    def build_base_prompt(task: str, tools: Dict[str, Tool]) -> str:
            """Return the initial ReAct prompt for *task* given *tools* using the [Thinking] tag."""
            tool_block = PromptBuilder._tool_descriptions(tools)
            tools_name_list = list(tools.keys())
            tools_list_str = ", ".join(tools_name_list)

            return (
                "You are a ReAct agent that can use external tools to solve tasks.\n\n"
                "The tools you can invoke are:\n"
                f"{tool_block}\n\n"
                "When solving a task, think step‑by‑step. Use the following annotated format strictly:\n\n"
                "Task: <the task>\n"
                f"{THOUGHT_TAG}: <your chain‑of‑thought reasoning>\n"
                f"Action: <one of [{tools_list_str}]>[<argument>]\n" 
                "Observation: <result returned by the action>\n"
                f"... (repeat Thought/Action/Observation as needed)\n"
                f"Thought: I now know the answer.\n"
                "Answer: <final answer>\n\n"
                "Begin!\n\n"
                f"Task: {task}\n"
            )