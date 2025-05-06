"""Prompt construction utilities for the ReAct agent."""
from typing import Callable, Dict

# ---------------------------------------------------------------------
# Prompt helpers
# ---------------------------------------------------------------------
THOUGHT_TAG = "[Thinking]" 
class PromptBuilder:
    def __init__(self):
        pass
    def _tool_descriptions(tools: Dict[str, Callable[[str], str]]) -> str:
        """Return a newline‑separated list of tool docstrings."""
        lines = []
        for name, fn in tools.items():
            doc = (fn.__doc__ or "").strip().replace("\n", " ")
            lines.append(f"{name}: {doc}")
        return "\n".join(lines)

    def build_base_prompt(task: str, tools: Dict[str, Callable[[str], str]]) -> str:
            """Return the initial ReAct prompt for *task* given *tools* using the [Thinking] tag."""
            tool_block = PromptBuilder._tool_descriptions(tools)
            tools_list = ", ".join(tools.keys())

            return (
                "You are a ReAct agent that can use external tools to solve tasks.\n\n"
                "The tools you can invoke are:\n"
                f"{tool_block}\n\n"
                "When solving a task, think step‑by‑step. Use the following annotated format strictly:\n\n"
                "Task: <the task>\n"
                f"{THOUGHT_TAG}: <your chain‑of‑thought reasoning>\n"
                f"Action: <one of [{tools_list}]>[<argument>]\n"
                "Observation: <result returned by the action>\n"
                f"... (repeat Thought/Action/Observation as needed)\n"
                f"Thought: I now know the answer.\n"
                "Answer: <final answer>\n\n"
                "Begin!\n\n"
                f"Task: {task}\n"
            )
                            