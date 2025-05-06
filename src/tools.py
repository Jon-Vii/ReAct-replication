"""Example built‑in tools for the ReAct agent.

Feel free to delete these and replace them with real domain‑specific tools.
"""
from __future__ import annotations

import math
from typing import Dict, Callable, Any

__all__ = ["search_web", "calculator", "Tool"]


class Tool:
    """A simple class to wrap a tool function and give it a name."""
    def __init__(self, name: str, function: Callable[..., str]):
        self.name = name
        self.function = function

    def __call__(self, *args: Any, **kwargs: Any) -> str:
        return self.function(*args, **kwargs)


def search_web(query: str) -> str:
    """Pretend‑search the web and return a short snippet (stub)."""

    # TODO: integrate a real search API (SerpAPI, Tavily, Bing, etc.)
    return f"[Search results for '{query}' ...]"


def calculator(expr: str) -> str:
    """Evaluate a basic math expression using `eval` in a sandboxed environment."""
    print("Wowweee")
    try:
        allowed_names = {k: v for k, v in math.__dict__.items() if not k.startswith("__")}
        return str(eval(expr, {"__builtins__": {}}, allowed_names))
    except Exception as exc:
        return f"calc error: {exc}"
