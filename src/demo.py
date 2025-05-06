"""Tiny demo showing how to wire up the ReactAgent package."""
import os

from .react_agent import ReactAgent
from .tools import search_web, calculator, Tool


def main() -> None:
    # Ensure the API key is set (export OPENAI_API_KEY=sk‑...)
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("Please set the OPENAI_API_KEY environment variable.")

    # Pass tools as a list of Tool objects
    tools_list = [
        Tool(name="Search", function=search_web),
        Tool(name="Calculator", function=calculator)
    ]
    agent = ReactAgent("gpt-4o", tools_list, max_turns=8)

    answer = agent.run("What is the square root of 2556? Use the calculator tool.")
    print("Final answer:", answer)


if __name__ == "__main__":
    main()
