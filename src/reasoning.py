#!/usr/bin/env python3
"""
BlackRoad DeepSeek — Reasoning Model Integration
Wraps DeepSeek-R1 for multi-step chain-of-thought reasoning.
"""
import os, httpx
from typing import Generator

GATEWAY_URL = os.environ.get("BLACKROAD_GATEWAY_URL", "http://127.0.0.1:8787")

def reason(
    problem: str,
    model: str = "deepseek-r1:7b",
    temperature: float = 0.1,
    show_thinking: bool = False,
) -> dict:
    """Send a problem to DeepSeek-R1 and get chain-of-thought reasoning."""
    system = """You are an expert reasoning system. When solving problems:
1. Think step-by-step inside <think>...</think> tags
2. Show your working clearly  
3. Give a concise final answer after </think>"""

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": problem},
        ],
        "temperature": temperature,
        "stream": False,
    }

    resp = httpx.post(f"{GATEWAY_URL}/chat", json=payload, timeout=120)
    resp.raise_for_status()
    data = resp.json()
    content = data.get("content", data.get("choices", [{}])[0].get("message", {}).get("content", ""))

    # Parse think block
    think = ""
    answer = content
    if "<think>" in content and "</think>" in content:
        start = content.index("<think>") + 7
        end = content.index("</think>")
        think = content[start:end].strip()
        answer = content[end + 8:].strip()

    result = {"thinking": think, "answer": answer, "model": model}
    if show_thinking:
        print(f"
[Thinking]
{think}
")
    return result


def reason_code(task: str, language: str = "python") -> dict:
    """Use DeepSeek for code generation with reasoning."""
    problem = f"Write {language} code to: {task}

Include docstrings and type hints."
    return reason(problem, model="deepseek-r1:7b", temperature=0.0, show_thinking=True)


def batch_reason(problems: list[str]) -> list[dict]:
    """Solve multiple problems in sequence."""
    return [reason(p) for p in problems]


if __name__ == "__main__":
    import json
    result = reason(
        "If an agent has 847 wins and 23 losses, what is their win rate? Round to 2 decimal places.",
        show_thinking=True,
    )
    print(f"Answer: {result['answer']}")
