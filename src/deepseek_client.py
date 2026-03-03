"""
BlackRoad DeepSeek Client
Wrapper for DeepSeek models via Ollama
"""
import os
import httpx
import json
from typing import AsyncIterator, Optional

from auth import get_auth_headers


OLLAMA_URL = os.environ.get("BLACKROAD_OLLAMA_URL", "http://localhost:11434")

MODEL_MAP = {
    "reasoning": "deepseek-r1:7b",
    "code": "deepseek-coder-v2:16b",
    "math": "deepseek-math:7b",
    "fast": "deepseek-r1:7b",
    "heavy": "deepseek-r1:32b",
}


async def chat(
    prompt: str,
    task_type: str = "reasoning",
    model: Optional[str] = None,
    stream: bool = False,
    temperature: float = 0.6,
    max_tokens: int = 4096,
) -> str:
    """Send a chat request to DeepSeek via Ollama."""
    model_name = model or MODEL_MAP.get(task_type, "deepseek-r1:7b")

    async with httpx.AsyncClient(timeout=120.0) as client:
        resp = await client.post(
            f"{OLLAMA_URL}/api/generate",
            headers=get_auth_headers(),
            json={
                "model": model_name,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens,
                }
            }
        )
        resp.raise_for_status()
        return resp.json()["response"]


async def stream_chat(
    prompt: str,
    task_type: str = "reasoning",
    model: Optional[str] = None,
    temperature: float = 0.6,
) -> AsyncIterator[str]:
    """Stream tokens from DeepSeek."""
    model_name = model or MODEL_MAP.get(task_type, "deepseek-r1:7b")

    async with httpx.AsyncClient(timeout=300.0) as client:
        async with client.stream(
            "POST",
            f"{OLLAMA_URL}/api/generate",
            headers=get_auth_headers(),
            json={"model": model_name, "prompt": prompt, "stream": True,
                  "options": {"temperature": temperature}},
        ) as resp:
            async for line in resp.aiter_lines():
                if line:
                    data = json.loads(line)
                    if token := data.get("response"):
                        yield token
                    if data.get("done"):
                        break


async def solve_code(problem: str) -> str:
    """Specialized code problem solver."""
    system = "You are an expert programmer. Write clean, correct code with brief explanations."
    return await chat(f"{system}\n\n{problem}", task_type="code")


async def solve_math(problem: str) -> str:
    """Specialized math solver with step-by-step reasoning."""
    system = "You are a mathematics expert. Show your work step by step. Use LaTeX for formulas."
    return await chat(f"{system}\n\n{problem}", task_type="math")


async def reason(problem: str) -> str:
    """Deep reasoning with chain-of-thought."""
    system = "Think step by step. Show your reasoning process before giving the answer."
    return await chat(f"{system}\n\n{problem}", task_type="reasoning", temperature=0.3)
