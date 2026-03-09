"""
BlackRoad AI — Mention Router
Routes @copilot, @lucidia, @blackboxprogramming, and @ollama mentions
directly to Ollama running on local hardware.  No external providers.
"""
import re
from typing import Optional

from deepseek_client import chat, stream_chat, MODEL_MAP

# Every recognised @mention is mapped to Ollama.
# Adding a new alias is as simple as adding it to this set.
OLLAMA_MENTIONS: set[str] = {
    "ollama",
    "copilot",
    "lucidia",
    "blackboxprogramming",
}

_MENTION_RE = re.compile(
    r"@(" + "|".join(re.escape(m) for m in OLLAMA_MENTIONS) + r")\b",
    re.IGNORECASE,
)


def extract_mention(text: str) -> Optional[str]:
    """Return the first recognised @mention found in *text*, or None."""
    match = _MENTION_RE.search(text)
    return match.group(1).lower() if match else None


def strip_mentions(text: str) -> str:
    """Remove all recognised @mentions from *text* and strip extra whitespace."""
    return _MENTION_RE.sub("", text).strip()


async def route(
    prompt: str,
    task_type: str = "reasoning",
    model: Optional[str] = None,
    stream: bool = False,
    temperature: float = 0.6,
    max_tokens: int = 4096,
):
    """
    Parse *prompt* for an @mention and forward the request to Ollama.

    If a known @mention is present the mention is stripped from the prompt
    before sending so the model receives only the actual user text.

    Returns either a complete response string (stream=False) or an async
    generator of token strings (stream=True).
    """
    mention = extract_mention(prompt)
    clean_prompt = strip_mentions(prompt) if mention else prompt

    if stream:
        return stream_chat(
            clean_prompt,
            task_type=task_type,
            model=model,
            temperature=temperature,
        )

    return await chat(
        clean_prompt,
        task_type=task_type,
        model=model,
        stream=False,
        temperature=temperature,
        max_tokens=max_tokens,
    )
