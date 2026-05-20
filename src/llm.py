import json
import urllib.request
from typing import Optional

OLLAMA_URL = "http://localhost:11434/api/chat"


def generate(
    model: str,
    system: str,
    messages: list[dict],
    seed: Optional[int] = None,
) -> str:
    """Send a chat request to a local Ollama model and return the reply text.

    Args:
        model:    Ollama model name, e.g. "llama3.1:8b".
        system:   System prompt string.
        messages: List of {"role": "user"|"assistant", "content": str} dicts
                  representing the conversation so far (excluding the system prompt).
        seed:     Optional integer seed for reproducibility.

    Returns:
        The model's reply as a plain string.

    Raises:
        RuntimeError: If Ollama returns a non-200 status or an error field.
    """
    payload: dict = {
        "model": model,
        "messages": [{"role": "system", "content": system}] + messages,
        "stream": False,
    }
    if seed is not None:
        payload["options"] = {"seed": seed}

    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        OLLAMA_URL,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req) as resp:
            body = json.loads(resp.read())
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"Ollama HTTP {e.code}: {e.read().decode()}") from e
    except urllib.error.URLError as e:
        raise RuntimeError(
            f"Cannot reach Ollama at {OLLAMA_URL}. Is `ollama serve` running?"
        ) from e

    if "error" in body:
        raise RuntimeError(f"Ollama error: {body['error']}")

    return body["message"]["content"].strip()
