from __future__ import annotations

PROMPTS = {
    "classic": "Read the text in the image. Return only the text as it is visible in the image."
}


def get_all_prompts() -> list[str]:
    return list(PROMPTS.values())


__all__ = ["PROMPTS", "get_all_prompts"]


