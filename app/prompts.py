"""Prompt templates shared by training, inference and evaluation.

Keeping one source of truth matters: the LoRA adapter is trained on exactly
these message formats, so the API must build prompts the same way.
"""

from __future__ import annotations

import re

ABSTAIN = "I don't have that information in the company documents."

SYSTEM_RAG = (
    "You are a helpful workplace assistant. Answer the employee's question in one or two short, "
    "friendly sentences using ONLY the facts in the context. Do not add facts that are not in the context. "
    f'If the context does not contain the answer, reply exactly: "{ABSTAIN}"'
)

SYSTEM_NO_RAG = (
    "You are a helpful workplace assistant. Reply in one or two short, friendly, professional sentences."
)


def format_context(passages: list[tuple[str, str]]) -> str:
    """passages: list of (title, text)."""
    return "\n".join(f"[{i}] {title}: {text}" for i, (title, text) in enumerate(passages, 1))


def build_messages(question: str, passages: list[tuple[str, str]] | None) -> list[dict]:
    if passages is None:  # RAG disabled
        return [
            {"role": "system", "content": SYSTEM_NO_RAG},
            {"role": "user", "content": question},
        ]
    context = format_context(passages) if passages else "(no relevant documents found)"
    return [
        {"role": "system", "content": SYSTEM_RAG},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
    ]


_ABSTAIN_PATTERNS = re.compile(
    r"(don'?t|do not) have (that|this|the|any|enough)? ?information"
    r"|not (mentioned|specified|stated|provided|available|included|in the (company )?documents|in the context)"
    r"|(i'?m not sure|i do not know|i don'?t know|cannot find|can'?t find|no information|unable to find)",
    re.IGNORECASE,
)


def is_abstention(answer: str) -> bool:
    return bool(_ABSTAIN_PATTERNS.search(answer))
