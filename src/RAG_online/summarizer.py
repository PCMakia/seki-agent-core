"""
Bullet point summarization for text.

Supports two modes:
- extractive (default): Extractive summarization via Qwen3-Embed. Selects most important sentences.
  Requires: pip install qwen3-embed
- llm: Abstractive summarization via Ollama (qwen3:4b). Produces condensed bullet points.
"""
from __future__ import annotations

import re
from typing import Literal, Optional

from src.LLM_handler.llm_client import LLMClient

# Rough token estimate: ~4 chars per token for English
CHARS_PER_TOKEN = 4
MAX_INPUT_TOKENS = 4000
MAX_CHARS_PER_CHUNK = MAX_INPUT_TOKENS * CHARS_PER_TOKEN

BULLET_PROMPT = """Summarize the following text in {max_bullets} bullet points.
Include only the most important, useful, or interesting information.
Format each point as a single line starting with "- ".
Do not add an introduction or conclusion, only the bullet points.

Text:
{text}"""


def _chunk_text(text: str, max_chars: int = MAX_CHARS_PER_CHUNK) -> list[str]:
    """Split text into chunks that fit within context limits.

    Prefers paragraph boundaries; falls back to sentence boundaries.
    """
    text = (text or "").strip()
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]

    chunks: list[str] = []
    remaining = text

    while remaining:
        if len(remaining) <= max_chars:
            chunks.append(remaining.strip())
            break

        # Take a chunk of max_chars
        candidate = remaining[:max_chars]

        # Try to break at paragraph boundary (double newline)
        last_para = candidate.rfind("\n\n")
        if last_para > max_chars // 2:
            split_at = last_para + 2
        else:
            # Fall back to sentence boundary (., !, ?)
            last_sent = max(
                candidate.rfind(". "),
                candidate.rfind("! "),
                candidate.rfind("? "),
            )
            if last_sent > max_chars // 2:
                split_at = last_sent + 2
            else:
                split_at = max_chars

        chunk = remaining[:split_at].strip()
        if chunk:
            chunks.append(chunk)
        remaining = remaining[split_at:].strip()

    return chunks


def _summarize_chunk_llm(
    chunk: str,
    max_bullets: int,
    llm: LLMClient,
) -> str:
    """Summarize a single chunk via LLM."""
    prompt = BULLET_PROMPT.format(max_bullets=max_bullets, text=chunk)
    result = llm.generate(
        prompt,
        system_prompt="You are a helpful assistant that summarizes text concisely.",
        history=[],
    )
    return (result.get("completion") or "").strip()


def bullet_summary_llm(
    text: str,
    *,
    max_bullets: int = 15,
    llm: Optional[LLMClient] = None,
) -> str:
    """Abstractive bullet point summarization via LLM (Ollama + qwen3:4b).

    Uses map-reduce for long text: summarize each chunk, then summarize the
    combined chunk summaries.
    """
    llm = llm or LLMClient()
    chunks = _chunk_text(text)

    if not chunks:
        return ""

    if len(chunks) == 1:
        return _summarize_chunk_llm(chunks[0], max_bullets, llm)

    # Map: summarize each chunk
    chunk_summaries: list[str] = []
    for chunk in chunks:
        summary = _summarize_chunk_llm(chunk, max_bullets=max_bullets + 5, llm=llm)
        if summary:
            chunk_summaries.append(summary)

    if not chunk_summaries:
        return ""

    # Reduce: summarize the combined summaries
    combined = "\n\n".join(chunk_summaries)
    return _summarize_chunk_llm(combined, max_bullets=max_bullets, llm=llm)


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences (simple regex-based)."""
    # Split on sentence-ending punctuation followed by space or end
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [s.strip() for s in parts if s.strip()]


def _shorten_sentence(
    sentence: str,
    *,
    max_words: int | None = None,
    remove_parentheticals: bool = True,
) -> str:
    """Shorten a sentence while preserving core meaning.

    Techniques from extractive summarization practice:
    - Remove parentheticals [...] and (...) which often contain citations or asides
    - Truncate to first clause (before comma/semicolon) for very long sentences
    - Cap at max_words when specified
    """
    s = sentence.strip()
    if not s:
        return s

    if remove_parentheticals:
        s = re.sub(r"\[[^\]]*\]", "", s)
        s = re.sub(r"\([^)]*\)", "", s)
        s = re.sub(r"\s+", " ", s).strip()

    if max_words is not None and max_words > 0:
        words = s.split()
        if len(words) > max_words:
            # Prefer truncating at clause boundary (comma, semicolon) if within range
            for sep in (",", ";", " – ", " - "):
                idx = s.find(sep)
                if 0 < idx < len(" ".join(words[:max_words])) + 20:
                    s = s[:idx].strip()
                    break
            else:
                s = " ".join(words[:max_words])
    return s


def bullet_summary_extractive(
    text: str,
    *,
    max_bullets: int = 15,
    ratio: float | None = 0.3,
    max_words_per_bullet: int | None = 40,
    remove_parentheticals: bool = True,
    model_name: str = "n24q02m/Qwen3-Embedding-0.6B-ONNX-Q4F16",
) -> str:
    """Extractive bullet point summarization via Qwen3-Embed.

    Embeds sentences, selects those closest to document centroid,
    and returns them as bullet points. Uses ratio and truncation so
    the summary is shorter than the source (see bert-extractive-summarizer,
    Stack Overflow on sentence shortening).
    """
    sentences = _split_sentences(text)
    if not sentences:
        return ""

    # Ratio: cap bullets to fraction of source sentences (ensures shorter output)
    k = max_bullets
    if ratio is not None and 0 < ratio < 1:
        k = min(k, max(1, int(len(sentences) * ratio)))
    k = min(k, len(sentences))

    if len(sentences) <= k:
        selected = sentences
    else:
        from qwen3_embed import TextEmbedding

        model = TextEmbedding(model_name=model_name)
        embeddings = list(model.embed(sentences))

        import numpy as np

        centroid = np.mean(embeddings, axis=0)
        centroid = centroid / (np.linalg.norm(centroid) + 1e-9)
        scores = [float(np.dot(emb, centroid)) for emb in embeddings]

        # Prefer shorter sentences when importance is similar (reduces output length)
        indexed = list(enumerate(scores))
        indexed.sort(
            key=lambda x: (x[1], -len(sentences[x[0]].split())),
            reverse=True,
        )
        top_indices = [i for i, _ in indexed[:k]]
        top_indices.sort()
        selected = [sentences[i] for i in top_indices]

    # Shorten each bullet to keep summary shorter than source
    shortened = [
        _shorten_sentence(
            s,
            max_words=max_words_per_bullet,
            remove_parentheticals=remove_parentheticals,
        )
        for s in selected
    ]
    return "\n".join(f"- {s}" for s in shortened if s.strip())


def bullet_summary(
    text: str,
    *,
    max_bullets: int = 15,
    ratio: float | None = 0.3,
    max_words_per_bullet: int | None = 20,
    remove_parentheticals: bool = True,
    mode: Literal["extractive", "llm"] = "extractive",
    llm: Optional[LLMClient] = None,
    embed_model: str = "n24q02m/Qwen3-Embedding-0.6B-ONNX-Q4F16",
) -> str:
    """Summarize text as bullet points.

    Args:
        text: Input text to summarize.
        max_bullets: Maximum number of bullet points (default 15).
        ratio: For extractive mode, cap bullets to this fraction of source
               sentences (e.g. 0.3 = max 30%). Ensures summary is shorter.
        max_words_per_bullet: For extractive mode, truncate each bullet to
                              at most this many words (default 20).
        remove_parentheticals: For extractive mode, strip [...] and (...).
        mode: "extractive" (default) for Qwen3-Embed-based extraction,
              "llm" for abstractive (Ollama/qwen3:4b).
        llm: LLMClient for llm mode (default: new instance).
        embed_model: Qwen3-Embed model name for extractive mode.

    Returns:
        Bullet-point summary string (each line starts with "- ").
    """
    if mode == "extractive":
        return bullet_summary_extractive(
            text,
            max_bullets=max_bullets,
            ratio=ratio,
            max_words_per_bullet=max_words_per_bullet,
            remove_parentheticals=remove_parentheticals,
            model_name=embed_model,
        )
    if mode == "llm":
        return bullet_summary_llm(text, max_bullets=max_bullets, llm=llm)
    raise ValueError(f"Unknown mode: {mode}. Use 'extractive' or 'llm'.")


def summarize_round(
    user_content: str,
    assistant_content: str,
    *,
    max_bullets: int = 5,
    mode: Literal["extractive", "llm"] = "llm",
    llm: Optional[LLMClient] = None,
) -> str:
    """Summarize a single chat round (one user message + one assistant reply).

    Used for session-scoped conversation summary: each round gets its own
    summarizer call. The result is intended to be appended to a running
    conversation_summary (ephemeral, not persisted).

    Args:
        user_content: The user's message in this round.
        assistant_content: The assistant's reply in this round.
        max_bullets: Maximum bullet points for this round (default 5).
        mode: "llm" (default) for abstractive via Ollama, "extractive" for
              Qwen3-Embed (requires qwen3-embed).
        llm: LLMClient for llm mode (default: new instance).

    Returns:
        Bullet-point summary of the round, or empty string if both inputs empty.
    """
    user = (user_content or "").strip()
    assistant = (assistant_content or "").strip()
    if not user and not assistant:
        return ""

    text = f"User: {user}\n\nAssistant: {assistant}"
    return bullet_summary(
        text,
        max_bullets=max_bullets,
        mode=mode,
        llm=llm,
    )


if __name__ == "__main__":
    import sys

    sample = """
    Machine learning is a subset of artificial intelligence that enables systems to learn from data.
    Deep learning uses neural networks with many layers to model complex patterns.
    Natural language processing helps computers understand and generate human language.
    Computer vision allows machines to interpret and analyze visual information from the world.
    Reinforcement learning trains agents through rewards and penalties in an environment.
    """
    mode = sys.argv[1] if len(sys.argv) > 1 else "extractive"
    print(f"Mode: {mode}\n")
    result = bullet_summary(sample.strip(), max_bullets=5, mode=mode)
    print(result)
