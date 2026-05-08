import time


def compute_grounding_score(response_text: str, chunks: list[dict]) -> float:
    """
    Simple keyword overlap score: what fraction of significant words
    in the response appear in the retrieved chunks.
    """
    if not chunks or not response_text:
        return 0.0

    stop_words = {"the", "a", "an", "is", "are", "was", "were", "be", "been",
                  "have", "has", "had", "do", "does", "did", "will", "would",
                  "could", "should", "may", "might", "to", "of", "in", "on",
                  "at", "by", "for", "with", "about", "and", "or", "but",
                  "if", "as", "it", "its", "this", "that", "your", "my"}

    response_words = {
        w.lower().strip(".,;:()[]") 
        for w in response_text.split() 
        if len(w) > 3 and w.lower() not in stop_words
    }

    corpus = " ".join(c.get("excerpt", "") for c in chunks).lower()
    corpus_words = set(corpus.split())

    if not response_words:
        return 0.0

    overlap = response_words & corpus_words
    return round(len(overlap) / len(response_words), 3)


def count_tokens_approx(text: str) -> int:
    """Approximate token count: ~4 chars per token."""
    return len(text) // 4
