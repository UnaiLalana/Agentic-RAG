from typing import List
from config import Config


def recursive_split(text: str, chunk_size: int = None, chunk_overlap: int = None) -> List[str]:
    """
    Recursive text chunking strategy.

    Splits text hierarchically:
    1. First by double newlines (paragraphs)
    2. Then by single newlines (lines)
    3. Then by sentences ('. ')
    4. Finally by words if still too large

    Each chunk has configurable overlap with the next chunk.

    Args:
        text: The full document text.
        chunk_size: Maximum characters per chunk.
        chunk_overlap: Number of overlapping characters between chunks.

    Returns:
        List of text chunks.
    """
    if chunk_size is None:
        chunk_size = Config.CHUNK_SIZE
    if chunk_overlap is None:
        chunk_overlap = Config.CHUNK_OVERLAP

    # Separators in order of preference
    separators = ["\n\n", "\n", ". ", " "]

    chunks = _split_recursive(text, separators, chunk_size)

    # Add overlap between chunks
    if chunk_overlap > 0 and len(chunks) > 1:
        chunks = _add_overlap(chunks, chunk_overlap)

    # Filter out empty chunks
    chunks = [c.strip() for c in chunks if c.strip()]

    return chunks


def _split_recursive(text: str, separators: List[str], chunk_size: int) -> List[str]:
    """Recursively split text using a hierarchy of separators."""
    if len(text) <= chunk_size:
        return [text]

    if not separators:
        # Last resort: hard split by character
        chunks = []
        for i in range(0, len(text), chunk_size):
            chunks.append(text[i:i + chunk_size])
        return chunks

    current_sep = separators[0]
    remaining_seps = separators[1:]

    parts = text.split(current_sep)

    chunks = []
    current_chunk = ""

    for part in parts:
        # If adding this part would exceed the limit
        test_chunk = current_chunk + current_sep + part if current_chunk else part

        if len(test_chunk) <= chunk_size:
            current_chunk = test_chunk
        else:
            # Save current chunk if it has content
            if current_chunk:
                chunks.append(current_chunk)

            # If the part itself is too large, split it further
            if len(part) > chunk_size:
                sub_chunks = _split_recursive(part, remaining_seps, chunk_size)
                chunks.extend(sub_chunks)
                current_chunk = ""
            else:
                current_chunk = part

    if current_chunk:
        chunks.append(current_chunk)

    return chunks


def _add_overlap(chunks: List[str], overlap: int) -> List[str]:
    """Add overlapping text between consecutive chunks."""
    overlapped = [chunks[0]]

    for i in range(1, len(chunks)):
        prev_chunk = chunks[i - 1]
        # Take the last `overlap` characters from the previous chunk
        overlap_text = prev_chunk[-overlap:] if len(prev_chunk) >= overlap else prev_chunk
        overlapped.append(overlap_text + chunks[i])

    return overlapped
