from typing import List
from config import Config


import re

def recursive_split(text: str, chunk_size: int = None, chunk_overlap: int = None) -> List[str]:
    """
    Chunking by paragraphs as requested.
    This replaces the character-limit recursive chunking with strict semantic paragraph chunks.
    It removes single newlines (hard wraps common in PDFs) but preserves double newlines as paragraph boundaries.
    """
    # Split text strictly into blocks by double newlines
    blocks = re.split(r'\n\s*\n', text)
    
    # Fallback for messy PDFs: if blocks are still huge, try to aggressively split by period+newline
    refined_blocks = []
    for block in blocks:
        if len(block) > 1000 and '\n' in block:
            # Look for lines ending with a period (or !?) as potential paragraph breaks
            sub_blocks = re.split(r'(?<=[.!?])\s*\n', block)
            refined_blocks.extend(sub_blocks)
        else:
            refined_blocks.append(block)
            
    chunks = []
    for block in refined_blocks:
        # Reflow the text inside the paragraph (turn single newlines into spaces)
        reflowed = re.sub(r'(?<!\n)\n(?!\n)', ' ', block).strip()
        
        # We optionally split massive paragraphs if they are ridiculously huge, 
        # but the request was "chunking por parrafos" so we keep them intact.
        if reflowed:
            chunks.append(reflowed)
            
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
