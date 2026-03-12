"""
Token-aware document chunker for SEC 10-Q filings.

Strategy (Phase 1 – demo grade):
    1.  Split each page into sentence-like segments (split on sentence-ending
        punctuation).
    2.  Greedily accumulate segments into a chunk until it would exceed
        max_tokens (800).
    3.  When a chunk reaches min_tokens (500) and the next segment would
        overflow, flush the chunk.
    4.  Apply a 100-token overlap: after flushing, back-fill the tail of the
        previous chunk so adjacent chunks share context.

Phase-1 limitation:
    Sentence splitting is a simple regex – it breaks on ".  " patterns, which
    may split mid-sentence inside tables or financial figures.  Phase 2 will
    add a proper sentence tokeniser and table-aware splitter.
"""
import logging
import re
from dataclasses import dataclass
from typing import List, Tuple

import tiktoken

from src.config import ChunkConfig
from src.ingestion.pdf_parser import ParsedDocument

logger = logging.getLogger(__name__)


# ── Data model ────────────────────────────────────────────────────────────────

@dataclass
class TextChunk:
    text: str
    chunk_index: int
    token_count: int
    page_start: int
    page_end: int
    source_file: str
    company: str
    quarter: str
    year: int


# ── Internal helpers ──────────────────────────────────────────────────────────

def _sentence_split(text: str) -> List[str]:
    """
    Split text into sentence-like segments on sentence-boundary punctuation.
    Keeps each segment non-empty.
    """
    # Split after '.', '!', '?' followed by whitespace (handles mid-para breaks)
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [p.strip() for p in parts if p.strip()]


def _token_len(text: str, enc: tiktoken.Encoding) -> int:
    return len(enc.encode(text))


def _segments_from_document(
    doc: ParsedDocument,
) -> List[Tuple[str, int]]:
    """Return list of (segment_text, page_num) tuples for a document."""
    result = []
    for page in doc.pages:
        for seg in _sentence_split(page.text):
            result.append((seg, page.page_num))
    return result


# ── Core chunking logic ───────────────────────────────────────────────────────

def chunk_document(doc: ParsedDocument, config: ChunkConfig) -> List[TextChunk]:
    """Chunk a single ParsedDocument into TextChunk objects."""
    enc = tiktoken.get_encoding(config.encoding)
    segments = _segments_from_document(doc)
    chunks: List[TextChunk] = []

    # Working state
    cur_texts: List[str] = []
    cur_pages: List[int] = []
    cur_tokens: int = 0
    chunk_index = 0

    i = 0
    while i < len(segments):
        seg_text, page_num = segments[i]
        seg_tokens = _token_len(seg_text, enc)

        if cur_tokens + seg_tokens <= config.max_tokens:
            # Fits – accumulate
            cur_texts.append(seg_text)
            cur_pages.append(page_num)
            cur_tokens += seg_tokens
            i += 1
        else:
            if cur_tokens >= config.min_tokens:
                # Flush current chunk
                chunk_text = " ".join(cur_texts)
                chunks.append(TextChunk(
                    text=chunk_text,
                    chunk_index=chunk_index,
                    token_count=cur_tokens,
                    page_start=min(cur_pages),
                    page_end=max(cur_pages),
                    source_file=doc.source_file,
                    company=doc.company,
                    quarter=doc.quarter,
                    year=doc.year,
                ))
                chunk_index += 1

                # ── Overlap: walk backwards collecting ~overlap_tokens ──────
                overlap_tokens = 0
                overlap_start = len(cur_texts) - 1
                while overlap_start > 0:
                    t = _token_len(cur_texts[overlap_start], enc)
                    if overlap_tokens + t > config.overlap_tokens:
                        break
                    overlap_tokens += t
                    overlap_start -= 1

                # Start next chunk from the overlap tail
                cur_texts = cur_texts[overlap_start + 1:]
                cur_pages = cur_pages[overlap_start + 1:]
                cur_tokens = sum(_token_len(t, enc) for t in cur_texts)
                # Don't advance i – re-evaluate the overflowing segment
            else:
                # Not enough tokens yet; accept even if > max to avoid tiny chunks
                cur_texts.append(seg_text)
                cur_pages.append(page_num)
                cur_tokens += seg_tokens
                i += 1

    # Flush any remaining content
    if cur_texts:
        chunk_text = " ".join(cur_texts)
        chunks.append(TextChunk(
            text=chunk_text,
            chunk_index=chunk_index,
            token_count=_token_len(chunk_text, enc),
            page_start=min(cur_pages),
            page_end=max(cur_pages),
            source_file=doc.source_file,
            company=doc.company,
            quarter=doc.quarter,
            year=doc.year,
        ))

    logger.info(
        "Chunked '%s' → %d chunks  (min=%d max=%d overlap=%d tokens)",
        doc.source_file, len(chunks),
        config.min_tokens, config.max_tokens, config.overlap_tokens,
    )
    return chunks


def chunk_documents(docs: List[ParsedDocument], config: ChunkConfig) -> List[TextChunk]:
    """Chunk all parsed documents and return a flat list of TextChunks."""
    all_chunks: List[TextChunk] = []
    for doc in docs:
        all_chunks.extend(chunk_document(doc, config))

    logger.info(
        "Total chunks across %d documents: %d", len(docs), len(all_chunks)
    )
    return all_chunks
