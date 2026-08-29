"""
Semantic document chunker for SEC 10-Q filings.

Strategy:
    1. Detect SEC section boundaries (reused from smart_chunker) — no token limit.
    2. Within each section, apply SemanticChunker (BGE-M3 embeddings) to split at
       topically coherent boundaries where inter-sentence similarity drops.
    3. Post-process: enforce 512-token ceiling, merge sub-128-token fragments,
       inject 256-character overlap between adjacent chunks.
    4. Generate dual-field output per chunk:
         text_for_search  – "{company quarter year} - {section}\\n\\n{chunk_with_overlap}"
                            embedded (BGE-M3) + BM25 indexed
         raw_content      – clean chunk text without overlap or title prefix
                            returned to the LLM after retrieval

Consistency principle: the same BGE-M3 model is used for both determining chunk
boundaries (SemanticChunker) and embedding chunks for search.  This ensures that
split points align with the semantic space used at retrieval time.
"""
import hashlib
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

import tiktoken
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import SemanticChunkConfig
from src.ingestion.pdf_parser import ParsedDocument
from src.ingestion.smart_chunker import (
    _build_page_index,
    _page_at,
    _split_into_sections,
)

logger = logging.getLogger(__name__)


# ── BGE-M3 model cache ────────────────────────────────────────────────────────

_bge_model_cache: dict = {}


class _OnnxBgeEmbeddings:
    """
    LangChain-compatible embeddings wrapper that runs BGE-M3 via ONNX Runtime.

    On first use the model is exported to ONNX (one-time ~3-5 min) and cached
    to ~/.cache/bge_onnx/<model>.  Subsequent runs load from the ONNX cache and
    run inference with onnxruntime, giving 3-5x faster CPU throughput than the
    PyTorch eager backend.

    Only the dense-embedding path is supported (last-hidden-state pooling + L2
    normalisation). Pooling mode (CLS or mean) is auto-detected from the model's
    sentence-transformers config; BGE-M3 uses CLS token.
    """

    _BATCH = 32  # sentences per onnxruntime call

    def __init__(self, model_name: str) -> None:
        import numpy as np
        import onnxruntime as ort
        from pathlib import Path
        from transformers import AutoTokenizer

        cache_dir = Path.home() / ".cache" / "bge_onnx" / model_name.replace("/", "--")
        onnx_path = cache_dir / "model.onnx"

        if not onnx_path.exists():
            _export_model_to_onnx(model_name, onnx_path)

        logger.info("Loading ONNX session from %s …", onnx_path)
        sess_opts = ort.SessionOptions()
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self._session = ort.InferenceSession(str(onnx_path), sess_options=sess_opts)
        self._input_names = {inp.name for inp in self._session.get_inputs()}
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._pooling = self._detect_pooling(model_name)
        self._np = np
        logger.info("ONNX session ready  pooling=%s", self._pooling)

    @staticmethod
    def _detect_pooling(model_name: str) -> str:
        """Return 'cls' or 'mean' by reading the sentence-transformers pooling config."""
        import json
        from pathlib import Path
        hf_cache = Path.home() / ".cache" / "huggingface" / "hub"
        slug = "models--" + model_name.replace("/", "--")
        for snap in (hf_cache / slug / "snapshots").glob("*/1_Pooling/config.json"):
            cfg = json.loads(snap.read_text())
            if cfg.get("pooling_mode_cls_token"):
                return "cls"
            if cfg.get("pooling_mode_mean_tokens"):
                return "mean"
        return "cls"  # BGE-M3 default

    def _pool_normalize(self, last_hidden: "np.ndarray", attention_mask: "np.ndarray") -> "np.ndarray":
        np = self._np
        if self._pooling == "cls":
            vecs = last_hidden[:, 0, :]
        else:
            mask = attention_mask[:, :, np.newaxis].astype(np.float32)
            vecs = (last_hidden * mask).sum(axis=1) / mask.sum(axis=1).clip(min=1e-9)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True).clip(min=1e-9)
        return vecs / norms

    def _encode_batch(self, texts: List[str]) -> "np.ndarray":
        np = self._np
        enc = self._tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="np",
        )
        inputs = {
            "input_ids":      enc["input_ids"].astype(np.int64),
            "attention_mask": enc["attention_mask"].astype(np.int64),
        }
        if "token_type_ids" in self._input_names and "token_type_ids" in enc:
            inputs["token_type_ids"] = enc["token_type_ids"].astype(np.int64)
        outputs = self._session.run(["last_hidden_state"], inputs)
        return self._pool_normalize(outputs[0], enc["attention_mask"])

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        all_vecs = []
        for i in range(0, len(texts), self._BATCH):
            all_vecs.extend(self._encode_batch(texts[i : i + self._BATCH]).tolist())
        return all_vecs

    def embed_query(self, text: str) -> List[float]:
        return self._encode_batch([text])[0].tolist()


def _export_model_to_onnx(model_name: str, onnx_path: "Path") -> None:
    """Export a HuggingFace sentence-encoder to ONNX (one-time cost)."""
    import torch
    from pathlib import Path
    from transformers import AutoModel, AutoTokenizer

    onnx_path = Path(onnx_path)
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info(
        "Exporting '%s' to ONNX at %s — one-time export, takes a few minutes …",
        model_name, onnx_path,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()

    dummy_enc = tokenizer(
        ["hello world", "dummy sentence for export"],
        padding=True, truncation=True, max_length=32, return_tensors="pt",
    )
    dummy_inputs = (dummy_enc["input_ids"], dummy_enc["attention_mask"])
    input_names = ["input_ids", "attention_mask"]
    if "token_type_ids" in dummy_enc:
        dummy_inputs += (dummy_enc["token_type_ids"],)
        input_names.append("token_type_ids")

    with torch.no_grad():
        torch.onnx.export(
            model,
            dummy_inputs,
            str(onnx_path),
            input_names=input_names,
            output_names=["last_hidden_state"],
            dynamic_axes={n: {0: "batch", 1: "seq"} for n in input_names}
                        | {"last_hidden_state": {0: "batch", 1: "seq"}},
            opset_version=14,
            do_constant_folding=True,
        )
    logger.info("ONNX export complete → %s", onnx_path)


def _get_bge_model(model_name: str, use_onnx: bool = True):
    """Load (or return cached) embedding model for BGE-M3."""
    cache_key = (model_name, use_onnx)
    if cache_key not in _bge_model_cache:
        if use_onnx:
            _bge_model_cache[cache_key] = _OnnxBgeEmbeddings(model_name)
        else:
            from langchain_huggingface import HuggingFaceEmbeddings
            logger.info("Loading BGE model '%s' via PyTorch (first use – may take a moment)…", model_name)
            _bge_model_cache[cache_key] = HuggingFaceEmbeddings(
                model_name=model_name,
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True},
            )
    return _bge_model_cache[cache_key]


def create_bge_embeddings(
    texts: List[str],
    model_name: str = "BAAI/bge-m3",
    use_onnx: bool = True,
) -> List[List[float]]:
    """
    Embed a list of texts using BGE-M3 (cached).

    Used by both the ingestion pipeline (embed text_for_search before storing
    in Weaviate) and the retriever (embed the query at search time).

    Args:
        texts:      Texts to embed.
        model_name: HuggingFace model ID (default: BAAI/bge-m3).
        use_onnx:   True → onnxruntime backend (model exported once to ~/.cache/bge_onnx/).
                    False → HuggingFaceEmbeddings PyTorch backend.
    """
    model = _get_bge_model(model_name, use_onnx)
    return model.embed_documents(texts)


# ── Data model ────────────────────────────────────────────────────────────────

@dataclass
class SemanticTextChunk:
    """A chunk produced by the semantic chunker."""
    text_for_search: str   # title-prefixed + overlap — embedded (BGE-M3) + BM25 indexed
    raw_content: str       # clean chunk text — returned to the LLM after retrieval
    content_hash: str      # SHA256 of raw_content — deduplication key in Weaviate
    chunk_index: int       # global index within the document
    token_count: int       # token count of raw_content
    page_num: int
    file_name: str
    source_file: str
    company: str
    quarter: str
    year: int
    section_title: str
    last_updated_date: str


# ── Token counter ─────────────────────────────────────────────────────────────

def _token_len(text: str, enc: tiktoken.Encoding) -> int:
    return len(enc.encode(text))


# ── Post-processing helpers ────────────────────────────────────────────────────

def _apply_token_ceiling(
    chunks: List[str],
    config: SemanticChunkConfig,
    enc: tiktoken.Encoding,
) -> List[str]:
    """
    Split any chunk above token_ceiling using RecursiveCharacterTextSplitter.
    This is a fallback — semantic boundaries should rarely produce oversized chunks
    when the threshold is calibrated correctly.
    """
    fallback = RecursiveCharacterTextSplitter(
        chunk_size=config.char_ceiling,
        chunk_overlap=config.fallback_chunk_overlap,
        separators=["\n\n", "\n", ". ", " "],
    )
    result: List[str] = []
    for chunk in chunks:
        if _token_len(chunk, enc) > config.token_ceiling:
            result.extend(fallback.split_text(chunk))
        else:
            result.append(chunk)
    return result


def _merge_small_chunks(
    chunks: List[str],
    config: SemanticChunkConfig,
    enc: tiktoken.Encoding,
) -> List[str]:
    """
    Merge fragments below min_char_size into the smaller of their two neighbours,
    as long as the merged result stays under the token ceiling.

    Prevents orphaned fragments that lack enough context for meaningful retrieval.
    """
    if not chunks:
        return chunks

    result = list(chunks)
    changed = True
    while changed:
        changed = False
        merged: List[str] = []
        i = 0
        while i < len(result):
            chunk = result[i]
            if len(chunk) < config.min_char_size and len(result) > 1:
                prev_merge = (merged[-1] + "\n" + chunk) if merged else None
                next_merge = (chunk + "\n" + result[i + 1]) if i + 1 < len(result) else None

                if prev_merge and (
                    not next_merge or len(prev_merge) <= len(next_merge)
                ) and _token_len(prev_merge, enc) <= config.token_ceiling:
                    merged[-1] = prev_merge
                    i += 1
                    changed = True
                elif next_merge and _token_len(next_merge, enc) <= config.token_ceiling:
                    merged.append(next_merge)
                    i += 2
                    changed = True
                else:
                    merged.append(chunk)
                    i += 1
            else:
                merged.append(chunk)
                i += 1
        result = merged

    return result


def _inject_overlap(chunks: List[str], overlap_chars: int) -> List[str]:
    """
    Prefix each chunk (index ≥ 1) with the last *overlap_chars* of the previous
    chunk, separated by a newline.  The returned strings are the text_for_search
    versions; callers retain the originals as raw_content.
    """
    result: List[str] = []
    for i, chunk in enumerate(chunks):
        if i == 0 or not chunks[i - 1]:
            result.append(chunk)
        else:
            tail = chunks[i - 1][-overlap_chars:]
            result.append(tail + "\n" + chunk)
    return result


# ── Public API ────────────────────────────────────────────────────────────────

def chunk_document_semantic(
    doc: ParsedDocument,
    config: SemanticChunkConfig,
    last_updated_date: Optional[str] = None,
) -> List[SemanticTextChunk]:
    """
    Produce SemanticTextChunks for a single ParsedDocument.

    Flow:
        pages → full_text → section split (SEC headers, no token cap)
              → SemanticChunker within each section (BGE-M3 boundaries)
              → post-process (ceiling, merge, overlap injection)
              → dual-field generation (text_for_search, raw_content)
    """
    if last_updated_date is None:
        last_updated_date = datetime.now(timezone.utc).isoformat()

    enc = tiktoken.get_encoding(config.encoding)

    full_text, page_starts, page_nums = _build_page_index(doc)
    if not full_text.strip():
        return []

    sections = _split_into_sections(full_text)
    logger.debug("SectionSplit '%s' → %d sections", doc.source_file, len(sections))

    # Lazy import — avoids paying the model-load cost unless semantic chunking is used.
    from langchain_experimental.text_splitter import SemanticChunker
    bge = _get_bge_model(config.embedding_model, config.use_onnx)
    semantic_splitter = SemanticChunker(
        embeddings=bge,
        breakpoint_threshold_type=config.breakpoint_threshold_type,
        breakpoint_threshold_amount=config.breakpoint_threshold_amount,
    )

    chunks: List[SemanticTextChunk] = []
    chunk_index = 0

    for section_text, section_title, start_offset in sections:
        if not section_text.strip():
            continue

        section_page = _page_at(start_offset, page_starts, page_nums)

        # Semantic split within this section (each section processed independently)
        raw_splits = semantic_splitter.split_text(section_text)
        if not raw_splits:
            continue

        # Post-processing
        splits = _apply_token_ceiling(raw_splits, config, enc)
        splits = _merge_small_chunks(splits, config, enc)
        overlapped = _inject_overlap(splits, config.overlap_chars)

        # Title prefix for text_for_search: filing identity + section
        title = f"{doc.company} {doc.quarter} {doc.year}"
        if section_title:
            title += f" - {section_title}"

        for raw_content, search_content in zip(splits, overlapped):
            content_hash = hashlib.sha256(raw_content.encode("utf-8")).hexdigest()
            text_for_search = f"{title}\n\n{search_content}"

            chunks.append(SemanticTextChunk(
                text_for_search=text_for_search,
                raw_content=raw_content,
                content_hash=content_hash,
                chunk_index=chunk_index,
                token_count=_token_len(raw_content, enc),
                page_num=section_page,
                file_name=Path(doc.source_file).name,
                source_file=doc.source_file,
                company=doc.company,
                quarter=doc.quarter,
                year=doc.year,
                section_title=section_title,
                last_updated_date=last_updated_date,
            ))
            chunk_index += 1

    logger.info(
        "SemanticChunker '%s' → %d sections → %d chunks",
        doc.source_file, len(sections), len(chunks),
    )
    return chunks


def chunk_documents_semantic(
    docs: List[ParsedDocument],
    config: SemanticChunkConfig,
    last_updated_date: Optional[str] = None,
) -> List[SemanticTextChunk]:
    """Chunk all parsed documents and return a flat list of SemanticTextChunks."""
    if last_updated_date is None:
        last_updated_date = datetime.now(timezone.utc).isoformat()
    all_chunks: List[SemanticTextChunk] = []
    for doc in docs:
        all_chunks.extend(chunk_document_semantic(doc, config, last_updated_date))
    logger.info(
        "SemanticChunker total across %d documents: %d chunks",
        len(docs), len(all_chunks),
    )
    return all_chunks
