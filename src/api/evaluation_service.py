"""
Core logic for the evaluation-run web UI: scanning past results on disk,
looking up whether a given parameter combination has already been evaluated,
and launching/monitoring/cancelling scripts/run_evaluation.py as a subprocess.

Job tracking is in-memory only (_JOBS). This is a known, accepted limitation
for a local single-user dev tool — job state does not survive a backend
restart, but actual results are safe regardless because they're written to
CSV files on disk, which remain the source of truth.

Do not run this API with `--reload` while a real evaluation job is active
(the reloader can kill the subprocess), and do not run it with multiple
Uvicorn workers (the in-memory _JOBS dict only exists in one process).
"""
import asyncio
import hashlib
import logging
import re
import sys
import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Deque, Dict, List, Optional, Tuple

import pandas as pd
from fastapi import HTTPException

from src.api.models import (
    EvaluationRunParams,
    HistoryRunSummary,
    JobStatus,
    MetricRow,
    OptionsResponse,
    RunDetail,
    SampleRow,
    StartRunResponse,
)
from src.config import Config
from src.evaluation.run_tag import build_run_tag, resolve_chunking_strategy

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
EVAL_RESULTS_DIR = REPO_ROOT / "evaluation_results"
SCRIPT_PATH = REPO_ROOT / "scripts" / "run_evaluation.py"

_config = Config()
_JOBS: Dict[str, "JobRecord"] = {}

_LOG_RING_SIZE = 300
_PROGRESS_RE = re.compile(r"\[(\d+)/(\d+)\]\s+RAG\s+\[[^\]]*\]:")

_ALPHA_RE = re.compile(r"_a([\d.]+)(?=_|$)")
_TOPK_RE = re.compile(r"_k(\d+)(?=_|$)")
_FILTER_RE = re.compile(r"(llm|regex)filters")
_RERANK_RE = re.compile(r"_(llm|cross_encoder)rerank")
_RERANK_TOPK_RE = re.compile(r"_rt(\d+)(?=_|$)")
_ENGINE_PREFIXES = ("custom", "llamaindex")

_CORE_MATCH_FIELDS = ("engine", "chunking_strategy", "retrieval_mode", "filter_mode", "rerank_mode")
_PARTIAL_MATCH_FIELDS = ("alpha", "top_k", "rerank_top_k")


# ═════════════════════════════════════════════════════════════════════════════
# History scanning
# ═════════════════════════════════════════════════════════════════════════════

def _is_summary_csv(path: Path) -> bool:
    """Detect a summary CSV by its header (metric,mean,...), not by filename —
    several legacy files don't follow the "_summary.csv" suffix convention."""
    try:
        with path.open("r", encoding="utf-8", errors="replace") as fh:
            header = fh.readline()
    except OSError:
        return False
    first_col = header.split(",", 1)[0].strip().strip('"')
    return first_col == "metric"


def _count_data_rows(csv_path: Path) -> Optional[int]:
    try:
        return int(len(pd.read_csv(csv_path)))
    except Exception:
        return None


def _make_id(csv_path: Path) -> str:
    rel = csv_path.resolve().relative_to(REPO_ROOT).as_posix()
    return hashlib.sha1(rel.encode("utf-8")).hexdigest()[:16]


def parse_run_tag_tokens(stem: str) -> dict:
    """
    Best-effort extraction of run parameters from a filename stem.

    Used only for display and fuzzy "close match" scoring — exact-match
    lookups always go through compute_run_tag()/build_run_tag(), never
    through this parser, since filenames are historically inconsistent
    (see module docstring in src/evaluation/run_tag.py).
    """
    tokens: dict = {}
    working = stem[len("evaluation_results_"):] if stem.startswith("evaluation_results_") else stem

    if m := _ALPHA_RE.search(working):
        try:
            tokens["alpha"] = float(m.group(1))
        except ValueError:
            pass
    if m := _TOPK_RE.search(working):
        tokens["top_k"] = int(m.group(1))
    if m := _FILTER_RE.search(working):
        tokens["filter_mode"] = m.group(1)
    if m := _RERANK_RE.search(working):
        tokens["rerank_mode"] = m.group(1)
    if m := _RERANK_TOPK_RE.search(working):
        tokens["rerank_top_k"] = int(m.group(1))

    segments = [s for s in working.split("_") if s]
    if segments and segments[0] in _ENGINE_PREFIXES:
        tokens["engine"] = segments[0]

    if "parent_child" in working or re.search(r"(^|_)smart(_|$)", working):
        tokens["chunking_strategy"] = "parent_child"
    elif re.search(r"(^|_)basic(_|$)", working):
        tokens["chunking_strategy"] = "basic"
    elif re.search(r"(^|_)semantic(_|$)", working):
        # The first "semantic" occurrence is almost always the chunking-strategy
        # position in build_run_tag's <engine>_<chunking>_<mode> ordering.
        tokens["chunking_strategy"] = "semantic"

    if re.search(r"(^|_)hybrid(_|$)", working):
        tokens["retrieval_mode"] = "hybrid"

    if tokens.get("engine") and tokens.get("chunking_strategy") and tokens.get("retrieval_mode"):
        tokens["run_tag_guess"] = working

    return tokens


def _build_history_entry(csv_path: Path, summary_path: Optional[Path], source: str) -> HistoryRunSummary:
    stem = csv_path.stem
    tokens = parse_run_tag_tokens(stem)
    row_count = _count_data_rows(csv_path) if not _is_summary_csv(csv_path) else None
    return HistoryRunSummary(
        id=_make_id(csv_path),
        source=source,  # type: ignore[arg-type]
        csv_path=str(csv_path.relative_to(REPO_ROOT)),
        summary_csv_path=str(summary_path.relative_to(REPO_ROOT)) if summary_path else None,
        display_name=stem,
        run_tag_guess=tokens.get("run_tag_guess"),
        parsed_params=tokens,
        modified_at=datetime.fromtimestamp(csv_path.stat().st_mtime),
        row_count=row_count,
    )


def scan_history() -> List[HistoryRunSummary]:
    """
    Scan evaluation_results/*.csv and root-level evaluation_results_*.csv,
    pairing each per-sample CSV with its summary CSV where one exists.
    Summary CSVs without a discoverable pair (legacy manually-renamed files)
    are surfaced as standalone entries rather than dropped.
    """
    entries: List[HistoryRunSummary] = []

    for directory, source, pattern in (
        (EVAL_RESULTS_DIR, "evaluation_results_dir", "*.csv"),
        (REPO_ROOT, "root", "evaluation_results_*.csv"),
    ):
        if not directory.is_dir():
            continue

        csv_files = sorted(directory.glob(pattern))
        summary_files = {p.stem: p for p in csv_files if _is_summary_csv(p)}
        sample_files = [p for p in csv_files if not _is_summary_csv(p)]
        consumed_summary_stems: set = set()

        for sample_path in sample_files:
            paired_stem = f"{sample_path.stem}_summary"
            summary_path = summary_files.get(paired_stem)
            if summary_path is not None:
                consumed_summary_stems.add(paired_stem)
            entries.append(_build_history_entry(sample_path, summary_path, source))

        for stem, summary_path in summary_files.items():
            if stem not in consumed_summary_stems:
                entries.append(_build_history_entry(summary_path, summary_path, source))

    entries.sort(key=lambda e: e.modified_at, reverse=True)
    return entries


# ═════════════════════════════════════════════════════════════════════════════
# Lookup (exact + fuzzy)
# ═════════════════════════════════════════════════════════════════════════════

def compute_run_tag(params: EvaluationRunParams) -> str:
    chunking_strategy = resolve_chunking_strategy(params.chunking_strategy, params.use_smart)
    return build_run_tag(
        engine=params.engine,
        chunking_strategy=chunking_strategy,
        filter_mode=params.filter_mode,
        config_retrieval_mode=_config.retrieval.mode,
        retrieval_mode=params.retrieval_mode,
        alpha=params.alpha,
        top_k=params.top_k,
        rerank_mode=params.rerank_mode,
        rerank_top_k=params.rerank_top_k,
        diversity_mode=params.diversity_mode,
        mmr_lambda=params.mmr_lambda,
        per_filing=params.per_filing,
        chunks_per_filing=params.chunks_per_filing,
        max_per_entity=params.max_per_entity,
        question_type=params.question_type,
        seed=params.seed,
    )


def find_matches(params: EvaluationRunParams) -> Tuple[Optional[HistoryRunSummary], List[HistoryRunSummary]]:
    run_tag = compute_run_tag(params)
    history = scan_history()

    exact_filename = f"evaluation_results_{run_tag}.csv"
    candidates = [e for e in history if Path(e.csv_path).name == exact_filename]
    exact = next((e for e in candidates if e.source == "root"), None) or (candidates[0] if candidates else None)

    resolved_chunking = resolve_chunking_strategy(params.chunking_strategy, params.use_smart)
    resolved_mode = params.retrieval_mode or _config.retrieval.mode
    requested = {
        "engine": params.engine,
        "chunking_strategy": resolved_chunking,
        "retrieval_mode": resolved_mode,
        "filter_mode": params.filter_mode,
        "rerank_mode": params.rerank_mode,
        "alpha": params.alpha,
        "top_k": params.top_k,
        "rerank_top_k": params.rerank_top_k,
    }

    scored: List[Tuple[int, HistoryRunSummary]] = []
    for entry in history:
        if exact is not None and entry.id == exact.id:
            continue
        score = 0
        for f in (*_CORE_MATCH_FIELDS, *_PARTIAL_MATCH_FIELDS):
            parsed_val = entry.parsed_params.get(f)
            if parsed_val is not None and parsed_val == requested.get(f):
                score += 1
        if score >= 2:
            scored.append((score, entry))

    scored.sort(key=lambda pair: pair[0], reverse=True)
    close = [entry for _, entry in scored[:5]]

    return exact, close


# ═════════════════════════════════════════════════════════════════════════════
# CSV parsing
# ═════════════════════════════════════════════════════════════════════════════

def _na(value):
    """
    Convert pandas NaN/NaT to None; pass everything else through unchanged.

    DataFrame.where(cond, None) silently coerces None back to NaN for
    float64 columns (pandas keeps the column's numeric dtype), so it does
    NOT reliably strip NaN before Pydantic serialization. Starlette's
    JSONResponse uses allow_nan=False, so any leftover NaN crashes the
    response with "Out of range float values are not JSON compliant" —
    this per-value check is what actually prevents that.
    """
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value


_META_PREFIX = "_meta_"


def read_summary(summary_csv_path: Path) -> List[MetricRow]:
    """
    Parse the numeric metric rows from a summary CSV.

    Newer summary CSVs also carry self-documenting "_meta_*" rows (input
    file, run tag, question count) appended by run_evaluation.py — those
    store strings/paths in the "mean" column, which is not a valid MetricRow
    value, so they're skipped here and read separately via read_summary_meta().
    """
    df = pd.read_csv(summary_csv_path)
    rows: List[MetricRow] = []
    for record in df.to_dict("records"):
        metric_name = str(record.get("metric"))
        if metric_name.startswith(_META_PREFIX):
            continue
        rows.append(MetricRow(
            metric=metric_name,
            mean=_na(record.get("mean")),
            std=_na(record.get("std")),
            min=_na(record.get("min")),
            max=_na(record.get("max")),
            avg_pct=_na(record.get("avg_pct")),
            flagged_count=_na(record.get("flagged_count")),
            total=_na(record.get("total")),
            flagged_pct=_na(record.get("flagged_pct")),
            threshold=_na(record.get("threshold")),
            note=_na(record.get("note")),
        ))
    return rows


def read_summary_meta(summary_csv_path: Path) -> Dict[str, str]:
    """Extract "_meta_*" rows (e.g. _meta_input_file, _meta_run_tag,
    _meta_n_questions) into a plain {key: value} dict, prefix stripped."""
    df = pd.read_csv(summary_csv_path)
    meta: Dict[str, str] = {}
    for record in df.to_dict("records"):
        metric_name = str(record.get("metric"))
        if metric_name.startswith(_META_PREFIX):
            value = _na(record.get("mean"))
            if value is not None:
                meta[metric_name[len(_META_PREFIX):]] = str(value)
    return meta


def read_samples(csv_path: Path) -> List[SampleRow]:
    df = pd.read_csv(csv_path)
    rows: List[SampleRow] = []
    for record in df.to_dict("records"):
        rows.append(SampleRow(
            question=str(_na(record.get("question")) or ""),
            question_type=_na(record.get("question_type")),
            source_chunk_type=_na(record.get("source_chunk_type")),
            golden_answer=str(_na(record.get("golden_answer")) or ""),
            generated_answer=str(_na(record.get("generated_answer")) or ""),
            exactness=_na(record.get("exactness")),
            answer_similarity=_na(record.get("answer_similarity")),
            correctness=_na(record.get("correctness")),
            context_precision=_na(record.get("context_precision")),
            context_recall=_na(record.get("context_recall")),
            context_relevance=_na(record.get("context_relevance")),
            faithfulness=_na(record.get("faithfulness")),
            hallucination_rate=_na(record.get("hallucination_rate")),
            answer_relevance=_na(record.get("answer_relevance")),
            factual_error_rate=_na(record.get("factual_error_rate")),
        ))
    return rows


def get_history_detail(run_id: str) -> RunDetail:
    """
    Look up a past run by its server-derived id.

    The id is only ever produced by scan_history() from paths the server
    itself discovered — a client can never submit a path directly, so an
    unknown/forged id simply misses the freshly-rebuilt map and 404s. This
    is what keeps this endpoint free of path-traversal risk.
    """
    history = scan_history()
    entry = next((e for e in history if e.id == run_id), None)
    if entry is None:
        raise HTTPException(status_code=404, detail=f"No evaluation run found for id '{run_id}'.")

    csv_path = REPO_ROOT / entry.csv_path
    summary_path = REPO_ROOT / entry.summary_csv_path if entry.summary_csv_path else None

    metrics = read_summary(summary_path) if summary_path is not None else []
    meta = read_summary_meta(summary_path) if summary_path is not None else {}
    samples = read_samples(csv_path) if not _is_summary_csv(csv_path) else []

    return RunDetail(summary=entry, metrics=metrics, samples=samples, meta=meta)


# ═════════════════════════════════════════════════════════════════════════════
# Options
# ═════════════════════════════════════════════════════════════════════════════

def get_options() -> OptionsResponse:
    from src.evaluation.evaluator import ALL_METRICS

    return OptionsResponse(
        engines=["custom", "llamaindex"],
        chunking_strategies=["basic", "parent_child", "semantic"],
        retrieval_modes=["semantic", "hybrid"],
        filter_modes=["llm", "regex"],
        rerank_modes=["llm", "cross_encoder"],
        diversity_modes=["none", "mmr", "metadata_slots", "source_cap"],
        metrics=sorted(ALL_METRICS),
        default_top_k=_config.retrieval.top_k,
        default_samples=25,
        default_seed=42,
    )


# ═════════════════════════════════════════════════════════════════════════════
# Job lifecycle
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class JobRecord:
    job_id: str
    status: str
    params: EvaluationRunParams
    run_tag: str
    process: Optional[asyncio.subprocess.Process] = None
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
    progress_current: Optional[int] = None
    progress_total: Optional[int] = None
    log_ring: Deque[str] = field(default_factory=lambda: deque(maxlen=_LOG_RING_SIZE))
    exit_code: Optional[int] = None
    error_message: Optional[str] = None
    output_csv_path: Optional[Path] = None
    summary_csv_path: Optional[Path] = None
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    def to_status(self) -> JobStatus:
        return JobStatus(
            job_id=self.job_id,
            status=self.status,  # type: ignore[arg-type]
            run_tag=self.run_tag,
            params=self.params,
            started_at=self.started_at,
            finished_at=self.finished_at,
            progress_current=self.progress_current,
            progress_total=self.progress_total,
            log_tail=list(self.log_ring),
            exit_code=self.exit_code,
            error_message=self.error_message,
            output_csv_path=str(self.output_csv_path.relative_to(REPO_ROOT)) if self.output_csv_path else None,
            summary_csv_path=str(self.summary_csv_path.relative_to(REPO_ROOT)) if self.summary_csv_path else None,
        )


def build_argv(params: EvaluationRunParams) -> List[str]:
    """Build the CLI argv for scripts/run_evaluation.py, only appending a flag
    when the corresponding field is explicitly set (mirrors CLI optionality
    exactly, so the run tag the CLI computes matches compute_run_tag())."""
    chunking_strategy = resolve_chunking_strategy(params.chunking_strategy, params.use_smart)
    argv: List[str] = [
        "--engine", params.engine,
        "--chunking-strategy", chunking_strategy,
        "--filter-mode", params.filter_mode,
        "--samples", str(params.samples),
        "--seed", str(params.seed),
    ]
    if params.retrieval_mode is not None:
        argv += ["--retrieval-mode", params.retrieval_mode]
    if params.alpha is not None:
        argv += ["--alpha", str(params.alpha)]
    if params.top_k is not None:
        argv += ["--top-k", str(params.top_k)]
    if params.rerank_mode is not None:
        argv += ["--rerank-mode", params.rerank_mode]
    if params.rerank_top_k is not None:
        argv += ["--rerank-top-k", str(params.rerank_top_k)]
    if params.diversity_mode and params.diversity_mode != "none":
        argv += ["--diversity-mode", params.diversity_mode]
    if params.diversity_mode == "mmr" and params.mmr_lambda != 0.5:
        argv += ["--mmr-lambda", str(params.mmr_lambda)]
    if params.diversity_mode == "source_cap" and params.max_per_entity is not None:
        argv += ["--max-per-entity", str(params.max_per_entity)]
    if params.per_filing:
        argv += ["--per-filing"]
        if params.chunks_per_filing != 3:
            argv += ["--chunks-per-filing", str(params.chunks_per_filing)]
    if params.company:
        argv += ["--company", params.company]
    if params.question_type:
        argv += ["--question-type", params.question_type]
    if params.metrics:
        argv += ["--metrics", ",".join(params.metrics)]
    if params.skip_ragas:
        argv += ["--skip-ragas"]
    if params.input_path:
        argv += ["--input", params.input_path]
    return argv


async def start_run(params: EvaluationRunParams, force: bool) -> StartRunResponse:
    run_tag = compute_run_tag(params)

    if not force:
        exact, _ = find_matches(params)
        if exact is not None:
            raise HTTPException(
                status_code=409,
                detail={
                    "message": f"An evaluation for this configuration already exists: {exact.display_name}",
                    "existing": exact.model_dump(mode="json"),
                },
            )

    job_id = uuid.uuid4().hex[:12]
    argv = [sys.executable, "-u", str(SCRIPT_PATH), *build_argv(params)]

    # -u (unbuffered): without it, Python fully block-buffers stdout/stderr
    # when piped, so log lines would arrive in one burst at exit instead of
    # streaming — silently defeating live progress.
    # stderr=STDOUT: run_evaluation.py's logging.basicConfig() has no stream=
    # argument, so its per-question progress lines go to stderr while
    # print_summary()'s tables go to stdout — both must merge onto one
    # stream for the progress regex below to see them.
    process = await asyncio.create_subprocess_exec(
        *argv,
        cwd=str(REPO_ROOT),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )

    job = JobRecord(
        job_id=job_id,
        status="running",
        params=params,
        run_tag=run_tag,
        process=process,
        started_at=datetime.now(),
    )
    _JOBS[job_id] = job

    asyncio.create_task(_stream_output(job_id))

    logger.info("Started evaluation job %s (run_tag=%s) argv=%s", job_id, run_tag, argv)
    return StartRunResponse(job_id=job_id, run_tag=run_tag, status=job.status)


async def _stream_output(job_id: str) -> None:
    job = _JOBS.get(job_id)
    if job is None or job.process is None or job.process.stdout is None:
        return

    try:
        async for raw_line in job.process.stdout:
            line = raw_line.decode("utf-8", errors="replace").rstrip("\n")
            if not line:
                continue
            job.log_ring.append(line)
            match = _PROGRESS_RE.search(line)
            if match:
                job.progress_current = int(match.group(1))
                job.progress_total = int(match.group(2))
    except Exception as exc:
        logger.exception("Error while streaming output for job %s", job_id)
        job.log_ring.append(f"[stream error] {exc}")

    exit_code = await job.process.wait()

    async with job.lock:
        if job.status != "running":
            # A concurrent cancel already finalized this job — don't override it.
            return
        job.exit_code = exit_code
        job.finished_at = datetime.now()

        if exit_code == 0:
            output_csv = REPO_ROOT / f"evaluation_results_{job.run_tag}.csv"
            summary_csv = REPO_ROOT / f"evaluation_results_{job.run_tag}_summary.csv"
            if output_csv.exists() and summary_csv.exists():
                job.output_csv_path = output_csv
                job.summary_csv_path = summary_csv
                job.status = "succeeded"
            else:
                job.status = "failed"
                missing = output_csv.name if not output_csv.exists() else summary_csv.name
                job.error_message = f"Process exited 0 but expected output file not found: {missing}"
        else:
            job.status = "failed"
            job.error_message = "\n".join(list(job.log_ring)[-20:])


def _get_job_or_404(job_id: str) -> JobRecord:
    job = _JOBS.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"No job found for id '{job_id}'.")
    return job


def get_job_status(job_id: str) -> JobStatus:
    return _get_job_or_404(job_id).to_status()


async def cancel_job(job_id: str) -> JobStatus:
    job = _get_job_or_404(job_id)

    async with job.lock:
        if job.status in ("queued", "running") and job.process is not None and job.process.returncode is None:
            job.process.terminate()
            try:
                await asyncio.wait_for(job.process.wait(), timeout=5.0)
            except asyncio.TimeoutError:
                job.process.kill()
                await job.process.wait()
            job.status = "cancelled"
            job.finished_at = datetime.now()
            job.exit_code = job.process.returncode

    return job.to_status()


def get_job_result(job_id: str) -> RunDetail:
    job = _get_job_or_404(job_id)
    if job.status != "succeeded" or job.output_csv_path is None or job.summary_csv_path is None:
        raise HTTPException(
            status_code=409,
            detail=f"Job '{job_id}' has not succeeded yet (status={job.status}).",
        )

    summary_entry = HistoryRunSummary(
        id=_make_id(job.output_csv_path),
        source="root",
        csv_path=str(job.output_csv_path.relative_to(REPO_ROOT)),
        summary_csv_path=str(job.summary_csv_path.relative_to(REPO_ROOT)),
        display_name=job.output_csv_path.stem,
        run_tag_guess=job.run_tag,
        parsed_params=parse_run_tag_tokens(job.output_csv_path.stem),
        modified_at=datetime.fromtimestamp(job.output_csv_path.stat().st_mtime),
        row_count=_count_data_rows(job.output_csv_path),
    )
    return RunDetail(
        summary=summary_entry,
        metrics=read_summary(job.summary_csv_path),
        samples=read_samples(job.output_csv_path),
        meta=read_summary_meta(job.summary_csv_path),
    )
