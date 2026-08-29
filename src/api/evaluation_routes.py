"""
HTTP endpoints for the evaluation-run web UI.

Wires src/api/evaluation_service.py's core logic into FastAPI routes.
Included into the main app (src/api/main.py) as `evaluation_router`.
"""
import logging

from fastapi import APIRouter

from src.api import evaluation_service as svc
from src.api.models import (
    EvaluationRunParams,
    HistoryRunSummary,
    JobStatus,
    LookupResponse,
    OptionsResponse,
    RunDetail,
    StartRunRequest,
    StartRunResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/evaluation", tags=["evaluation"])


@router.get("/options", response_model=OptionsResponse)
def get_options() -> OptionsResponse:
    return svc.get_options()


@router.get("/history", response_model=list[HistoryRunSummary])
def list_history() -> list[HistoryRunSummary]:
    return svc.scan_history()


@router.get("/history/{run_id}", response_model=RunDetail)
def get_history_detail(run_id: str) -> RunDetail:
    return svc.get_history_detail(run_id)


@router.post("/lookup", response_model=LookupResponse)
def lookup_run(params: EvaluationRunParams) -> LookupResponse:
    exact, close = svc.find_matches(params)
    return LookupResponse(
        requested_run_tag=svc.compute_run_tag(params),
        exact_match=exact,
        close_matches=close,
    )


@router.post("/runs", response_model=StartRunResponse, status_code=202)
async def start_run(request: StartRunRequest) -> StartRunResponse:
    params = EvaluationRunParams(**request.model_dump(exclude={"force"}))
    return await svc.start_run(params, force=request.force)


@router.get("/runs/{job_id}", response_model=JobStatus)
def get_job_status(job_id: str) -> JobStatus:
    return svc.get_job_status(job_id)


@router.post("/runs/{job_id}/cancel", response_model=JobStatus)
async def cancel_job(job_id: str) -> JobStatus:
    return await svc.cancel_job(job_id)


@router.get("/runs/{job_id}/result", response_model=RunDetail)
def get_job_result(job_id: str) -> RunDetail:
    return svc.get_job_result(job_id)
