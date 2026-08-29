import type {
  EvaluationRunParams,
  HistoryRunSummary,
  JobStatus,
  LookupResponse,
  OptionsResponse,
  RunDetail,
  StartRunRequest,
  StartRunResponse,
} from "./types";

const BASE = import.meta.env.VITE_API_BASE_URL;

export class ApiError extends Error {
  status: number;
  detail: unknown;
  existing: HistoryRunSummary | null;

  constructor(status: number, detail: unknown) {
    const message =
      typeof detail === "object" && detail !== null && "message" in detail
        ? String((detail as { message: unknown }).message)
        : typeof detail === "string"
          ? detail
          : `Request failed with status ${status}`;
    super(message);
    this.status = status;
    this.detail = detail;
    this.existing =
      typeof detail === "object" && detail !== null && "existing" in detail
        ? ((detail as { existing: HistoryRunSummary }).existing ?? null)
        : null;
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    ...init,
    headers: { "Content-Type": "application/json", ...(init?.headers ?? {}) },
  });
  if (!res.ok) {
    let detail: unknown = null;
    try {
      const body = await res.json();
      detail = body?.detail ?? body;
    } catch {
      detail = await res.text();
    }
    throw new ApiError(res.status, detail);
  }
  return res.json() as Promise<T>;
}

export function getOptions(): Promise<OptionsResponse> {
  return request("/api/evaluation/options");
}

export function listHistory(): Promise<HistoryRunSummary[]> {
  return request("/api/evaluation/history");
}

export function getHistoryDetail(id: string): Promise<RunDetail> {
  return request(`/api/evaluation/history/${encodeURIComponent(id)}`);
}

export function lookupRun(params: EvaluationRunParams): Promise<LookupResponse> {
  return request("/api/evaluation/lookup", {
    method: "POST",
    body: JSON.stringify(params),
  });
}

export function startRun(req: StartRunRequest): Promise<StartRunResponse> {
  return request("/api/evaluation/runs", {
    method: "POST",
    body: JSON.stringify(req),
  });
}

export function getJobStatus(jobId: string): Promise<JobStatus> {
  return request(`/api/evaluation/runs/${encodeURIComponent(jobId)}`);
}

export function getJobResult(jobId: string): Promise<RunDetail> {
  return request(`/api/evaluation/runs/${encodeURIComponent(jobId)}/result`);
}

export function cancelJob(jobId: string): Promise<JobStatus> {
  return request(`/api/evaluation/runs/${encodeURIComponent(jobId)}/cancel`, {
    method: "POST",
  });
}
