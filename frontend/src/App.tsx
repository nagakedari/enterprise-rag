import { useState } from "react";
import { ApiError, getHistoryDetail, getJobResult, lookupRun, startRun } from "./api";
import type { EvaluationRunParams, JobStatus, LookupResponse, RunDetail } from "./types";
import ParamForm from "./components/ParamForm";
import LookupResultPanel from "./components/LookupResultPanel";
import JobProgress from "./components/JobProgress";
import ResultsView from "./components/ResultsView";
import Sidebar from "./components/Sidebar";

type View = "form" | "progress" | "results";

export default function App() {
  const [view, setView] = useState<View>("form");
  const [currentParams, setCurrentParams] = useState<EvaluationRunParams | null>(null);
  const [lookupResult, setLookupResult] = useState<LookupResponse | null>(null);
  const [activeJobId, setActiveJobId] = useState<string | null>(null);
  const [resultDetail, setResultDetail] = useState<RunDetail | null>(null);
  const [selectedHistoryId, setSelectedHistoryId] = useState<string | null>(null);
  const [historyRefreshKey, setHistoryRefreshKey] = useState(0);
  const [busy, setBusy] = useState(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  const resetToForm = () => {
    setView("form");
    setLookupResult(null);
    setResultDetail(null);
    setActiveJobId(null);
    setSelectedHistoryId(null);
    setErrorMessage(null);
  };

  const handleCheck = async (params: EvaluationRunParams) => {
    setBusy(true);
    setErrorMessage(null);
    setCurrentParams(params);
    try {
      const result = await lookupRun(params);
      setLookupResult(result);
    } catch (err) {
      setErrorMessage(err instanceof Error ? err.message : "Lookup failed");
    } finally {
      setBusy(false);
    }
  };

  const handleViewHistory = async (id: string) => {
    setErrorMessage(null);
    try {
      const detail = await getHistoryDetail(id);
      setResultDetail(detail);
      setSelectedHistoryId(id);
      setView("results");
    } catch (err) {
      setErrorMessage(err instanceof Error ? err.message : "Failed to load run");
    }
  };

  const handleRun = async (force: boolean) => {
    if (!currentParams) return;
    setBusy(true);
    setErrorMessage(null);
    try {
      const res = await startRun({ ...currentParams, force });
      setActiveJobId(res.job_id);
      setSelectedHistoryId(null);
      setView("progress");
    } catch (err) {
      if (err instanceof ApiError && err.status === 409) {
        // Server-side conflict guard caught a race — refresh the lookup panel
        // with the existing match it reported rather than just erroring out.
        setLookupResult({
          requested_run_tag: lookupResult?.requested_run_tag ?? "",
          exact_match: err.existing,
          close_matches: [],
        });
      } else {
        setErrorMessage(err instanceof Error ? err.message : "Failed to start run");
      }
    } finally {
      setBusy(false);
    }
  };

  const handleJobDone = async (status: JobStatus) => {
    setHistoryRefreshKey((k) => k + 1);
    if (status.status === "succeeded") {
      try {
        const detail = await getJobResult(status.job_id);
        setResultDetail(detail);
        setView("results");
      } catch (err) {
        setErrorMessage(err instanceof Error ? err.message : "Failed to load completed run");
      }
    }
    // failed/cancelled: JobProgress itself shows the error and a "back to form" button
  };

  return (
    <div className="app-shell">
      <Sidebar
        selectedId={selectedHistoryId}
        onSelect={handleViewHistory}
        onNewRun={resetToForm}
        refreshKey={historyRefreshKey}
      />
      <div className="main-panel">
        {errorMessage && (
          <div className="panel-state error">
            <strong>Error</strong>
            <div className="muted" style={{ marginTop: 4 }}>
              {errorMessage}
            </div>
          </div>
        )}

        {view === "form" && (
          <>
            <ParamForm initialParams={currentParams ?? undefined} onCheck={handleCheck} busy={busy} />
            {lookupResult && currentParams && (
              <LookupResultPanel
                lookup={lookupResult}
                params={currentParams}
                busy={busy}
                onView={handleViewHistory}
                onRun={handleRun}
              />
            )}
          </>
        )}

        {view === "progress" && activeJobId && (
          <JobProgress jobId={activeJobId} onDone={handleJobDone} onBackToForm={resetToForm} />
        )}

        {view === "results" && resultDetail && <ResultsView detail={resultDetail} onBack={resetToForm} />}
      </div>
    </div>
  );
}
