import { useEffect, useRef, useState } from "react";
import { cancelJob, getJobStatus } from "../api";
import type { JobStatus } from "../types";

interface Props {
  jobId: string;
  onDone: (status: JobStatus) => void;
  onBackToForm: () => void;
}

const POLL_MS = 1500;
const TERMINAL: JobStatus["status"][] = ["succeeded", "failed", "cancelled"];

export default function JobProgress({ jobId, onDone, onBackToForm }: Props) {
  const [status, setStatus] = useState<JobStatus | null>(null);
  const logRef = useRef<HTMLDivElement>(null);
  const doneRef = useRef(false);

  useEffect(() => {
    doneRef.current = false;
    let cancelled = false;
    let timer: ReturnType<typeof setTimeout>;

    const poll = async () => {
      try {
        const s = await getJobStatus(jobId);
        if (cancelled) return;
        setStatus(s);
        if (TERMINAL.includes(s.status)) {
          if (!doneRef.current) {
            doneRef.current = true;
            onDone(s);
          }
          return; // stop polling
        }
      } catch {
        // transient fetch error — keep polling
      }
      timer = setTimeout(poll, POLL_MS);
    };

    poll();
    return () => {
      cancelled = true;
      clearTimeout(timer);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [jobId]);

  useEffect(() => {
    logRef.current?.scrollTo({ top: logRef.current.scrollHeight });
  }, [status?.log_tail]);

  if (!status) {
    return (
      <div className="card">
        <span className="muted">Connecting to job…</span>
      </div>
    );
  }

  const pct =
    status.progress_current != null && status.progress_total
      ? Math.round((status.progress_current / status.progress_total) * 100)
      : 0;
  const isTerminal = TERMINAL.includes(status.status);

  return (
    <div className="card">
      <h2>
        Run <span className="name">{status.run_tag}</span>{" "}
        <span className={`badge ${status.status}`}>{status.status}</span>
      </h2>

      <div className="muted">
        {status.progress_current != null && status.progress_total
          ? `Question ${status.progress_current} / ${status.progress_total}`
          : "Waiting for progress…"}
      </div>
      <div className="progress-bar-track">
        <div className="progress-bar-fill" style={{ width: `${pct}%` }} />
      </div>

      {status.error_message && (
        <div className="panel-state error">
          <strong>Error</strong>
          <div className="muted" style={{ marginTop: 4 }}>
            {status.error_message}
          </div>
        </div>
      )}

      <div className="section-title">Log</div>
      <div className="log-tail" ref={logRef}>
        {status.log_tail.length > 0 ? status.log_tail.join("\n") : "(no output yet)"}
      </div>

      <div className="btn-row">
        {!isTerminal && (
          <button
            className="danger"
            onClick={() => {
              cancelJob(jobId).then(setStatus);
            }}
          >
            Cancel
          </button>
        )}
        {isTerminal && status.status !== "succeeded" && (
          <button className="primary" onClick={onBackToForm}>
            Back to form
          </button>
        )}
      </div>
    </div>
  );
}
