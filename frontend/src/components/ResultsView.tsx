import { useState } from "react";
import type { RunDetail } from "../types";
import MetricsTable from "./MetricsTable";
import SampleTable from "./SampleTable";

export default function ResultsView({ detail, onBack }: { detail: RunDetail; onBack: () => void }) {
  const [showSamples, setShowSamples] = useState(false);
  const { summary } = detail;

  return (
    <div>
      <div className="card">
        <h2>
          <span className="name">{summary.display_name}</span>
        </h2>
        <div className="muted">
          {summary.source === "root" ? "repo root" : "evaluation_results/"} — {new Date(summary.modified_at).toLocaleString()}
          {summary.row_count != null && ` — ${summary.row_count} samples`}
        </div>
        {detail.meta.input_file && (
          <div className="muted" style={{ marginTop: 4 }}>
            Input dataset: <span className="name">{detail.meta.input_file}</span>
          </div>
        )}
        <div className="btn-row">
          <button onClick={onBack}>← Back</button>
        </div>
      </div>

      <div className="card">
        <h2>Summary metrics</h2>
        <MetricsTable rows={detail.metrics} />
      </div>

      <div className="card">
        <button className="advanced-toggle" onClick={() => setShowSamples((v) => !v)}>
          {showSamples ? "▾ Hide per-sample results" : `▸ Show per-sample results (${detail.samples.length})`}
        </button>
        {showSamples && <SampleTable rows={detail.samples} />}
      </div>
    </div>
  );
}
