import type { EvaluationRunParams, HistoryRunSummary, LookupResponse } from "../types";

interface Props {
  lookup: LookupResponse;
  params: EvaluationRunParams;
  busy: boolean;
  onView: (id: string) => void;
  onRun: (force: boolean) => void;
}

const DIFF_FIELDS: Array<[keyof EvaluationRunParams, string]> = [
  ["engine", "engine"],
  ["chunking_strategy", "chunking_strategy"],
  ["retrieval_mode", "retrieval_mode"],
  ["filter_mode", "filter_mode"],
  ["rerank_mode", "rerank_mode"],
  ["alpha", "alpha"],
  ["top_k", "top_k"],
  ["rerank_top_k", "rerank_top_k"],
];

function diffChips(entry: HistoryRunSummary, params: EvaluationRunParams) {
  return DIFF_FIELDS.filter(([, key]) => {
    const requested = params[key as keyof EvaluationRunParams];
    const parsed = entry.parsed_params[key];
    return parsed !== undefined && parsed !== null && String(parsed) !== String(requested ?? "");
  }).map(([, key]) => (
    <span key={key} className="diff-chip">
      {key}: {String(entry.parsed_params[key])}
    </span>
  ));
}

export default function LookupResultPanel({ lookup, params, busy, onView, onRun }: Props) {
  if (lookup.exact_match) {
    const m = lookup.exact_match;
    return (
      <div className="panel-state exact">
        <strong>Found an exact match</strong>
        <div className="muted" style={{ margin: "6px 0" }}>
          <span className="name">{m.display_name}</span> — run on {new Date(m.modified_at).toLocaleString()}
          {m.row_count != null && ` — ${m.row_count} samples`}
        </div>
        <div className="btn-row">
          <button className="primary" onClick={() => onView(m.id)}>
            View results
          </button>
          <button className="danger" disabled={busy} onClick={() => onRun(true)}>
            {busy ? "Starting…" : "Force re-run anyway"}
          </button>
        </div>
      </div>
    );
  }

  if (lookup.close_matches.length > 0) {
    return (
      <div className="panel-state close">
        <strong>No exact match, but {lookup.close_matches.length} similar past run(s) found</strong>
        <div style={{ marginTop: 8 }}>
          {lookup.close_matches.map((m) => (
            <div key={m.id} className="close-match-row">
              <div>
                <div className="name">{m.display_name}</div>
                <div style={{ marginTop: 4 }}>{diffChips(m, params)}</div>
              </div>
              <button onClick={() => onView(m.id)}>View</button>
            </div>
          ))}
        </div>
        <div className="btn-row">
          <button className="primary" disabled={busy} onClick={() => onRun(false)}>
            {busy ? "Starting…" : "Run this exact configuration"}
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="panel-state none">
      <strong>No past results for this configuration</strong>
      <div className="muted" style={{ margin: "6px 0" }}>
        Requested run tag: <span className="name">{lookup.requested_run_tag}</span>
      </div>
      <div className="btn-row">
        <button className="primary" disabled={busy} onClick={() => onRun(false)}>
          {busy ? "Starting…" : "Run now"}
        </button>
      </div>
    </div>
  );
}
