import { useEffect, useState } from "react";
import { listHistory } from "../api";
import type { HistoryRunSummary } from "../types";

interface Props {
  selectedId: string | null;
  onSelect: (id: string) => void;
  onNewRun: () => void;
  refreshKey: number;
}

export default function Sidebar({ selectedId, onSelect, onNewRun, refreshKey }: Props) {
  const [history, setHistory] = useState<HistoryRunSummary[]>([]);
  const [filterText, setFilterText] = useState("");

  useEffect(() => {
    listHistory()
      .then(setHistory)
      .catch(() => setHistory([]));
  }, [refreshKey]);

  const filtered = history.filter((h) =>
    h.display_name.toLowerCase().includes(filterText.toLowerCase()),
  );

  return (
    <div className="sidebar">
      <div className="sidebar-header">
        <h1>RAG Evaluation Runs</h1>
        <input
          className="sidebar-filter"
          placeholder="Filter past runs…"
          value={filterText}
          onChange={(e) => setFilterText(e.target.value)}
        />
      </div>
      <button className="primary sidebar-new-btn" onClick={onNewRun}>
        + New evaluation
      </button>
      <div className="sidebar-list">
        {filtered.length === 0 && <p className="muted" style={{ padding: "0 14px" }}>No past runs found.</p>}
        {filtered.map((h) => (
          <div
            key={h.id}
            className={`sidebar-item${h.id === selectedId ? " selected" : ""}`}
            onClick={() => onSelect(h.id)}
          >
            <div className="name">{h.display_name}</div>
            <div className="meta">
              {h.run_tag_guess ? "parsed" : "legacy / unparsed"} — {h.row_count ?? "?"} rows —{" "}
              {new Date(h.modified_at).toLocaleDateString()}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
