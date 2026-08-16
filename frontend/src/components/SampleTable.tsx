import { useState } from "react";
import type { SampleRow } from "../types";

const COLUMNS: Array<{ key: keyof SampleRow; label: string; wrap?: boolean }> = [
  { key: "question", label: "Question", wrap: true },
  { key: "question_type", label: "Type" },
  { key: "source_chunk_type", label: "Chunk type" },
  { key: "golden_answer", label: "Golden answer", wrap: true },
  { key: "generated_answer", label: "Generated answer", wrap: true },
  { key: "exactness", label: "Exactness" },
  { key: "answer_similarity", label: "Ans. sim." },
  { key: "correctness", label: "Correctness" },
  { key: "context_precision", label: "Ctx precision" },
  { key: "context_recall", label: "Ctx recall" },
  { key: "context_relevance", label: "Ctx relevance" },
  { key: "faithfulness", label: "Faithfulness" },
  { key: "hallucination_rate", label: "Hallucination" },
  { key: "answer_relevance", label: "Ans. relevance" },
  { key: "factual_error_rate", label: "Factual error" },
];

const PAGE_SIZE = 25;

function fmt(v: unknown): string {
  if (v == null) return "—";
  if (typeof v === "number") return Number.isInteger(v) ? String(v) : v.toFixed(4);
  return String(v);
}

export default function SampleTable({ rows }: { rows: SampleRow[] }) {
  const [showAll, setShowAll] = useState(false);

  if (rows.length === 0) {
    return <p className="muted">No per-sample rows available for this run.</p>;
  }

  const visibleColumns = COLUMNS.filter((col) => rows.some((r) => r[col.key] != null && r[col.key] !== ""));
  const visibleRows = showAll ? rows : rows.slice(0, PAGE_SIZE);

  return (
    <div>
      <div className="table-scroll">
        <table>
          <thead>
            <tr>
              {visibleColumns.map((c) => (
                <th key={c.key}>{c.label}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {visibleRows.map((r, i) => (
              <tr key={i}>
                {visibleColumns.map((c) => (
                  <td key={c.key} className={c.wrap ? "wrap" : undefined}>
                    {fmt(r[c.key])}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {rows.length > PAGE_SIZE && (
        <button style={{ marginTop: 10 }} onClick={() => setShowAll((v) => !v)}>
          {showAll ? `Show first ${PAGE_SIZE}` : `Show all ${rows.length}`}
        </button>
      )}
    </div>
  );
}
