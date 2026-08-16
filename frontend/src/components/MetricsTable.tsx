import type { MetricRow } from "../types";

function fmt(n: number | null): string {
  return n == null ? "—" : Number.isInteger(n) ? String(n) : n.toFixed(4);
}

export default function MetricsTable({ rows }: { rows: MetricRow[] }) {
  if (rows.length === 0) {
    return <p className="muted">No summary metrics available for this run.</p>;
  }

  return (
    <div className="table-scroll">
      <table>
        <thead>
          <tr>
            <th>Metric</th>
            <th>Mean</th>
            <th>Std</th>
            <th>Min</th>
            <th>Max</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => {
            const isFlaggedRow = r.flagged_count != null;
            return (
              <tr key={r.metric} className={isFlaggedRow ? "flagged-row" : undefined}>
                <td>{r.metric}</td>
                <td>{fmt(r.mean)}</td>
                <td>{fmt(r.std)}</td>
                <td>{fmt(r.min)}</td>
                <td>{fmt(r.max)}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
      {rows
        .filter((r) => r.flagged_count != null)
        .map((r) => (
          <p key={r.metric} className="muted" style={{ marginTop: 4 }}>
            <strong>{r.metric}</strong>: {r.flagged_count}/{r.total} flagged (&gt; {r.threshold}) — {r.note}
          </p>
        ))}
    </div>
  );
}
