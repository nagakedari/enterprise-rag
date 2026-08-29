import { useEffect, useState } from "react";
import { getOptions } from "../api";
import { DEFAULT_PARAMS, type DiversityMode, type EvaluationRunParams, type OptionsResponse } from "../types";

interface Props {
  initialParams?: EvaluationRunParams;
  onCheck: (params: EvaluationRunParams) => void;
  busy: boolean;
}

function numOrNull(raw: string): number | null {
  return raw === "" ? null : Number(raw);
}

function strOrNull(raw: string): string | null {
  return raw.trim() === "" ? null : raw;
}

export default function ParamForm({ initialParams, onCheck, busy }: Props) {
  const [options, setOptions] = useState<OptionsResponse | null>(null);
  const [params, setParams] = useState<EvaluationRunParams>(initialParams ?? DEFAULT_PARAMS);
  const [showAdvanced, setShowAdvanced] = useState(false);

  useEffect(() => {
    getOptions()
      .then((opts) => {
        setOptions(opts);
        // Only backfill defaults for brand-new forms — never overwrite a
        // prefilled "retry with same params" form.
        if (!initialParams) {
          setParams((p) => ({ ...p, samples: opts.default_samples, seed: opts.default_seed }));
        }
      })
      .catch(() => {
        /* form still usable with hardcoded fallbacks below */
      });
  }, [initialParams]);

  const set = <K extends keyof EvaluationRunParams>(key: K, value: EvaluationRunParams[K]) =>
    setParams((p) => ({ ...p, [key]: value }));

  const engines = options?.engines ?? ["custom", "llamaindex"];
  const chunkingStrategies = options?.chunking_strategies ?? ["basic", "parent_child", "semantic"];
  const retrievalModes = options?.retrieval_modes ?? ["semantic", "hybrid"];
  const filterModes = options?.filter_modes ?? ["llm", "regex"];
  const rerankModes = options?.rerank_modes ?? ["llm", "cross_encoder"];
  const diversityModes = options?.diversity_modes ?? ["none", "mmr", "metadata_slots"];
  const allMetrics = options?.metrics ?? [];

  const toggleMetric = (metric: string) => {
    setParams((p) => {
      const current = p.metrics ?? [];
      const next = current.includes(metric)
        ? current.filter((m) => m !== metric)
        : [...current, metric];
      return { ...p, metrics: next.length === 0 ? null : next };
    });
  };

  return (
    <div className="card">
      <h2>Evaluation parameters</h2>

      <div className="field-grid">
        <div className="field">
          <label htmlFor="engine">Engine</label>
          <select id="engine" value={params.engine} onChange={(e) => set("engine", e.target.value as EvaluationRunParams["engine"])}>
            {engines.map((e) => (
              <option key={e} value={e}>
                {e}
              </option>
            ))}
          </select>
        </div>

        <div className="field">
          <label htmlFor="chunking">Chunking strategy</label>
          <select
            id="chunking"
            value={params.chunking_strategy ?? ""}
            onChange={(e) =>
              set("chunking_strategy", (strOrNull(e.target.value) as EvaluationRunParams["chunking_strategy"]) ?? null)
            }
          >
            <option value="">(auto)</option>
            {chunkingStrategies.map((c) => (
              <option key={c} value={c}>
                {c}
              </option>
            ))}
          </select>
        </div>

        <div className="field">
          <label htmlFor="samples">Samples</label>
          <input
            id="samples"
            type="number"
            min={1}
            value={params.samples}
            onChange={(e) => set("samples", Number(e.target.value) || 1)}
          />
        </div>

        <div className="field">
          <label htmlFor="company">Company (optional)</label>
          <input
            id="company"
            placeholder="e.g. AAPL"
            value={params.company ?? ""}
            onChange={(e) => set("company", strOrNull(e.target.value))}
          />
        </div>

        <div className="field">
          <label htmlFor="question_type">Question type (optional)</label>
          <input
            id="question_type"
            placeholder="e.g. Multi-Doc RAG"
            value={params.question_type ?? ""}
            onChange={(e) => set("question_type", strOrNull(e.target.value))}
          />
        </div>
      </div>

      <button type="button" className="advanced-toggle" onClick={() => setShowAdvanced((v) => !v)}>
        {showAdvanced ? "▾ Hide advanced options" : "▸ Show advanced options"}
      </button>

      {showAdvanced && (
        <>
          <div className="field-grid">
            <div className="field">
              <label htmlFor="retrieval_mode">Retrieval mode</label>
              <select
                id="retrieval_mode"
                value={params.retrieval_mode ?? ""}
                onChange={(e) =>
                  set("retrieval_mode", (strOrNull(e.target.value) as EvaluationRunParams["retrieval_mode"]) ?? null)
                }
              >
                <option value="">(auto)</option>
                {retrievalModes.map((m) => (
                  <option key={m} value={m}>
                    {m}
                  </option>
                ))}
              </select>
            </div>

            <div className="field">
              <label htmlFor="alpha">Alpha (0-1, hybrid only)</label>
              <input
                id="alpha"
                type="number"
                step="0.05"
                min={0}
                max={1}
                placeholder="(auto)"
                value={params.alpha ?? ""}
                onChange={(e) => set("alpha", numOrNull(e.target.value))}
              />
            </div>

            <div className="field">
              <label htmlFor="top_k">Top K</label>
              <input
                id="top_k"
                type="number"
                min={1}
                placeholder="(auto)"
                value={params.top_k ?? ""}
                onChange={(e) => set("top_k", numOrNull(e.target.value))}
              />
            </div>

            <div className="field">
              <label htmlFor="filter_mode">Filter mode</label>
              <select id="filter_mode" value={params.filter_mode} onChange={(e) => set("filter_mode", e.target.value as EvaluationRunParams["filter_mode"])}>
                {filterModes.map((m) => (
                  <option key={m} value={m}>
                    {m}
                  </option>
                ))}
              </select>
            </div>

            <div className="field">
              <label htmlFor="rerank_mode">Rerank mode</label>
              <select
                id="rerank_mode"
                value={params.rerank_mode ?? ""}
                onChange={(e) =>
                  set("rerank_mode", (strOrNull(e.target.value) as EvaluationRunParams["rerank_mode"]) ?? null)
                }
              >
                <option value="">(none)</option>
                {rerankModes.map((m) => (
                  <option key={m} value={m}>
                    {m}
                  </option>
                ))}
              </select>
            </div>

            <div className="field">
              <label htmlFor="rerank_top_k">Rerank top K</label>
              <input
                id="rerank_top_k"
                type="number"
                min={1}
                placeholder="(auto)"
                disabled={!params.rerank_mode}
                value={params.rerank_top_k ?? ""}
                onChange={(e) => set("rerank_top_k", numOrNull(e.target.value))}
              />
            </div>

            <div className="field">
              <label htmlFor="diversity_mode">Diversity mode</label>
              <select
                id="diversity_mode"
                disabled={!params.rerank_mode}
                value={params.diversity_mode}
                onChange={(e) => set("diversity_mode", e.target.value as DiversityMode)}
                title={
                  !params.rerank_mode
                    ? "Select a rerank mode first — diversity replaces the final top-k sort inside the reranker"
                    : undefined
                }
              >
                {diversityModes.map((m) => (
                  <option key={m} value={m}>
                    {m === "none" ? "none (default)" : m === "metadata_slots" ? "metadata_slots" : m}
                  </option>
                ))}
              </select>
            </div>

            {params.diversity_mode === "mmr" && (
              <div className="field">
                <label htmlFor="mmr_lambda">MMR lambda (0–1)</label>
                <input
                  id="mmr_lambda"
                  type="number"
                  step="0.05"
                  min={0}
                  max={1}
                  value={params.mmr_lambda}
                  onChange={(e) => set("mmr_lambda", Number(e.target.value) ?? 0.5)}
                  title="1.0 = pure relevance (same as no diversity). 0.0 = pure diversity. 0.5 = balanced (default)."
                />
              </div>
            )}

            {params.diversity_mode === "source_cap" && (
              <div className="field">
                <label htmlFor="max_per_entity">Max chunks per entity (optional)</label>
                <input
                  id="max_per_entity"
                  type="number"
                  min={1}
                  max={20}
                  placeholder="(auto: ceil(top_k / n_entities))"
                  value={params.max_per_entity ?? ""}
                  onChange={(e) => set("max_per_entity", numOrNull(e.target.value))}
                  title="Override the per-(company, quarter, year) slot cap. Default: ceil(top_k / n_entities), minimum 2."
                />
              </div>
            )}

            <div className="field checkbox">
              <input
                id="per_filing"
                type="checkbox"
                checked={params.per_filing}
                onChange={(e) => set("per_filing", e.target.checked)}
              />
              <label htmlFor="per_filing">
                Per-filing retrieval
                <span style={{ fontWeight: "normal", color: "var(--color-muted, #888)", marginLeft: "0.4em" }}>
                  (one query per filing when no quarter filter — improves temporal coverage)
                </span>
              </label>
            </div>

            {params.per_filing && (
              <div className="field">
                <label htmlFor="chunks_per_filing">Chunks per filing</label>
                <input
                  id="chunks_per_filing"
                  type="number"
                  min={1}
                  max={20}
                  value={params.chunks_per_filing}
                  onChange={(e) => set("chunks_per_filing", Number(e.target.value) || 3)}
                  title="How many chunks to retrieve per (year, quarter) filing (default: 3)."
                />
              </div>
            )}

            <div className="field">
              <label htmlFor="seed">Seed</label>
              <input id="seed" type="number" value={params.seed} onChange={(e) => set("seed", Number(e.target.value) || 0)} />
            </div>

            <div className="field">
              <label htmlFor="input_path">Input CSV override (optional)</label>
              <input
                id="input_path"
                placeholder="(use script default)"
                value={params.input_path ?? ""}
                onChange={(e) => set("input_path", strOrNull(e.target.value))}
              />
            </div>

            <div className="field checkbox">
              <input
                id="skip_ragas"
                type="checkbox"
                checked={params.skip_ragas}
                onChange={(e) => set("skip_ragas", e.target.checked)}
              />
              <label htmlFor="skip_ragas">Skip RAGAS metrics (fast/cheap)</label>
            </div>
          </div>

          <div className="section-title">Metrics (empty = all)</div>
          <div className="metrics-checklist">
            {allMetrics.map((m) => (
              <label key={m}>
                <input
                  type="checkbox"
                  checked={(params.metrics ?? []).includes(m)}
                  onChange={() => toggleMetric(m)}
                />
                {m}
              </label>
            ))}
          </div>
        </>
      )}

      <div className="btn-row">
        <button type="button" className="primary" disabled={busy} onClick={() => onCheck(params)}>
          {busy ? "Checking…" : "Check for existing results"}
        </button>
      </div>
    </div>
  );
}
