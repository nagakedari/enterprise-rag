// Mirrors src/api/models.py field-for-field. snake_case is preserved
// deliberately — FastAPI returns JSON verbatim, and a camelCase mapping
// layer would be needless complexity for this internal tool.

export type Engine = "custom" | "llamaindex";
export type ChunkingStrategy = "basic" | "parent_child" | "semantic";
export type RetrievalMode = "semantic" | "hybrid";
export type FilterMode = "llm" | "regex";
export type RerankMode = "llm" | "cross_encoder";
export type DiversityMode = "none" | "mmr" | "metadata_slots" | "source_cap";
export type JobState = "queued" | "running" | "succeeded" | "failed" | "cancelled";
export type HistorySource = "root" | "evaluation_results_dir";

// Fields left undefined/null represent an OMITTED CLI flag, not "use the
// default value" — the run-tag algorithm only appends a tag segment when a
// field is set, so "(auto)" form choices MUST serialize as null, never as a
// filled-in default number, or generated run tags drift from what already
// exists on disk.
export interface EvaluationRunParams {
  engine: Engine;
  chunking_strategy: ChunkingStrategy | null;
  use_smart: boolean;
  retrieval_mode: RetrievalMode | null;
  alpha: number | null;
  top_k: number | null;
  filter_mode: FilterMode;
  rerank_mode: RerankMode | null;
  rerank_top_k: number | null;
  diversity_mode: DiversityMode;
  mmr_lambda: number;
  per_filing: boolean;
  chunks_per_filing: number;
  max_per_entity: number | null;
  samples: number;
  company: string | null;
  question_type: string | null;
  metrics: string[] | null;
  skip_ragas: boolean;
  seed: number;
  input_path: string | null;
}

export interface StartRunRequest extends EvaluationRunParams {
  force: boolean;
}

export interface HistoryRunSummary {
  id: string;
  source: HistorySource;
  csv_path: string;
  summary_csv_path: string | null;
  display_name: string;
  run_tag_guess: string | null;
  parsed_params: Record<string, unknown>;
  modified_at: string;
  row_count: number | null;
}

export interface LookupResponse {
  requested_run_tag: string;
  exact_match: HistoryRunSummary | null;
  close_matches: HistoryRunSummary[];
}

export interface MetricRow {
  metric: string;
  mean: number | null;
  std: number | null;
  min: number | null;
  max: number | null;
  avg_pct: number | null;
  flagged_count: number | null;
  total: number | null;
  flagged_pct: number | null;
  threshold: number | null;
  note: string | null;
}

export interface SampleRow {
  question: string;
  question_type: string | null;
  source_chunk_type: string | null;
  golden_answer: string;
  generated_answer: string;
  exactness: number | null;
  answer_similarity: number | null;
  correctness: number | null;
  context_precision: number | null;
  context_recall: number | null;
  context_relevance: number | null;
  faithfulness: number | null;
  hallucination_rate: number | null;
  answer_relevance: number | null;
  factual_error_rate: number | null;
}

export interface RunDetail {
  summary: HistoryRunSummary;
  metrics: MetricRow[];
  samples: SampleRow[];
  meta: Record<string, string>;
}

export interface OptionsResponse {
  engines: string[];
  chunking_strategies: string[];
  retrieval_modes: string[];
  filter_modes: string[];
  rerank_modes: string[];
  diversity_modes: string[];
  metrics: string[];
  default_top_k: number;
  default_samples: number;
  default_seed: number;
}

export interface JobStatus {
  job_id: string;
  status: JobState;
  run_tag: string;
  params: EvaluationRunParams;
  started_at: string | null;
  finished_at: string | null;
  progress_current: number | null;
  progress_total: number | null;
  log_tail: string[];
  exit_code: number | null;
  error_message: string | null;
  output_csv_path: string | null;
  summary_csv_path: string | null;
}

export interface StartRunResponse {
  job_id: string;
  run_tag: string;
  status: string;
}

export const DEFAULT_PARAMS: EvaluationRunParams = {
  engine: "custom",
  chunking_strategy: null,
  use_smart: false,
  retrieval_mode: null,
  alpha: null,
  top_k: null,
  filter_mode: "llm",
  rerank_mode: null,
  rerank_top_k: null,
  diversity_mode: "none",
  mmr_lambda: 0.5,
  per_filing: false,
  chunks_per_filing: 3,
  max_per_entity: null,
  samples: 25,
  company: null,
  question_type: null,
  metrics: null,
  skip_ragas: false,
  seed: 42,
  input_path: null,
};
