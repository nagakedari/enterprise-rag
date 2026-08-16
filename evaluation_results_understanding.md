# Problem 1 — You are retrieving the wrong chunks 71% of the time (context_precision = 0.29)
Precision of 0.29 means for every 5 parent chunks you return, only 1-2 are actually relevant to the question. The other 3-4 are noise the LLM has to wade through.

## Why: 
The parent chunks are ~1000 tokens. They contain a lot of boilerplate — legal disclaimers, forward-looking statement warnings, table formatting. When BM25 scores these chunks, shared boilerplate vocabulary inflates their scores. A question about "operating margin" will pull in chunks that mention "margin" in a legal disclaimer context, not a financial one.

The high std (0.31) reveals the real shape: it's not consistently mediocre — it's binary. Some questions hit exactly the right chunk (precision = 1.0) while most questions retrieve completely wrong material (precision = 0.0). The min=0.0 and max=1.0 confirm this. The system has no middle ground — it either gets lucky or completely misses.

# Problem 2 — Missing 45% of the information needed to answer (context_recall = 0.55, std = 0.45)
SEC 10-Q questions often require information from multiple sections. A question like "What were the main risks Apple disclosed that affected revenue?" needs content from Item 1A (Risk Factors), Item 2 (MD&A), and possibly a footnote. With top_k=5, you can only retrieve 5 parent chunks. If the answer is spread across 3 different sections, you have a structural ceiling on recall.

The std of 0.45 again shows binary behavior — some questions are single-section and you nail them (recall = 1.0), others require cross-section synthesis and you get recall = 0.0.

The child embedding is also hurting you here: You embed a 300-token child to find the relevant parent. But a child chunk from the middle of a section might not contain the keywords the question uses — those keywords might only appear in the section header (which is in a different child). So the right parent never surfaces.

# Problem 3 — The LLM doesn't fully answer the question (answer_relevance = 0.41)
Answer relevance of 0.41 means the LLM's response addresses less than half of what the question asked. This is the generator's reaction to noisy context. When you pass 5 parent chunks containing mixed relevant and irrelevant material, the LLM doesn't know which parts to focus on. It hedges, picks the easiest aspects to address, and ignores the harder parts that would require synthesizing across chunks.
Fix 1 — Pass filters in evaluation (will reduce factual_error_rate significan
This is different from hallucination — the LLM is not making things up, it is just giving an incomplete, partially-addressed answer.

# Problem 4 — 42% factual errors even when context is retrieved (factual_error_rate = 0.42)
This is the most alarming number. Even with faithfulness of 0.70 (the LLM is grounding answers in context), 42% of the factual claims in those answers are wrong.

Why: A 1000-token parent chunk from a financial filing contains many numbers — quarterly revenue, YoY comparisons, segment breakdowns, per-share figures — often in adjacent sentences. When the LLM is told "use this context", it reads all those numbers and frequently attributes the wrong figure to the wrong line item. For example, it might say operating income was $X when $X was actually net income, because both appeared in the same parent paragraph.

The retrieval-from-wrong-period problem compounds this: The evaluation in run_evaluation.py doesn't pass company, year, or quarter filters to the retriever. So when evaluating a question about Apple Q2 2023, the retriever might surface Apple Q1 2023 or Q3 2022 parent chunks — they are semantically very similar (same company, same document structure, similar language) but have completely different numbers. The LLM faithfully uses those numbers and they are factually wrong relative to the correct quarter.

# Problem 5 — 30% hallucination on top of all the above (faithfulness = 0.70, hallucination = 0.30)
Faithfulness measures grounding — are the answer's claims traceable to something in the retrieved context? 0.70 means 30% of claims aren't. This happens when recall misses a piece of information the question requires. The LLM fills the gap using its OpenAI training data about Apple, Microsoft, and other SEC filers — data it absorbed during pretraining. Those answers feel confident and financial-sounding but are not from the documents you ingested.

What Needs to Be Fixed
# tly)

In run_evaluation.py, the run_rag() call doesn't pass company, year, or quarter. The CSV has a Source Docs column (e.g., *AAPL 2023 Q2*). Parse it and pass those values:


# In run_rag() call — parse source doc metadata from the row
source = str(row.get("Source Docs", ""))
company, year, quarter = parse_source_doc(source)  # extract AAPL, 2023, Q2

chunks = retrieve(
    query=question,
    config=config,
    company=company,
    year=year,
    quarter=quarter,
    ...
)
This alone will likely cut factual_error_rate significantly because you stop retrieving from the wrong filing period.

# Fix 2 — Add a re-ranker after initial retrieval (will improve context_precision)

After fetching top_k * 3 candidates, score them with a cross-encoder or a cheap LLM call asking "is this passage relevant to this question?" and keep only the top 5. This reduces the noise from 71% to much less and gives the LLM cleaner context to work with.

# Fix 3 — Increase top_k for multi-section questions (will improve context_recall)

Change top_k from 5 to 8-10 in the config or evaluation CLI. The marginal cost of extra retrieval is low and recall directly improves answer correctness.

# Fix 4 — Tune hybrid alpha toward BM25 for financial figures

Financial questions often contain specific numbers, ticker symbols, and SEC-specific terms (gross margin, diluted EPS, Item 7). These exact tokens matter more than semantic similarity — a question about "EPS" should prioritize chunks containing the exact string "EPS" or "earnings per share." Lower alpha from 0.5 to 0.3 (more BM25 weight) and re-evaluate. You'll likely see context_precision improve because BM25 is more discriminative on specific financial vocabulary.

# Fix 5 — Split parent chunks by section boundary, not by token count

The RecursiveCharacterTextSplitter splits at SEC-aware separators (\n\nITEM , \n\nPART ) at the parent level but falls back to character splitting when those aren't found. Many parent chunks straddle two sub-sections, mixing numbers from different topics. If a parent chunk contains both revenue figures and risk factor language, it will match many different questions but answer none of them well — producing the precision = 0.0 or precision = 1.0 bimodal distribution you're seeing.


# Fix 4 — Tune Hybrid Alpha for Financial Figures
What alpha controls

In Weaviate hybrid search, alpha is a weight between BM25 and vector similarity:

alpha=0.0 → pure BM25 (keyword matching only)
alpha=1.0 → pure vector (semantic only)
alpha=0.5 → equal weight (current default)
Why SEC financials skew toward BM25

SEC 10-Q questions are heavily token-specific:

Question pattern	Why BM25 wins
"What was net revenue in Q2?"	"net revenue", "Q2" are exact tokens in the filing
"EPS for fiscal 2023?"	"EPS", "earnings per share", "2023" — exact matches
"Item 7A interest rate risk"	"Item 7A" is a structural label — semantically opaque
"$2.3 billion operating loss"	Numeric literals are not encoded meaningfully by embeddings
"AAPL Q3 2023 gross margin"	Ticker + quarter + year = 3 exact tokens the vector can't handle better than BM25
Vector search helps when phrasing varies ("revenue" ↔ "top line" ↔ "net sales") but those are the minority of financial questions. The majority ask about specific numbers in specific periods.

Recommended alpha to try: 0.25

That gives BM25 75% weight while retaining enough semantic signal to handle paraphrasing. The current 0.5 under-weights BM25 for this domain.

Running the A/B evaluation

The --alpha flag is already wired in. Run both sides back-to-back:


# Baseline: current default alpha (0.5)
python scripts/run_evaluation.py \
  --samples 25 --engine custom --use-smart \
  --retrieval-mode hybrid --alpha 0.5 \
  --filter-mode llm --rerank-mode cross_encoder

# BM25-heavy: alpha 0.25
python scripts/run_evaluation.py \
  --samples 25 --engine custom --use-smart \
  --retrieval-mode hybrid --alpha 0.25 \
  --filter-mode llm --rerank-mode cross_encoder

# Even more BM25: alpha 0.1
python scripts/run_evaluation.py \
  --samples 25 --engine custom --use-smart \
  --retrieval-mode hybrid --alpha 0.1 \
  --filter-mode llm --rerank-mode cross_encoder
The output files will be named evaluation_results_custom_smart_hybrid_a0.25_llmfilters_cross_encoderrerank_summary.csv etc. so you can compare them directly.

What to look for in results

context_relevance — primary indicator; higher = BM25 finding more on-topic chunks
faithfulness — should stay stable or improve (less noise in context)
exactness — token F1 against golden answers; often goes up with BM25 on numeric questions
answer_similarity — semantic similarity; watch that this doesn't degrade (means vector signal was needed)
The tradeoff to watch

If answer_similarity drops while context_relevance rises, it means some questions genuinely need semantic matching (e.g., "how did operational efficiency improve?" — no exact numeric token to match). In that case 0.25 may be too aggressive and 0.35 is a better landing point.

Do you want to run these evaluations, or move to Fix 5 (section-boundary chunk splitting) first?

# Fix 5

Good — only langchain_text_splitters remains (used for the parent/child splitters), and langchain_core.documents.Document is gone since we no longer need the per-page LangChain Document wrapper.

Here's a summary of what changed and why:

What changed in smart_chunker.py

Old	New
RecursiveCharacterTextSplitter with SEC headers as separators (hit token limit first)	Regex split at section boundaries → then RecursiveCharacterTextSplitter with generic separators within each section
section_title detected retrospectively inside parent text (approximate)	section_title is the actual SEC header we split on (exact)
Adjacent short sections could merge into one parent	Each section is its own scope — no merging across ITEM/PART/NOTE boundaries
Long section (3000 tokens) split mid-paragraph across arbitrary parents	Long section split into multiple parents, all tagged with the same section_title
One LangChain Document per page → fed to parent splitter	Full text built from all pages → section split → parent split within section
page_num from LangChain Document metadata	page_num from binary-search offset → page boundary index
What did NOT change — the SmartTextChunk dataclass, the child splitter parameters, context_window format, everything downstream (embedder, weaviate_store, retriever, API, evaluation). Re-run ingestion with --engine custom --use-smart to rebuild the collection with section-scoped parents.

