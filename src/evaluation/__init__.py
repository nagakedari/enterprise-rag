"""RAG evaluation module.

Metrics computed:
  Golden-answer metrics (require reference):
    - Exactness           – token F1 overlap, no LLM required
    - AnswerSimilarity    – embedding cosine similarity, no LLM required
    - Correctness         – RAGAS AnswerCorrectness (LLM + semantic)

  RAG-quality metrics (no reference needed, GEval-style LLM scoring):
    - Faithfulness        – RAGAS Faithfulness
    - ContextRelevance    – custom GEval via OpenAI
    - AnswerRelevance     – RAGAS AnswerRelevancy
"""
