"""
Airflow DAG: sec_10q_ingestion
─────────────────────────────
Triggers the SEC 10-Q document ingestion pipeline.
Schedule: manual (schedule_interval=None) — trigger via the Airflow UI
          or `airflow dags trigger sec_10q_ingestion`.

Task graph:
    validate_docs → parse_and_chunk → embed_and_store → verify_ingestion

Airflow Variables (set in Admin → Variables):
    ingestion_engine               "custom" | "llamaindex"  (default: "custom")
        Selects the ingestion engine.
        custom     → direct Weaviate client pipeline
        llamaindex → LlamaIndex document parser + vector store

    ingestion_use_smart_chunking   "true" | "false"  (default: "false")
        Controls which chunking strategy and Weaviate collection are used.
        false → Phase-1 basic token chunker  → SecDocument / SecDocumentLI
        true  → Phase-2 parent-child chunker → SecDocumentSmart / SecDocumentSmartLI

    ingestion_recreate_collection  "true" | "false"  (default: "false")
        Drop and recreate the target collection before ingestion.
        Set to "true" only when you want a clean re-ingest.
"""
import logging
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

from airflow import DAG
from airflow.models import Variable
from airflow.operators.python import PythonOperator

# Make the repo's src/ package importable inside Airflow workers.
# In production you'd install it as a proper package instead.
sys.path.insert(0, "/opt/airflow")

logger = logging.getLogger(__name__)

DOCS_PATH = os.getenv(
    "DOCS_PATH",
    "/Users/manjusri/learning/generative_ai/KG-RAG-datasets/sec-10-q/data/v1/docs",
)

default_args = {
    "owner": "airflow",
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
    "email_on_failure": False,
}


# ── Task callables ────────────────────────────────────────────────────────────

def validate_docs(**context):
    """Fail fast if no PDFs are found in the docs directory."""
    docs_path = Path(DOCS_PATH)
    if not docs_path.exists():
        raise FileNotFoundError(f"DOCS_PATH does not exist: {docs_path}")

    pdf_files = sorted(docs_path.glob("*.pdf"))
    if not pdf_files:
        raise FileNotFoundError(f"No PDF files found in {docs_path}")

    filenames = [f.name for f in pdf_files]
    logger.info("Found %d PDF files: %s", len(filenames), filenames)
    context["ti"].xcom_push(key="pdf_count", value=len(filenames))
    return len(filenames)


def _get_use_smart() -> bool:
    """
    Read the ingestion_use_smart_chunking Airflow Variable.
    Always call this inside a task callable — never at module level —
    so the scheduler does not query the metadata DB on every DAG parse.
    """
    return Variable.get("ingestion_use_smart_chunking", default_var="false").lower() == "true"


def _get_recreate_collection() -> bool:
    return Variable.get("ingestion_recreate_collection", default_var="false").lower() == "true"


def _get_engine() -> str:
    """
    Read the ingestion_engine Airflow Variable.
    Allowed values: "custom" | "llamaindex"  (default: "custom")
    """
    return Variable.get("ingestion_engine", default_var="custom").lower()


def parse_and_chunk(**context):
    """
    Parse PDFs and chunk them (mode determined by ingestion_use_smart_chunking).
    Pushes chunk count to XCom; actual chunk data is NOT stored in XCom
    (too large) — embed_and_store re-runs the full pipeline cheaply.
    """
    from dotenv import load_dotenv
    load_dotenv()

    from src.config import Config
    from src.ingestion.chunker import chunk_documents
    from src.ingestion.smart_chunker import chunk_documents_smart
    from src.ingestion.pdf_parser import parse_all_pdfs

    use_smart = _get_use_smart()
    config = Config(docs_path=Path(DOCS_PATH))
    documents = parse_all_pdfs(config.docs_path)

    if use_smart:
        chunks = chunk_documents_smart(documents, config.smart_chunk)
    else:
        chunks = chunk_documents(documents, config.chunk)

    mode = "smart" if use_smart else "basic"
    logger.info("[mode=%s] Parsed %d documents → %d chunks", mode, len(documents), len(chunks))
    context["ti"].xcom_push(key="chunk_count", value=len(chunks))
    context["ti"].xcom_push(key="use_smart", value=use_smart)
    return len(chunks)


def embed_and_store(**context):
    """
    Full pipeline: parse → chunk → embed → store.
    Mode (basic vs smart) is read from the ingestion_use_smart_chunking Variable.
    Re-parsing is cheap; avoids large XCom payloads.
    """
    from dotenv import load_dotenv
    load_dotenv()

    from src.ingestion.pipeline import run_ingestion_pipeline

    use_smart = _get_use_smart()
    recreate_collection = _get_recreate_collection()
    engine = _get_engine()

    logger.info(
        "Starting ingestion  engine=%s  use_smart=%s  recreate_collection=%s",
        engine, use_smart, recreate_collection,
    )

    result = run_ingestion_pipeline(
        docs_path=Path(DOCS_PATH),
        engine=engine,
        use_smart=use_smart,
        recreate_collection=recreate_collection,
    )
    context["ti"].xcom_push(key="pipeline_result", value=result)
    logger.info("Pipeline result: %s", result)
    # custom pipeline  → "chunks_stored"
    # llamaindex pipeline → "nodes_indexed"
    stored = result.get("chunks_stored") or result.get("nodes_indexed", 0)
    return stored


def verify_ingestion(**context):
    """
    Confirm Weaviate contains at least as many chunks as were ingested.
    Queries SecDocumentSmart when use_smart=True, SecDocument otherwise.
    """
    from dotenv import load_dotenv
    load_dotenv()

    from src.config import Config
    from src.ingestion import weaviate_store

    use_smart = _get_use_smart()
    engine = _get_engine()
    config = Config()
    client = weaviate_store.get_client(config.weaviate)
    try:
        if engine == "llamaindex":
            collection = (
                config.weaviate.llamaindex_smart_collection_name
                if use_smart
                else config.weaviate.llamaindex_collection_name
            )
            col = client.collections.get(collection)
            total = col.aggregate.over_all(total_count=True).total_count or 0
        elif use_smart:
            total = weaviate_store.get_total_count_smart(client, config.weaviate)
            collection = config.weaviate.smart_collection_name
        else:
            total = weaviate_store.get_total_count(client, config.weaviate)
            collection = config.weaviate.collection_name
    finally:
        client.close()

    expected = context["ti"].xcom_pull(task_ids="embed_and_store")
    logger.info(
        "Verification [%s]: %d chunks in Weaviate (expected >= %d)",
        collection, total, expected or 0,
    )

    if total == 0:
        raise RuntimeError(f"Weaviate collection '{collection}' is empty after ingestion!")

    return total


# ── DAG definition ────────────────────────────────────────────────────────────

with DAG(
    dag_id="sec_10q_ingestion",
    description="Ingest SEC 10-Q PDF filings into Weaviate vector DB",
    default_args=default_args,
    start_date=datetime(2024, 1, 1),
    schedule_interval=None,   # manual trigger only
    catchup=False,
    tags=["rag", "ingestion", "sec"],
    doc_md=__doc__,
) as dag:

    t_validate = PythonOperator(
        task_id="validate_docs",
        python_callable=validate_docs,
        doc_md="Check that PDF files exist in DOCS_PATH before starting.",
    )

    t_parse = PythonOperator(
        task_id="parse_and_chunk",
        python_callable=parse_and_chunk,
        doc_md="Parse PDFs and chunk into token windows. Result pushed to XCom.",
    )

    t_ingest = PythonOperator(
        task_id="embed_and_store",
        python_callable=embed_and_store,
        doc_md="Embed chunks with OpenAI and store in Weaviate.",
    )

    t_verify = PythonOperator(
        task_id="verify_ingestion",
        python_callable=verify_ingestion,
        doc_md="Confirm Weaviate contains the expected number of chunks.",
    )

    t_validate >> t_parse >> t_ingest >> t_verify
