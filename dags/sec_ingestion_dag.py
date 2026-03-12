"""
Airflow DAG: sec_10q_ingestion
─────────────────────────────
Triggers the SEC 10-Q document ingestion pipeline.
Schedule: manual (schedule_interval=None) — trigger via the Airflow UI
          or `airflow dags trigger sec_10q_ingestion`.

Task graph:
    validate_docs → parse_and_chunk → embed_and_store → verify_ingestion

Each task is intentionally coarse-grained for Phase 1.
Phase 2 will fan-out parse/chunk per document for parallelism.
"""
import logging
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

from airflow import DAG
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


def parse_and_chunk(**context):
    """
    Parse PDFs and chunk them.
    Pushes chunk count to XCom; actual chunk data is NOT stored in XCom
    (too large) — embed_and_store re-runs this step cheaply.
    """
    from dotenv import load_dotenv
    load_dotenv()

    from src.config import Config
    from src.ingestion.chunker import chunk_documents
    from src.ingestion.pdf_parser import parse_all_pdfs

    config = Config(docs_path=Path(DOCS_PATH))
    documents = parse_all_pdfs(config.docs_path)
    chunks = chunk_documents(documents, config.chunk)

    logger.info("Parsed %d documents → %d chunks", len(documents), len(chunks))
    context["ti"].xcom_push(key="chunk_count", value=len(chunks))
    return len(chunks)


def embed_and_store(**context):
    """
    Full pipeline in one task: parse → chunk → embed → store.
    Re-parsing is cheap (~seconds for 20 PDFs); avoids large XCom payloads.
    """
    from dotenv import load_dotenv
    load_dotenv()

    from src.ingestion.pipeline import run_ingestion_pipeline

    result = run_ingestion_pipeline(docs_path=Path(DOCS_PATH))
    context["ti"].xcom_push(key="pipeline_result", value=result)
    logger.info("Pipeline result: %s", result)
    return result["chunks_stored"]


def verify_ingestion(**context):
    """Confirm that Weaviate contains at least as many docs as were ingested."""
    from dotenv import load_dotenv
    load_dotenv()

    from src.config import Config
    from src.ingestion import weaviate_store

    config = Config()
    client = weaviate_store.get_client(config.weaviate)
    try:
        total = weaviate_store.get_total_count(client, config.weaviate)
    finally:
        client.close()

    expected = context["ti"].xcom_pull(task_ids="embed_and_store")
    logger.info("Verification: %d chunks in Weaviate (expected >= %d)", total, expected or 0)

    if total == 0:
        raise RuntimeError("Weaviate collection is empty after ingestion!")

    return total


# ── DAG definition ────────────────────────────────────────────────────────────

with DAG(
    dag_id="sec_10q_ingestion",
    description="Ingest SEC 10-Q PDF filings into Weaviate vector DB",
    default_args=default_args,
    start_date=datetime(2024, 1, 1),
    schedule_interval=None,   # manual trigger only
    catchup=False,
    tags=["rag", "ingestion", "sec", "phase-1"],
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
