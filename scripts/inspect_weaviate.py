#!/usr/bin/env python3
"""
Inspect chunks and vectors stored in Weaviate.

Usage:
    # Show first 5 chunks (no vectors)
    python scripts/inspect_weaviate.py

    # Show first 5 chunks WITH their vectors
    python scripts/inspect_weaviate.py --vectors

    # Show chunks for a specific company
    python scripts/inspect_weaviate.py --company AAPL

    # Show N chunks
    python scripts/inspect_weaviate.py --limit 20

    # Total count only
    python scripts/inspect_weaviate.py --count
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit",   type=int, default=5,    help="Number of chunks to show")
    parser.add_argument("--company", type=str, default=None, help="Filter by company ticker e.g. AAPL")
    parser.add_argument("--vectors", action="store_true",    help="Print full embedding vectors")
    parser.add_argument("--count",   action="store_true",    help="Print total count only")
    args = parser.parse_args()

    from src.config import Config
    import weaviate
    from weaviate.classes.query import MetadataQuery, Filter

    config = Config()
    client = weaviate.connect_to_custom(
        http_host=config.weaviate.http_host,
        http_port=config.weaviate.http_port,
        http_secure=False,
        grpc_host=config.weaviate.http_host,
        grpc_port=config.weaviate.grpc_port,
        grpc_secure=False,
    )

    try:
        collection = client.collections.get(config.weaviate.collection_name)

        # ── Total count ───────────────────────────────────────────────────────
        total = collection.aggregate.over_all(total_count=True).total_count
        print(f"\nTotal chunks in Weaviate: {total}")

        if args.count:
            return

        # ── Per-company breakdown ─────────────────────────────────────────────
        print("\nBreakdown by company:")
        agg = collection.aggregate.over_all(
            group_by="company",
            total_count=True,
        )
        for group in agg.groups:
            print(f"  {group.grouped_by.value:<8} {group.total_count} chunks")

        # ── Fetch objects ─────────────────────────────────────────────────────
        fetch_kwargs = dict(
            limit=args.limit,
            include_vector=args.vectors,
            return_metadata=MetadataQuery(creation_time=True, distance=False),
        )

        if args.company:
            fetch_kwargs["filters"] = Filter.by_property("company").equal(args.company)

        response = collection.query.fetch_objects(**fetch_kwargs)

        print(f"\n{'─'*70}")
        print(f"Showing {len(response.objects)} chunk(s)"
              + (f" for company={args.company}" if args.company else ""))
        print(f"{'─'*70}\n")

        for obj in response.objects:
            p = obj.properties
            print(f"UUID        : {obj.uuid}")
            print(f"Company     : {p['company']}  |  Quarter: {p['quarter']}  |  Year: {p['year']}")
            print(f"Source file : {p['source_file']}")
            print(f"Chunk index : {p['chunk_index']}")
            print(f"Pages       : {p['page_start']} – {p['page_end']}")
            print(f"Tokens      : {p['token_count']}")
            print(f"Text preview: {p['chunk_text'][:300].replace(chr(10), ' ')} …")

            if args.vectors and obj.vector:
                vec = obj.vector.get("default", [])
                print(f"Vector dim  : {len(vec)}")
                print(f"Vector[:8]  : {[round(v, 5) for v in vec[:8]]} …")

            print()

    finally:
        client.close()


if __name__ == "__main__":
    main()
