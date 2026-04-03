from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

from adapters.gateway import GatewayClient
from adapters.qdrant import QdrantScrollClient
from datasets.loader import load_benchmark_dataset
from datasets.schema import BenchmarkCase
from source_mapping.matcher import SourceMatcher


logger = logging.getLogger(__name__)


def _configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build run-specific chunk assignments for benchmark cases.")
    parser.add_argument("--dataset", required=True, help="Path to the benchmark dataset JSON file.")
    parser.add_argument("--mapping-label", default=None, help="Run-specific label for this assignment artifact.")
    parser.add_argument("--strategy", default=None, help="Deprecated alias for --mapping-label.")
    parser.add_argument("--output", required=True, help="Path for the assignment artifact JSON.")
    parser.add_argument("--output-enriched-dataset", default=None, help="Deprecated. Ignored on purpose.")
    parser.add_argument("--gateway-url", default="http://localhost:8000")
    parser.add_argument("--qdrant-url", default="http://localhost:6333")
    parser.add_argument("--auth-token", default=None, help="Optional bearer token for the gateway reverse-proxy.")
    parser.add_argument(
        "--verify-ssl",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Verify TLS certificates when the gateway URL uses HTTPS.",
    )
    parser.add_argument("--collection", default="guidance_chunks")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--run-ingestion", action="store_true")
    parser.add_argument(
        "--delete-collection-first",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Delete the current Qdrant-backed ingestion collection before re-ingesting.",
    )
    parser.add_argument("--cleaning-strategy", default="deep")
    parser.add_argument("--chunking-strategy", default="naive")
    parser.add_argument("--embedding-model", default=None)
    parser.add_argument("--chunk-size", type=int, default=300)
    parser.add_argument("--chunk-overlap", type=int, default=100)
    parser.add_argument("--poll-interval", type=float, default=2.0)
    parser.add_argument("--max-matches", type=int, default=5)
    parser.add_argument("--page-window", type=int, default=2)
    parser.add_argument(
        "--page-offset-candidates",
        default="0,-1,-2,1,-3,2,3",
        help="Comma-separated candidate offsets to compare dataset source_page against payload page_number.",
    )
    parser.add_argument(
        "--semantic-fallback",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Allow semantic-only fallback when lexical reconstruction found nothing usable.",
    )
    parser.add_argument(
        "--include-chunk-pairs",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Also test adjacent chunk pairs in addition to single chunks.",
    )
    parser.add_argument(
        "--mapping-profile",
        default="legacy_v1",
        choices=("legacy_v1", "semantic_recovery_v2"),
        help="Versioned source-mapping profile. Defaults to legacy_v1 for backward compatibility.",
    )
    parser.add_argument(
        "--llm-labeling-profile",
        default="heuristic_v1",
        choices=("heuristic_v1", "semantic_recovery_v2"),
        help="Heuristic second-pass labeling profile.",
    )
    parser.add_argument(
        "--max-sequence-length",
        type=int,
        default=3,
        help="Maximum adjacent chunk sequence length to reconstruct against the gold passage.",
    )
    parser.add_argument(
        "--semantic-fallback-max-matches",
        type=int,
        default=1,
        help="Maximum number of semantic-recovery strict matches to keep when the mapping profile enables it.",
    )
    parser.add_argument("--verbose", action="store_true")
    return parser


def _build_cases(raw_dataset: dict[str, Any]) -> list[BenchmarkCase]:
    return [BenchmarkCase(**case) for case in raw_dataset.get("cases", [])]


def _maybe_run_ingestion(args: argparse.Namespace) -> dict[str, Any] | None:
    if not args.run_ingestion:
        return None
    client = GatewayClient(base_url=args.gateway_url, auth_token=args.auth_token, verify_ssl=args.verify_ssl)
    if args.delete_collection_first:
        delete_result = client.delete_ingestion_collection()
        logger.info(
            "Deleted collection before ingestion collection=%s existed=%s",
            delete_result.get("collection"),
            delete_result.get("existed"),
        )
    payload = {
        "options": {
            "cleaning_strategy": args.cleaning_strategy,
            "cleaning_params": {},
            "chunking_strategy": args.chunking_strategy,
            "chunking_params": {
                "chunk_size": args.chunk_size,
                "chunk_overlap": args.chunk_overlap,
            },
            "embedding_model": args.embedding_model,
        }
    }
    logger.info(
        "Running ingestion before chunk assignment with cleaning=%s chunking=%s",
        args.cleaning_strategy,
        args.chunking_strategy,
    )
    result = client.run_ingestion_and_wait(
        payload,
        poll_interval_seconds=args.poll_interval,
    )
    return result.record


def _parse_page_offset_candidates(raw_value: str) -> tuple[int, ...]:
    parsed: list[int] = []
    for part in (raw_value or "").split(","):
        candidate = part.strip()
        if not candidate:
            continue
        parsed.append(int(candidate))
    return tuple(parsed or [0, -1, -2, 1, -3, 2, 3])


def _write_json(path: str | Path, payload: dict[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    _configure_logging(args.verbose)

    mapping_label = args.mapping_label or args.strategy
    if not mapping_label:
        parser.error("One of --mapping-label or --strategy is required.")
    if args.output_enriched_dataset:
        logger.warning("--output-enriched-dataset is deprecated and ignored. Chunk assignments now live only in the run artifact.")

    raw_dataset = load_benchmark_dataset(args.dataset)
    cases = _build_cases(raw_dataset)
    logger.info("Loaded dataset with %s cases", len(cases))

    ingestion_record = _maybe_run_ingestion(args)

    qdrant_client = QdrantScrollClient(url=args.qdrant_url, collection_name=args.collection)
    payloads = qdrant_client.fetch_all_payloads(batch_size=args.batch_size)
    logger.info("Loaded %s payloads from Qdrant collection=%s", len(payloads), args.collection)

    matcher = SourceMatcher(
        max_matches=args.max_matches,
        page_window=args.page_window,
        page_offset_candidates=_parse_page_offset_candidates(args.page_offset_candidates),
        semantic_fallback_enabled=args.semantic_fallback,
        include_chunk_pairs=args.include_chunk_pairs,
        max_sequence_length=args.max_sequence_length,
        mapping_profile=args.mapping_profile,
        semantic_fallback_max_matches=args.semantic_fallback_max_matches,
    )
    assignments: list[dict[str, Any]] = []

    for case in cases:
        assignment = matcher.build_chunk_assignment(case=case, mapping_label=mapping_label, payloads=payloads)
        assignments.append(assignment.to_dict())
        logger.info(
            "Mapped case=%s matches=%s best=%s",
            case.id,
            len(assignment.matches),
            assignment.matches[0].chunk_ids if assignment.matches else None,
        )

    artifact = {
        "dataset_version": raw_dataset.get("dataset_version"),
        "mapping_label": mapping_label,
        "collection": args.collection,
        "gateway_url": args.gateway_url,
        "qdrant_url": args.qdrant_url,
        "run_config": {
            "cleaning_strategy": args.cleaning_strategy,
            "chunking_strategy": args.chunking_strategy,
            "chunking_params": {
                "chunk_size": args.chunk_size,
                "chunk_overlap": args.chunk_overlap,
            },
            "embedding_model": args.embedding_model,
        },
        "matcher_config": {
            "max_matches": args.max_matches,
            "page_window": args.page_window,
            "page_offset_candidates": list(_parse_page_offset_candidates(args.page_offset_candidates)),
            "semantic_fallback_enabled": args.semantic_fallback,
            "include_chunk_pairs": args.include_chunk_pairs,
            "max_sequence_length": args.max_sequence_length,
            "semantic_fallback_max_matches": args.semantic_fallback_max_matches,
            "mapping_profile": args.mapping_profile,
            "llm_labeling_profile": args.llm_labeling_profile,
            "source_mapping_version": "2.15",
        },
        "ingestion_record": ingestion_record,
        "case_chunk_assignments": assignments,
    }
    _write_json(args.output, artifact)
    logger.info("Wrote chunk assignment artifact to %s", args.output)


if __name__ == "__main__":
    main()
