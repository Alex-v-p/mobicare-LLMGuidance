# Fine-tuning experiments

## Dataset generation

Generate a benchmark dataset from a PDF:

```bash
python -m dataset_generation.cli \
  --input-pdf ./dataset_generation/docs/HF_GuidLine_Christof_ehab368_Suppl.pdf \
  --output ./datasets/benchmark_v1.json \
  --output-jsonl ./datasets/benchmark_v1.jsonl \
  --config ./configs/dataset_generation.example.json
```

Notes:
- generation is deterministic at the dataset serialization layer: cases are sorted by `id` and JSON keys are sorted
- progress is appended to JSONL while the run is in progress
- rerunning with the same JSONL path resumes from already generated cases by default
- failed cases are written to a sibling `*.failed.jsonl` file instead of aborting the whole run when `continue_on_error` is enabled
- for local Ollama models, lower concurrency is usually more stable than high concurrency for strict JSON generation


## Source mapping

Build run-specific chunk assignments from the frozen benchmark dataset after ingestion:

```bash
python -m cli.main build-source-maps --dataset ./datasets/document_sets/benchmark_v1.json --mapping-label naive_chunk300_overlap100 --run-ingestion --delete-collection-first --chunking-strategy naive --cleaning-strategy deep --gateway-url http://localhost:8000 --qdrant-url http://localhost:6333 --collection guidance_chunks --output ./artifacts/source_maps/benchmark_v1_naive_chunk300_overlap100.json --max-matches 5 --max-sequence-length 3 --page-window 2 
```

What this does:
- optionally triggers ingestion through the gateway and polls until completion
- deletes the current collection before ingestion by default, so runs do not stack into the same collection
- reads the current Qdrant payloads directly
- matches each benchmark case against chunks from the same source document
- writes a standalone run artifact with ranked chunk assignments for that exact run configuration

Important design change:
- the dataset is no longer enriched with `source_lists_by_strategy`
- assignments are saved only inside the run artifact for the specific configuration that produced them
- deterministic labels like `direct_evidence` / `partial_direct_evidence` / `supporting` are no longer written here
- this step now produces only ranked candidate chunk links, while later LLM-based grading can decide stricter evidence classes if you want that

Matching behavior now focuses on paragraph reconstruction rather than broad similarity:
- lexical passage coverage over the full gold passage
- anchor coverage from the start and end anchors
- sliding windows over chunk text
- adjacent chunk-pair reconstruction
- semantic score only as a last-resort optional fallback
