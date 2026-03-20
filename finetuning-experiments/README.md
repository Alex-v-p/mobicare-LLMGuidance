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


## Manual Source mapping

Build run-specific chunk assignments from the frozen benchmark dataset after ingestion:

```bash
python -m cli.main build-source-maps   --dataset ./datasets/document_sets/benchmark_v1.json   --mapping-label naive_chunk300_overlap100   --run-ingestion   --delete-collection-first   --chunking-strategy naive   --cleaning-strategy deep   --output ./artifacts/source_maps/benchmark_v1_naive_chunk300_overlap100.json   --gateway-url http://localhost:8000   --qdrant-url http://localhost:6333   --collection guidance_chunks   --max-matches 5
```

## Benchmakrs & Campaigns
```bash
cd finetuning-experiments

python -m cli.main run-benchmark \
  --config ./configs/versions/minimal_benchmark.json \
  --format text
```

Campaigns allow you to run multiple benchmark configurations automatically.
```bash
python -m cli.main run-campaign \
  --config ./campaigns/my_first_campaign.json \
  --format text
```

## Open the Streamlit dashboard:

```bash
streamlit run ui/streamlit_app.py
```

The dashboard shows:
- run overview
- run comparison
- per-case drilldown
- retrieved chunks
- source match candidates
- raw endpoint result
