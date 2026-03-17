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
