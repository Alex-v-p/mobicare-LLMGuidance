# Fine-tuning experiments

## Dataset generation

### 1. Generate a benchmark dataset from a PDF

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

---

### 2. Append biomarker-only cases to an existing dataset

You can extend an existing benchmark dataset with additional **biomarker-only** cases.

These cases:
- are generated with your **local LLM**
- are added to the **same dataset** as the normal paragraph-based questions
- still get evaluated by the **LLM judge** and other general answer-quality metrics
- are marked as **observation-only**, so deterministic paragraph-grounded scoring is skipped for them

Use this when you want to add cases where the inference request contains only biomarkers and no explicit user question.

```bash
python -m dataset_generation.biomarker_generator \
  --marker-ranges ../services/inference/src/inference/clinical/marker_ranges.json \
  --base-dataset ./datasets/benchmark_v1.json \
  --output ./datasets/benchmark_v1_plus_biomarkers.json \
  --dataset-size 25 \
  --model qwen2.5:7b
```

Notes:
- `--base-dataset` points to your existing benchmark dataset
- `--output` should be a **new file** so the original dataset is not overwritten
- `--dataset-size 25` appends 25 new biomarker-only cases
- the generated cases keep an evaluation question in the dataset, but the benchmark runner can omit that question from the actual inference request
- these cases are intended more for **behavior observation** than paragraph-level retrieval correctness

After generating the merged dataset, update your benchmark config to point to the new dataset file.

Example:
```json
"dataset_path": "./datasets/benchmark_v1_plus_biomarkers.json"
```

---

## Manual Source mapping

Build run-specific chunk assignments from the frozen benchmark dataset after ingestion:

```bash
python -m cli.main build-source-maps \
  --dataset ./datasets/document_sets/benchmark_v1.json \
  --mapping-label naive_chunk300_overlap100 \
  --run-ingestion \
  --delete-collection-first \
  --chunking-strategy naive \
  --cleaning-strategy deep \
  --output ./artifacts/source_maps/benchmark_v1_naive_chunk300_overlap100.json \
  --gateway-url http://localhost:8000 \
  --qdrant-url http://localhost:6333 \
  --collection guidance_chunks \
  --max-matches 5
```

Notes:
- `gateway_url` now points at the **Nginx reverse proxy** front door
- for a production-like stack with bearer auth, you can also add `--auth-token YOUR_TOKEN`
- for self-signed HTTPS in local testing, add `--no-verify-ssl`
- manual source mapping is mainly relevant for the normal paragraph-based benchmark cases
- biomarker-only observation cases do not rely on paragraph-to-question “best chunk” mappings in the same way

---

## Benchmarks & Campaigns

Run a single benchmark:

```bash
cd finetuning-experiments

python -m cli.main run-benchmark \
  --config ./configs/versions/minimal_benchmark.json \
  --format text
```

Campaigns allow you to run multiple benchmark configurations automatically:

```bash
python -m cli.main run-campaign \
  --config ./campaigns/my_first_campaign.json \
  --format text
```

Notes:
- a merged dataset containing both normal and biomarker-only cases can be run with the same benchmark config
- normal cases use the standard deterministic scoring and LLM judge flow
- biomarker-only observation cases still use LLM-based evaluation and general answer-quality metrics, but deterministic paragraph-grounded scoring is skipped where not applicable
- benchmark configs now support `execution.gateway_auth_mode`, bearer tokens, gateway login, locally generated JWTs, custom CA bundles, and `execution.gateway_verify_ssl`

Example execution block:

```json
"execution": {
  "gateway_url": "http://localhost:8000",
  "gateway_auth_mode": "none",
  "gateway_auth_token": null,
  "gateway_auth_email": null,
  "gateway_auth_password": null,
  "gateway_jwt_secret": null,
  "gateway_jwt_issuer": "mobicare-llm-api",
  "gateway_jwt_audience": "mobicare-gateway",
  "gateway_jwt_exp_minutes": 60,
  "gateway_verify_ssl": true,
  "gateway_ca_bundle_path": null
}
```

---

## Open the Streamlit dashboard

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
- an endpoint playground that can talk to the reverse-proxied gateway with bearer auth or a locally generated test JWT
