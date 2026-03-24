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

### 3. Append drug-dosing observation cases to an existing dataset

You can also extend an existing benchmark dataset with **drug-dosing-only observation cases**.

These cases:
- are added to the **same dataset** as the normal paragraph-based questions and biomarker-only observation cases
- are sent to the guidance API with `options.pipeline_variant = "drug_dosing"`
- omit the visible question from the actual request, just like the biomarker-only observation flow
- keep a hidden evaluation question and reference answer in the dataset for qualitative scoring and behavior review
- skip deterministic paragraph-grounded scoring, because they are intended for **behavior observation** rather than gold-passage matching

Use this when you want to benchmark the **sole drug pipeline** on realistic medication/safety profiles without forcing them into a paragraph-grounded question format.

```bash
python -m dataset_generation.drug_dosing_generator \
  --base-dataset ./datasets/document_sets/benchmark_v2_plus_biomarkers.json \
  --output ./datasets/document_sets/benchmark_v2_plus_biomarkers_plus_drug_dosing.json \
  --dataset-size 20
```

Notes:
- the generated cases are marked as `request_mode = "drug_dosing_only"` and `evaluation_profile = "observation_only"`
- each generated case stores `pipeline_variant = "drug_dosing"` in its metadata so mixed datasets can still route standard cases to the standard pipeline
- a mixed dataset can therefore contain **normal QA**, **biomarker-only observation cases**, and **drug-dosing-only observation cases** at the same time
- older run artifacts remain loadable in Streamlit; the dashboard simply surfaces the pipeline variant when it is present

Example mixed-dataset benchmark config:
```json
"dataset_path": "./datasets/document_sets/benchmark_v2_plus_biomarkers_plus_drug_dosing.json"
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
