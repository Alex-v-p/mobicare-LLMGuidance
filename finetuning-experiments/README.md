# Experiments

This package contains experimentation tooling for building and evaluating benchmark datasets and model runs.

## Dataset generation from PDF

The dataset generation logic is intentionally separated from `datasets/` so that `datasets/` can remain focused on dataset schemas and loading logic.

### What it does

The dataset generator:
1. reads a PDF document,
2. extracts page-aware passages,
3. sends those passages to an OpenAI-compatible model endpoint such as Ollama,
4. produces a benchmark dataset matching the `BenchmarkCase` schema.

### Start Ollama

Use the included compose file:

```bash
docker compose -f docker-compose.ollama.yml up -d
```

Then pull a model into the running container:

```bash
docker exec -it experiments-ollama ollama pull llama3.1
```

The dataset generator defaults to:
- base URL: `http://localhost:11434/v1`
- API key: `ollama`
- model: `llama3.1`

### Generate a dataset


pip install -r requirements.txt

```bash
python -m dataset_generation.cli \
  --input-pdf ./dataset_generation/docs/HF_GuidLine_Christof_ehab368_Suppl.pdf \
  --output ./datasets/benchmark_v1.json \
  --config ./configs/dataset_generation.example.json
```

You can override the question-type mix directly:

```bash
python -m cli.main generate-dataset \
  --input-pdf ./my-guideline.pdf \
  --output ./datasets/benchmark_v1.json \
  --mix '{"factual":0.35,"clinical_scenario":0.35,"paraphrased_factual":0.2,"slightly_indirect":0.1}'
```

### Output shape

Each generated case matches this structure:

```json
{
  "id": "case-001",
  "question": "What therapy should be considered for symptomatic HFrEF despite ACE inhibitor and beta blocker?",
  "patient_variables": {
    "age": 72,
    "ef": 32
  },
  "gold_passage_id": "hf_p17_03",
  "gold_passage_text": "For symptomatic HFrEF despite ACE inhibitor and beta-blocker therapy, mineralocorticoid receptor antagonist therapy should be considered.",
  "page": 17,
  "reference_answer": "Mineralocorticoid receptor antagonist therapy should be considered.",
  "required_facts": [
    "consider mineralocorticoid receptor antagonist",
    "therapy should be based on symptomatic HFrEF guidance"
  ],
  "forbidden_facts": [
    "stop beta blocker immediately"
  ],
  "tags": ["heart-failure", "therapy", "factual"]
}
```

### Notes

- The dataset generator does **not** add runtime-derived benchmark fields such as expected chunk IDs or source lists.
- Those should be added later during benchmark preparation and source mapping.
- If the PDF has poor text extraction quality, the generated questions will also suffer. In that case, use a cleaner source PDF.
