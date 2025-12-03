# Querying GPT-OSS Locally (Ollama)

How this repo talks to the local GPT-OSS model and how to debug it.

## Quick setup
- Use the repo venv: `uv venv && source .venv/bin/activate && uv sync`.
- Install/run Ollama and pull the model: `ollama pull gpt-oss:latest` (optional: `ollama pull qwen3:latest`).
- Verify Ollama sees the tag: `ollama list` (expects `gpt-oss:latest`).

## Configuration the code expects
- `config/models.yaml` defines the Ollama provider and model tag:
  ```yaml
  ollama:
    endpoint: http://localhost:11434/api/generate
    models:
      gpt-oss:
        name: gpt-oss:latest
        supports_seed: true
        default_max_tokens: 8192
  ```
- `config/experiment.yaml` uses the `gpt-oss` alias for the pilot run. Override at runtime with `--models`:
  ```
  python scripts/run_experiment.py --phase pilot --models gpt-oss
  ```

## Sanity checks
- End-to-end check (also confirms Ollama reachability):  
  `uv run python scripts/verify_setup.py` → look for `✓ Ollama/GPT-OSS reachable`.
- Direct single query using the same code path as experiments:
  ```bash
  uv run python - <<'PY'
  from src.models import create_provider

  provider = create_provider(
      "ollama",
      "gpt-oss:latest",
      endpoint="http://localhost:11434/api/generate",
  )
  resp = provider.generate(
      "You are a test probe. Reply with CHOICE A and one sentence of reasoning.",
      temperature=0.0,
      top_p=1.0,
      max_tokens=128,
      seed=123,
  )
  print("Raw:", resp.raw_text)
  print("Parsed choice:", resp.parsed_choice)
  PY
  ```
  This hits Ollama exactly as the framework does (`POST /api/generate` with `stream:false` and `options.temperature/top_p/num_predict/seed`).

## Running the pilot locally
- Ensure Ollama is running, then:  
  `uv run python scripts/run_experiment.py --phase pilot`  
  (adds Qwen3 locally with `--models gpt-oss,qwen3` if desired).
- Results land in `data/results/<experiment_id>/`; check `runs/all_runs.jsonl` for per-call payloads and responses.

## Common blockers
- Model not pulled → `ollama pull gpt-oss:latest`.
- Ollama daemon not running or wrong port → expected `http://localhost:11434`.
- Changed model tag → update `config/models.yaml` `name:` field to match.***
