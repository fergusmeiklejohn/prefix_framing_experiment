# Getting Started

This guide walks you through running the prefix framing experiment locally with Ollama.

## Prerequisites

- Python 3.10+
- [Ollama](https://ollama.ai) installed and running
- A local model pulled (e.g., `llama3.2`, `mistral`, `qwen3`, `gpt-oss`)

## Setup

```bash
# Clone and install
cd prefix_framing_experiment
uv venv (if no virtual env already)
uv sync

# Start Ollama (in a separate terminal)
ollama serve

# Pull a model
ollama pull llama3.2
```

## Verify Setup

```bash
python scripts/test_setup.py --model llama3.2
```

This tests imports, metrics extraction, storage, and Ollama connectivity.

## Run Experiments

### Quick Test (27 trials, ~2 minutes)

```bash
python scripts/run_pilot.py -p 3 -x 3 -r 3 --no-eval
```

### Pilot Run (250 trials, ~20 minutes)

```bash
python scripts/run_pilot.py
```

### Full Experiment (9,000+ trials)

```bash
python scripts/run_experiment.py --replications 30
```

### Common Options

| Flag | Description |
|------|-------------|
| `--model`, `-m` | Ollama model name (default: llama3.2) |
| `--replications`, `-r` | Replications per condition |
| `--no-eval` | Skip LLM-as-judge evaluations (faster) |
| `--db` | Database path for results |

## Analyze Results

### List Experiments

```bash
python scripts/analyze.py --db results/pilot.db
```

### Generate Report

```bash
python scripts/analyze.py <experiment_id> --db results/pilot.db
```

This creates:
- `results/analysis/summary_stats.csv` - Means/std by category
- `results/analysis/effect_sizes.csv` - Cohen's d vs control
- `results/analysis/anova_results.csv` - Statistical tests
- `results/analysis/full_data.csv` - All trial data
- Various plots (boxplots, heatmaps, radar charts)

### Quick Summary Only

```bash
python scripts/analyze.py <experiment_id> --quick
```

## Resume Interrupted Experiments

If an experiment is interrupted, resume it:

```bash
python scripts/run_experiment.py --resume <experiment_id> --db results/experiment.db
```


## Example Workflow

```bash
# 1. Test setup
python scripts/test_setup.py

# 2. Quick pilot to verify pipeline
python scripts/run_pilot.py -p 3 -x 3 -r 3 --no-eval

# 3. Check results look sensible
python scripts/analyze.py --db results/pilot.db

# 4. Full pilot with evaluations
python scripts/run_pilot.py

# 5. Analyze pilot
python scripts/analyze.py <id> --db results/pilot.db

# 6. If satisfied, run full experiment
python scripts/run_experiment.py --replications 50
```

## Experiment Parameters

From the spec, the full experiment uses:
- **15 prompts** across 5 categories (factual, reasoning, problem, creative, technical)
- **20 prefixes** across 5 categories (positive, negative, cognitive, epistemic, control)
- **30-50 replications** per condition
- **Temperature 0.7** for generation diversity

Total: 15 × 20 × 30 = **9,000 trials** (minimum)
