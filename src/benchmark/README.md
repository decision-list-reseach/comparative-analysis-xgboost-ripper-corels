# Research Benchmark Orchestrator

This directory contains the orchestration script designed to sequentially run, evaluate, and compare all baseline models (XGBoost, RIPPER, and CORELS) for the customer churn research paper.

## Purpose

The `run_all.py` script serves as the centralized testing harness for the repository. By running it, you ensure that:
1. Every model is evaluated using the exact same deterministic 5-fold cross-validation splits.
2. The predictive metrics (Accuracy, Precision, Recall, F1, Time) are correctly aggregated from the raw logs.
3. The interpretability metrics (Total Rules, Logical Conditions) are parsed from the exported rule lists.
4. A final unified leaderboard (`results/final_benchmark_leaderboard.md`) is automatically generated for easy inclusion into the final manuscript.

## Usage

You should run the script from the root of the repository to ensure all relative paths resolve correctly.

```bash
# Ensure you are at the root of the repository
cd /path/to/Research

# Activate the virtual environment
source .venv_corels_310/bin/activate

# Execute the benchmark
python benchmark/run_all.py
```

## Output

The script generates its final markdown output in the `results/` directory:
- `results/final_benchmark_leaderboard.md`

You can open this markdown file directly in your IDE or render it using a Markdown viewer to see the side-by-side metric tables and rankings.
