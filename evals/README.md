# Evaluation

24 questions with known answers about the seeded retail sample data (`sample_data/out/ground_truth.json`).
The generator *plants* the answers (a promotion uplift, a West-region price premium, churn drivers, a region
effect that does not exist, spike and outage days), so a score is a real measure of correctness.

Check types (`backend/src/adsci/evals/checks.py`):

- `numeric`: a number in the answer matches the truth within a relative tolerance (percentages are matched as
  ratios too, so "10.8% higher" matches a ratio of 1.108).
- `set` / `contains_all` / `contains_any`: required values or phrases appear in the answer.
- `rubric`: a judge model grades the answer against written criteria (default judge: latest Claude Opus).

Run it (this calls the real model APIs and costs money):

```bash
uv run --project backend python -m adsci.evals.run --models opus,sonnet,gpt,grok
```

Options: `--limit 5` for a quick smoke run, `--judge sonnet` to change the grader. Results go to
`evals/results/results.md` (summary and per-question matrix) and `evals/results/raw/*.jsonl` (full answers).
