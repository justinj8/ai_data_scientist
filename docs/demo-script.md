# Demo video script (60–90 seconds)

Record at 1440×900 with the dark theme, after `docker compose up --build` and adding a key in Settings.

1. **Data (0:00–0:12).** Click *Load sample data*. Point at the two sources: the uploaded files and a live
   SQLite database. Click `orders` to show the overview: quality warnings, distributions, correlations.
2. **A rigorous question (0:12–0:40).** Ask *"Which region has the highest average order value, and is the
   difference significant?"* Let the plan appear, then the SQL step and the statistical test with its effect size.
   Expand the SQL to show the code is real.
3. **Verification (0:40–0:50).** Pause on the *Verified* stamp: every number was found in the results.
4. **Modeling (0:50–1:10).** Ask *"What predicts customer churn?"* Show the baseline-vs-model table and the
   feature-importance chart. Pin the chart to the board.
5. **Switch model and export (1:10–1:25).** Switch to GPT or Grok in the model picker and ask a follow-up
   (*"Forecast daily revenue for the next 30 days"*). Open the Board tab, then click *Notebook* to export.

Suggested resume bullets (fill in the numbers from `evals/results/results.md`):

- Built an AI data-scientist agent (FastAPI, React/TypeScript) that answers plain-English questions over CSV/Excel
  files and six live databases by planning analyses and calling 9 tools: SQL, a stateful Jupyter kernel, and
  tested statistics, ML, forecasting, anomaly-detection and segmentation routines.
- Designed a provider-neutral agent loop supporting Claude, GPT and Grok, with API-key and ChatGPT-plan
  sign-in (OAuth 2.0 + PKCE), and encrypted local credential storage.
- Added answer verification (numeric grounding plus an LLM critic with one revision round); reached __% accuracy
  on a 24-question ground-truth evaluation built from synthetic data with planted effects.
- Enforced read-only data access (sqlglot SQL guard, read-only DB sessions, a token-protected kernel data bridge);
  shipped with about 250 backend tests, CI and a one-command Docker Compose stack.
