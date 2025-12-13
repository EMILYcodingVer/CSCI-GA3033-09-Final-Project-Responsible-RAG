# Responsible RAG Copilot — Step-by-Step Tutorial

This tutorial helps you run the demo and reproduce the experiment results.

## 0) Prerequisites

- Python 3.10+
- An OpenAI API key

## 1) Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 2) Configure environment

```bash
cp .env.example .env
# edit .env and set OPENAI_API_KEY
```

Optional (override defaults):

- `CHAT_MODEL`
- `EMBED_MODEL`

## 3) Run the UI

```bash
uvicorn app_new:app --reload --port 8000
```

Open `http://127.0.0.1:8000`.

### What to try

1. Pick **BASELINE**, paste the loan approval prompt, click **ASK**.
2. Switch to **RAG** and repeat (same prompt).
3. Switch to **RESPONSIBLE** and repeat.

Observe:

- whether sources are shown (`Retrieved Docs`)
- whether the system surfaces trust indicators (`Reliability Snapshot`)
- whether risks are decomposed (`Safety Critic`)
- whether the final answer is rewritten safely (`Safe Final Answer`)
- optional structured reasoning (`Planner` collapsible)

## 4) Reproduce experiment results (script)

Run the same scenario from the command line and save outputs to JSON for reporting.

```bash
python scripts/experiment_loan_approval.py \
  --out outputs/loan_approval_results.json
```

This script:

- runs BASELINE vs RAG vs RESPONSIBLE
- stores answers and intermediate artifacts
- computes lightweight *trust metrics* used in the report (verifiability / risk visibility / actionability)

## 5) Unit tests

Recommended:

```bash
pytest -q
```

The tests mock LLM calls, so they are deterministic and do not spend tokens.

## 6) Troubleshooting

### A) `OPENAI_API_KEY` not found

- Ensure `.env` exists.
- Or export it:

```bash
export OPENAI_API_KEY='sk-...'
```

### B) Retrieval returns empty

- Confirm `corpus/` contains `.txt` files.
- If you add new docs, restart the server.

### C) App loads but buttons do nothing

- Check browser console + server logs.
- Confirm you started with `uvicorn app_new:app`.

### D) Slow responses

- Use a smaller chat model.
- Reduce number of retrieved docs / shorten prompt.

