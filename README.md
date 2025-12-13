# Responsible RAG Copilot — Critic-Enhanced Trust-Aware Decision Support

**Course:** CSCI-GA 3033-091 (NYU) · **Authors:** Haoran Li, Lin Chen  
This repository implements a *critic-enhanced*, *inference-time* Responsible RAG system that improves **user trust** without modifying LLM parameters.

## 1) What this project does
We compare three modes under **identical prompts and model parameters**:

- **BASELINE**: plain LLM answer (fluent but unverifiable)
- **RAG**: answer grounded in retrieved documents (more evidence, but risks can remain implicit)
- **RESPONSIBLE**: RAG + **Planner** + **Safety Critic** + **Safe Rewrite** + **Trust-centered UI**

Key UI panels:
- **Retrieved Docs** (source transparency)
- **Safe Final Answer** (decision-ready output)
- **Safety Critic** (risks → impacts → suggested mitigations)
- **Reliability Snapshot** (overall / grounding / safety / evidence)
- **Planner (collapsible)** (transparent reasoning when needed)

## 2) Repository structure
- `app_new.py` — FastAPI web UI (three modes + panels)
- `answer_pipeline.py` — end-to-end Responsible RAG pipeline
- `rag.py` — retrieval and grounding utilities
- `planner.py` — planning / structuring module
- `critic.py` — safety critic review + suggestions
- `compare_systems.py` — baseline vs RAG vs Responsible comparison harness
- `config.py` — model/config defaults
- `documents/` — curated corpus (e.g., NIST AI RMF excerpts)
- `static/`, `templates/` — UI assets

## 3) Installation
### Option A: Local (recommended)
```bash
# 1) Create venv
python3 -m venv .venv
source .venv/bin/activate

# 2) Install dependencies
pip install -r requirements.txt

# 3) Configure environment
cp .env.example .env
# edit .env to set OPENAI_API_KEY
```

### Option B: Conda
```bash
conda create -n responsible-rag python=3.11 -y
conda activate responsible-rag
pip install -r requirements.txt
cp .env.example .env
```

## 4) Run the web app
```bash
uvicorn app_new:app --reload --port 8000
```
Open: `http://127.0.0.1:8000`

**Usage**
1. Select **BASELINE / RAG / RESPONSIBLE** in the left sidebar.
2. Paste the same query.
3. Click **ASK** to generate outputs.
4. Compare: evidence visibility, risk clarity, and actionability.

## 5) Troubleshooting
- **`OPENAI_API_KEY` not found**: ensure `.env` exists and contains a valid key.
- **Port already in use**: change the port, e.g. `--port 8080`.
- **RAG returns weak evidence**: add/curate documents in `documents/` (RAG is only as strong as the corpus).
- **Slow responses**: switch to a smaller model via `CHAT_MODEL` in `.env`.

