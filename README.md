# TCS Financial Forecasting Agent

A LangChain-powered FastAPI service that analyses Tata Consultancy Services (TCS) quarterly financial reports and earnings-call transcripts to generate structured JSON forecasts using **Google Gemini**.

This project only needs a free Gemini API key from Google AI Studio — no paid Google Cloud / Vertex AI project, billing account, or service account is required. (Note: free-tier keys are capped at ~20 requests/day per model, which is enough for testing but may need a short wait between forecast runs.)

---

## Architecture

```
FastAPI (POST /forecast/tcs)
    └── ForecastAgent (sequential tool pipeline + synthesis)
            ├── financial_data_extractor  --> ChromaDB --> Gemini (with model fallback)
            ├── qualitative_analysis_tool --> ChromaDB RAG --> Gemini (with model fallback)
            └── market_data_tool          --> yfinance live data (optional)
                        |
                  MySQL 8.0 (async SQLAlchemy + aiomysql)
```

The agent uses a **fixed sequential pipeline** — not a dynamic LLM-driven loop. Each tool is called in order, their outputs are collected, and a single final LLM call synthesises everything into a structured JSON forecast. This sidesteps the Gemini `thought_signature` issue that breaks LangChain's AgentExecutor multi-turn tool loop, and reduces total LLM round-trips per request.

---

## Agent & Tool Design

### Tool 1 — `financial_data_extractor`

Retrieves the top-8 most relevant chunks from ChromaDB (filtered to financial report documents only) using cosine similarity search, then calls Gemini with a structured extraction prompt.

**Prompt strategy:** instructs the model to extract only what is present in the provided context — never fabricate — and return a strict JSON schema with `quarters[]`, `data_quality`, and `source_documents`. Uses `response_mime_type: application/json` to enforce JSON-only output at the API level.

### Tool 2 — `qualitative_analysis_tool`

Runs **5 themed RAG queries** across earnings-call transcripts (demand outlook, attrition, deal wins, margin pressure, macro risks), deduplicates results, and passes the top-20 chunks to Gemini for sentiment analysis.

**Prompt strategy:** asks for `management_tone` (positive / cautious / negative), `key_themes`, `forward_looking_statements` (near-verbatim quotes only), `risks`, `opportunities`, `guidance_summary`, and a `sentiment_score` (0.0–1.0). The final score is blended 50/50 with a heuristic keyword counter to reduce hallucination drift.

### Tool 3 — `market_data_tool` *(optional bonus)*

Fetches live NSE price, 52-week range, P/E ratio, and market cap via `yfinance`. Only called when `include_market: true` is set in the request.

---

### Master Synthesis Prompt

After all tools run, a single LLM call combines their outputs into the final forecast. The prompt enforces **predictable, machine-readable JSON output**:

```
You are an expert financial analyst specialising in Indian IT services companies,
particularly Tata Consultancy Services (TCS).

Below is data already extracted for the last {N} quarter(s) by dedicated extraction
tools. Synthesise it into a single structured JSON forecast for the upcoming quarter.

Financial metrics (from financial_data_extractor): {financial_data}
Qualitative analysis (from qualitative_analysis_tool): {qualitative_data}
Market snapshot (from market_data_tool, only present if requested): {market_data}

CRITICAL — your response must be ONLY valid JSON (no markdown, no prose):

{
  "quarter_forecast": "<upcoming quarter label, e.g. Q1 FY27>",
  "financial_metrics": {
    "revenue_crore":      "<string or null>",
    "revenue_growth_yoy": "<string or null>",
    "net_profit_crore":   "<string or null>",
    "operating_margin":   "<string or null>",
    "ebitda_margin":      "<string or null>",
    "eps":                "<string or null>"
  },
  "qualitative_analysis": {
    "management_tone": "positive|cautious|negative",
    "key_themes": ["<theme>"],
    "forward_looking_statements": ["<quote>"],
    "risks": ["<risk>"],
    "opportunities": ["<opportunity>"]
  },
  "market_snapshot": null,
  "overall_outlook": "<2-3 sentence narrative forecast>",
  "confidence_score": 0.0,
  "source_documents": ["<filename>"]
}

Rules:
- Never fabricate numbers. Only use the data provided above.
- Set confidence_score 0.0-1.0 based on data completeness.
- market_snapshot = null unless market data was provided above.
```

**Why this controls the LLM reliably:**
- `CRITICAL` and `ONLY valid JSON (no markdown, no prose)` appear immediately before the schema — the model sees them last before generating.
- All fields use `"<string or null>"` to make null handling explicit.
- `confidence_score` is defined as a float with range, preventing vague qualifiers.
- The three rules act as a grounding checklist the model implicitly evaluates before responding.
- At the API level, `response_mime_type: application/json` is passed so Gemini structurally refuses non-JSON output — a hard guardrail on top of the prompt.

---

## Technical Stack

| Component | Choice |
|---|---|
| Framework | FastAPI + Uvicorn |
| LLM | Google Gemini — auto-fallback: `gemini-flash-latest` → `gemini-1.5-flash` → `gemini-1.5-pro` |
| Agent | Sequential LangChain tool calls + single Gemini synthesis call |
| Embeddings | Google `gemini-embedding-001` |
| Vector Store | ChromaDB (persistent, cosine similarity) |
| PDF Extraction | pdfplumber + pytesseract OCR fallback |
| Database | MySQL 8.0 (async SQLAlchemy 2.0 + aiomysql) |
| Containers | Docker Compose |

---

## Model Fallback

Every LLM call goes through `app/utils/llm_factory.py`, which tries models in this order:

```
gemini-flash-latest  -->  gemini-1.5-flash  -->  gemini-1.5-pro
```

Fallback triggers only on retryable errors (HTTP 429 quota exhausted, 503 service unavailable). Auth errors (`API_KEY_INVALID`) and content-policy rejections fail immediately — retrying with another model won't help. The embedding model is never changed by the fallback.

---

## Prerequisites

You only need **two things** installed before running:

| Requirement | Download |
|---|---|
| **Docker Desktop** | https://www.docker.com/products/docker-desktop |
| **Git** | https://git-scm.com/downloads |

No Python, no MySQL, no pip — Docker handles everything else.

> **Why Docker?** The project depends on MySQL 8.0, Tesseract OCR, and Poppler — three system-level packages that are painful to install manually on Windows. Docker bundles all of it: one command starts the database, the app, and every system dependency with zero manual setup.

---

## Quick Start (One Command)

### 1. Clone the repo

```bash
git clone https://github.com/analyticswithsujit/tcs-financial-agent.git
cd tcs-financial-agent
```

### 2. Get a FREE Gemini API key

Go to **https://aistudio.google.com/app/apikey** → click **Create API Key** → copy it.

### 3. Run the setup script

**Windows:**
```
Double-click setup.bat
```
or from Command Prompt / PowerShell:
```bat
setup.bat
```

**Mac / Linux:**
```bash
chmod +x setup.sh && ./setup.sh
```

The script will:
- Check Docker is running
- Ask for your Gemini API key (one-time only)
- Create the `.env` file automatically
- Build and start everything with Docker Compose

> **First run takes 3–5 minutes.** Docker pulls MySQL 8.0, builds the Python image, and installs all packages. Subsequent starts are instant.

---

## Environment Variables

The setup script creates `.env` automatically. You can edit it manually if needed:

| Variable | Default | Description |
|---|---|---|
| `GOOGLE_API_KEY` | – | **Required** — from aistudio.google.com |
| `GOOGLE_MODEL` | `gemini-flash-latest` | Primary Gemini model (fallback chain kicks in on errors) |
| `MYSQL_PASSWORD` | `tcsagent2024` | MySQL root password |
| `MYSQL_DB` | `tcs_forecast` | Database name |
| `CHROMA_PERSIST_DIR` | `./chroma_db` | ChromaDB storage |

---

## Using the API

Once running, open your browser:

| URL | Description |
|---|---|
| **http://localhost:8000** | Live forecasting dashboard UI |
| http://localhost:8000/docs | Interactive Swagger UI — test all endpoints here |
| http://localhost:8000/health/capabilities | Check DB + vector store status |

### Run a forecast (API)

```bash
curl -X POST http://localhost:8000/forecast/tcs \
  -H "Content-Type: application/json" \
  -d '{"ticker": "TCS", "quarters": 3, "sources": ["screener", "company-ir"], "include_market": false}'
```

> The **first forecast request takes 2–3 minutes** — the agent downloads TCS financial PDFs from screener.in and indexes them into ChromaDB. Every request after that is fast.

**Response:**
```json
{
  "status": "success",
  "request_id": "abc-123",
  "result_json": {
    "quarter_forecast": "Q1 FY27",
    "financial_metrics": {
      "revenue_crore": "~65,000",
      "revenue_growth_yoy": "~5.5%",
      "net_profit_crore": "~12,500",
      "operating_margin": "~24.5%"
    },
    "qualitative_analysis": {
      "management_tone": "cautious",
      "key_themes": ["deal wins", "AI demand", "margin pressure"],
      "risks": ["macro uncertainty", "pricing pressure"]
    },
    "overall_outlook": "TCS is expected to maintain steady growth...",
    "confidence_score": 0.78
  },
  "tools_used": ["financial_data_extractor", "qualitative_analysis_tool"]
}
```

### Check forecast status

```bash
curl http://localhost:8000/status/<request_id>
```

---

## Project Structure

```
tcs-financial-agent/
├── setup.bat                        <- Windows: double-click to run
├── setup.sh                         <- Mac/Linux: ./setup.sh
├── docker-compose.yml               <- MySQL + app services
├── Dockerfile                       <- Python 3.11 + tesseract + poppler
├── requirements.txt
├── .env.example
└── app/
    ├── agents/forecast_agent.py     <- ForecastAgent (sequential pipeline)
    ├── api/endpoints.py             <- FastAPI routes
    ├── db/mysql_client.py           <- Async MySQL (SQLAlchemy)
    ├── rag/
    │   ├── document_loader.py       <- Downloads TCS PDFs from screener.in
    │   ├── chunker.py               <- pdfplumber + OCR chunking
    │   └── vector_store.py          <- ChromaDB wrapper
    ├── tools/
    │   ├── financial_extractor_tool.py   <- @tool: extract metrics
    │   ├── qualitative_analysis_tool.py  <- @tool: RAG sentiment analysis
    │   └── market_data_tool.py           <- @tool: live yfinance data
    ├── utils/
    │   └── llm_factory.py           <- Model fallback chain
    ├── static/
    │   └── index.html               <- Live forecasting dashboard UI
    ├── schemas/forecast.py
    ├── config.py
    └── main.py
```

---

## Stop / Restart

```bash
# Stop containers
docker compose down

# Start again (no rebuild needed)
docker compose up -d

# View live logs
docker compose logs -f
```

---

## Troubleshooting

**`Docker is not running` error**
→ Open Docker Desktop and wait for "Engine running" in the bottom left, then retry.

**First forecast times out**
→ Normal on slow connections — the agent downloads several PDFs. Wait 3–4 minutes and retry.

**`GOOGLE_API_KEY invalid` error**
→ Open `.env`, check the key starts with `AIza` and has no extra spaces. Restart with `docker compose up -d`.

**Port 8000 already in use**
→ Change the port in `docker-compose.yml` from `"8000:8000"` to `"8001:8000"` and access via `localhost:8001`.

**`429 ... generate_content_free_tier_requests` error**
→ Free-tier keys are capped at 20 requests/day **per model**. The agent automatically falls back across the model chain (`gemini-flash-latest` → `gemini-1.5-flash` → `gemini-1.5-pro`), so all three models must be exhausted before you hit this wall. Either wait ~24 h for the quota to reset, or enable billing on your Google Cloud project to remove the cap.
