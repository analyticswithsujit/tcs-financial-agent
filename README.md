# TCS Financial Forecasting Agent

A LangChain-powered FastAPI service that analyses Tata Consultancy Services (TCS) quarterly financial reports and earnings-call transcripts to generate structured JSON forecasts using **Google Gemini (gemini-flash-latest)**.

This project only needs a free Gemini API key from Google AI Studio — no paid Google Cloud / Vertex AI project, billing account, or service account is required. (Note: free-tier keys are capped at ~20 requests/day per model, which is enough for testing but may need a short wait between forecast runs.)

---

## Prerequisites

You only need **two things** installed before running:

| Requirement | Download |
|---|---|
| **Docker Desktop** | https://www.docker.com/products/docker-desktop |
| **Git** | https://git-scm.com/downloads |

No Python, no MySQL, no pip — Docker handles everything else.

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

## Using the API

Once running, open your browser:

| URL | Description |
|---|---|
| http://localhost:8000/docs | Interactive Swagger UI — test all endpoints here |
| http://localhost:8000/health/capabilities | Check DB + vector store status |

### Run a forecast

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

## Architecture

```
FastAPI (POST /forecast/tcs)
    └── ForecastAgent (sequential tool pipeline + synthesis)
            ├── financial_data_extractor  → ChromaDB → Gemini (gemini-flash-latest)
            ├── qualitative_analysis_tool → ChromaDB RAG → Gemini (gemini-flash-latest)
            └── market_data_tool          → yfinance live data (optional)
                        ↓
                  MySQL 8.0 (async SQLAlchemy + aiomysql)
```

## Technical Stack

| Component | Choice |
|---|---|
| Framework | FastAPI + Uvicorn |
| LLM | Google Gemini `gemini-flash-latest` (`langchain-google-genai`) |
| Agent | Sequential LangChain tool calls + single Gemini synthesis call |
| Embeddings | Google `gemini-embedding-001` |
| Vector Store | ChromaDB (persistent, cosine similarity) |
| PDF Extraction | pdfplumber + pytesseract OCR fallback |
| Database | MySQL 8.0 (async SQLAlchemy 2.0 + aiomysql) |
| Containers | Docker Compose |

---

## Environment Variables

The setup script creates `.env` automatically. You can edit it manually if needed:

| Variable | Default | Description |
|---|---|---|
| `GOOGLE_API_KEY` | – | **Required** — from aistudio.google.com |
| `GOOGLE_MODEL` | `gemini-flash-latest` | Gemini model |
| `MYSQL_PASSWORD` | `tcsagent2024` | MySQL root password |
| `MYSQL_DB` | `tcs_forecast` | Database name |
| `CHROMA_PERSIST_DIR` | `./chroma_db` | ChromaDB storage |

---

## Project Structure

```
tcs-financial-agent/
├── setup.bat                        ← Windows: double-click to run
├── setup.sh                         ← Mac/Linux: ./setup.sh
├── docker-compose.yml               ← MySQL + app services
├── Dockerfile                       ← Python 3.11 + tesseract + poppler
├── requirements.txt
├── .env.example
└── app/
    ├── agents/forecast_agent.py     ← ForecastAgent (Gemini + LangChain)
    ├── api/endpoints.py             ← FastAPI routes
    ├── db/mysql_client.py           ← Async MySQL (SQLAlchemy)
    ├── rag/
    │   ├── document_loader.py       ← Downloads TCS PDFs from screener.in
    │   ├── chunker.py               ← pdfplumber + OCR chunking
    │   └── vector_store.py          ← ChromaDB wrapper
    ├── tools/
    │   ├── financial_extractor_tool.py   ← @tool: extract metrics
    │   ├── qualitative_analysis_tool.py  ← @tool: RAG sentiment analysis
    │   └── market_data_tool.py           ← @tool: live yfinance data
    ├── schemas/forecast.py
    ├── config.py
    └── main.py
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

**`429 ... generate_content_free_tier_requests, limit: 20` error**
→ Free-tier AI Studio keys are capped at 20 Gemini requests/day per model. This resets roughly every 24h. Either wait for the reset, or enable billing on the Google Cloud project behind your key to remove the cap.
