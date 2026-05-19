# RetailMind AI ◈ Executive Analytics Dashboard

RetailMind AI is an interactive, local AI-powered analytics dashboard built for retail executives. It processes complex transaction, demographic, and campaign data (up to millions of rows) to provide conversational insights, automated metric calculations, and interactive visualizations while guaranteeing 100% data privacy.

<div align="center">
  <img src="assets/Dashboard.png" alt="RetailMind AI Dashboard" width="800"/>
</div>

## Architecture

```text
┌──────────────────┐    Pandas / SQLite    ┌───────────────────┐    Local API    ┌──────────────┐
│  Streamlit UI    │ ────────────────────▶ │ CrewAI / LangChain│ ───────────────▶│    Ollama    │
│    (app.py)      │ ◀──────────────────── │  (ai_agents.py)   │ ◀───────────────│ (Llama 3.1)  │
└──────────────────┘   SQL + AI Insights   └───────────────────┘                 └──────────────┘
```

The application combines a reactive frontend with a multi-agent backend. The backend constrains the LLM logic using CrewAI, strictly separating natural language routing, SQLite query generation, and narrative insight generation.

## Features
- **Natural Language to SQL:** Ask questions in plain English ("Which categories are driving growth?"). The local AI Agent translates them into SQL queries executed against an in-memory SQLite database.
- **Graceful Data Engineering:** Memory-optimized Pandas pipelines parse multiple large CSVs, downcast data types automatically, and execute fast programmatic joins.
- **Behavioral & Demographic Analytics:** Automatically segments customers by spending trends, engagement scores, and demographic profiles.
- **Privacy-Preserving Local AI:** Utilizes local LLMs via Ollama. No proprietary retail data is ever sent to OpenAI, Anthropic, or external APIs.

<div align="center">
  <img src="assets/AIOutput.png" alt="AI Natural Language to SQL" width="800"/>
</div>

## Prerequisites
- **Python 3.10+**
- **Ollama** (Required for local structured outputs and inference)
- ~5 GB disk space for the default local model (Llama 3.1)

## Setup

**1. Install Ollama & Pull Models**
Start the Ollama daemon in your terminal, then pull the required models:

```bash
ollama serve            # Leave running in its own terminal
ollama pull llama3.1    # Default model
ollama pull qwen2.5-coder:3b # Optional: Highly recommended for SQL generation
```

**2. Install Python Dependencies**

```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
pip install -r requirements.txt
```

**3. Prepare Data**
Ensure your retail datasets are placed in the root directory. Required files:
- `transaction_data.csv`
- `product.csv`
- `hh_demographic.csv`
- `coupon_redempt.csv`

## Run
You will need two terminals running simultaneously:

**Terminal 1 — Ollama Backend**
```bash
ollama serve
```

**Terminal 2 — Streamlit Frontend**
```bash
streamlit run src/tools/app.py
```

Open the Streamlit URL provided in the terminal (usually http://localhost:8501).

## File Map

| File | Purpose |
| :--- | :--- |
| `app.py` | Main Streamlit UI, page configuration, and Plotly charting. |
| `ai_agents.py` | CrewAI agent definitions (Semantic Router, SQL Generator, Retail Analyst). |
| `analysis_engine.py` | Pure Pandas data manipulation for KPIs and customer segmentation. |
| `sql_engine.py` | In-memory SQLite database creation and query execution. |
| `data_loader.py` | Data normalization, type downcasting, and automated schema merging. |

## Notes on Behavior
- **Memory Management:** The system handles Out-Of-Memory (OOM) risks by automatically downcasting `float64` and `int64` to 32-bit/16-bit equivalents, and converting low-cardinality strings to Pandas `category` types.
- **Fallback UI:** If the local LLM is unreachable or times out, the application gracefully degrades, still providing the full suite of Pandas-calculated metrics and Plotly charts without the AI narrative block.
- **Causal Data:** To optimize standard loading times, `causal_data.csv` (600MB+) is excluded from the default boot sequence but can be toggled via the application cache logic.

## Author
**Kinjal Goyal**
