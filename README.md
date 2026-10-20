# CrewDesk

Production-minded multi-agent research report application built with FastAPI, LangGraph, Groq, Pydantic, pytest, React, and Vite.

## Installation

1. Clone the repository and navigate into it.
2. Copy `.env.example` to `.env` and configure your `GROQ_API_KEY` and `TAVILY_API_KEY`.

### Backend Setup

#### Windows (PowerShell)
```powershell
python -m venv venv
.\venv\Scripts\Activate
pip install -r backend/requirements.txt
```

#### macOS / Linux
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r backend/requirements.txt
```

### Frontend Setup

```bash
cd frontend
npm install
```

## Running the Application

### 1. Start Backend (Terminal 1)

#### Windows (PowerShell)
```powershell
$env:PYTHONPATH="."
uvicorn backend.app.main:app --reload --host 0.0.0.0 --port 8000
```

#### macOS / Linux
```bash
PYTHONPATH=. uvicorn backend.app.main:app --reload --host 0.0.0.0 --port 8000
```

### 2. Start Frontend (Terminal 2)

```bash
cd frontend
npm run dev
```

### 3. Open Browser

Navigate to `http://localhost:3000` to access the CrewDesk dashboard.

## Features

- **Research Topic Input**: Enter a research topic and start a multi-agent research run.
- **Live Workflow Graph**: Visual status of Planner, Researcher, Writer, Critic, and Finalizer nodes.
- **Real-time Event Feed**: Stream events from the backend via Server-Sent Events (SSE).
- **Budget Tracking**: Monitor LLM call usage against configured budget.
- **Report Rendering**: View the final Markdown report with clickable references.
- **Revision Counter**: Track revision loops (0/2, 1/2, 2/2).

## Testing

```bash
pytest backend/tests
```

## API Endpoints

- `POST /api/runs` - Start a new research run.
- `GET /api/runs/{run_id}` - Get run status and summary.
- `GET /api/runs/{run_id}/report` - Get the completed report.
- `GET /api/runs/{run_id}/events` - Stream live events (SSE).
- `GET /health` - Health check.

## Configuration

Edit `.env` to configure:
- `GROQ_API_KEY` - Your Groq API key.
- `TAVILY_API_KEY` - Your Tavily Search API key.
- `GROQ_MODEL_NAME` - Model to use (default: llama-3.3-70b-versatile).
- `MAX_LLM_CALLS_PER_RUN` - Maximum LLM calls per run (default: 25).
- `MAX_SEARCH_CALLS_PER_RUN` - Maximum search requests per run (default: 15).