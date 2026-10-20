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

### Backend

#### Windows (PowerShell)
```powershell
$env:PYTHONPATH="."
uvicorn backend.app.main:app --reload --host 0.0.0.0 --port 8000
```

#### macOS / Linux
```bash
PYTHONPATH=. uvicorn backend.app.main:app --reload --host 0.0.0.0 --port 8000
```

### Frontend

```bash
cd frontend
npm run dev
```

## Testing

```bash
pytest backend/tests
```
