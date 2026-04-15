# Data Agent (Text-to-SQL)

A modular Text-to-SQL data agent built with FastAPI, LangChain, and SQLite.

This system converts natural language queries into SQL, executes them on a database, and returns answers in natural language.

---

## 🚀 Features

- Natural language → SQL pipeline
- Knowledge-base-driven retrieval (schema + SQL templates + business context)
- SQL generation (LLM + fallback mechanism)
- Safe SQL execution (SELECT-only)
- Natural language answer generation
- FastAPI `/query` endpoint
- Evaluation dataset and evaluation script

---

## 🧱 Tech Stack

- FastAPI
- LangChain
- SQLite
- Python
- OpenRouter (LLM, optional)
- JSON-based knowledge base

---

## 📁 Project Structure

```
app/
  api/           # FastAPI routes
  services/      # retrieval, SQL generation, execution, answer generation
  models/        # request / response schema
  core/          # config
kb/              # schema docs, SQL templates, business context
scripts/         # db init, seed, tests, evaluation
tests/           # evaluation dataset
```

---

## 🧠 How It Works

1. User sends a natural language query
2. Retrieval module finds relevant:
   - schema docs
   - SQL templates
   - business context

3. SQL generation module creates SQL (LLM or fallback)
4. Execution module runs SQL against SQLite
5. Answer generation module converts result into natural language

---

## 📡 API

### POST `/query`

Request:

```json
{
  "query": "Who has the highest deposit?"
}
```

Response:

```json
{
  "answer": "The customer with the highest deposit is 劉志強, with a deposit amount of 9,200,000.",
  "generated_sql": "SELECT ...",
  "query_result": [
    {
      "customer_name": "劉志強",
      "deposit_amount": 9200000.0
    }
  ]
}
```

---

# 🖥️ Run Locally

## ✅ Windows

### 1. Create virtual environment

```bash
python -m venv venv
```

### 2. Activate environment

```bash
venv\Scripts\Activate.ps1
```

If blocked:

```bash
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

---

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

### 4. Initialize database

```bash
python scripts\init_db.py
python scripts\seed_data.py
```

---

### 5. Start API server

```bash
python -m uvicorn app.main:app --reload
```

---

### 6. Open API docs

```
http://127.0.0.1:8000/docs
```

---

### 7. Test API

```bash
python scripts\test_api.py
```

---

## 🍎 macOS

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

python scripts/init_db.py
python scripts/seed_data.py

python -m uvicorn app.main:app --reload
```

---

# 🧪 Evaluation

Run:

```bash
python -m scripts.run_eval
```

---

## What it evaluates

- Exact SQL match
- Query result match

---

# 🤖 AI Coding Tool Usage

AI coding tools (ChatGPT / LLM-based assistants) were used for:

- system design and architecture planning
- code scaffolding
- debugging environment issues (venv, packages)
- refining prompt design
- improving modular structure

All AI-generated code was reviewed and modified manually.

---

# ⚠️ Current Limitations

- Retrieval is currently rule-based (not OpenSearch yet)
- OpenRouter requires credits (fallback is used if unavailable)
- SQL generation may fallback to template-based logic
- Answer generation fallback is rule-based

---

# 🔮 Future Improvements

- Integrate OpenSearch as vector database
- Improve SQL logical equivalence checking
- Expand evaluation dataset
- Enhance prompt engineering
- Support more complex queries

---

# 🎯 Summary

This project demonstrates a complete Text-to-SQL pipeline:

- Retrieval
- SQL generation
- Execution
- Answer generation
- Evaluation

Designed to be modular, extensible, and production-ready.
