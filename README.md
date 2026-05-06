# Data Agent (Text-to-SQL)

A modular Text-to-SQL data agent built with FastAPI, LangChain, and SQLite.

This system converts natural language queries into SQL, executes them on a database, and returns answers in natural language.

---

## Features

- Natural language to SQL pipeline
- Knowledge-base-driven retrieval using schema docs, SQL templates, and business context
- SQL generation with LLM-first flow plus fallback and repair logic
- Safe SQL execution with SELECT-only enforcement
- Natural language answer generation
- FastAPI `POST /query` endpoint
- Evaluation dataset and evaluation report generation
- OpenSearch vector DB support implemented at code level, with local fallback retrieval

---

## Tech Stack

- FastAPI
- LangChain
- SQLite
- Python
- OpenRouter as the primary LLM provider
- Sentence Transformers for local embeddings
- OpenSearch for optional vector / keyword / hybrid retrieval
- JSON-based knowledge base

---

## Project Structure

```text
app/
  api/           # FastAPI routes
  services/      # retrieval, SQL generation, execution, answer generation
  skills/        # modular pipeline skills
  models/        # request / response schema
  core/          # config
kb/              # schema docs, SQL templates, business context
scripts/         # db init, seed, eval, inspection scripts
tests/           # evaluation dataset
reports/         # evaluation outputs
```

---

## Architecture Design

The system follows a modular pipeline architecture:

- Retrieval Layer: selects relevant schema, SQL templates, and business context
- SQL Generation Layer: generates SQL using LLM or fallback strategy
- Execution Layer: safely executes SQL against SQLite
- Answer Layer: converts structured results into natural language

The main request flow is:

`POST /query -> RetrievalSkill -> SQLSkill -> ExecutionSkill -> AnswerSkill`

---

## How It Works

1. User sends a natural language query.
2. Retrieval finds relevant:
   - schema docs
   - SQL templates
   - business context
3. SQL generation combines the user query and retrieval context.
4. The system generates SQL with an LLM, or falls back to template-based logic if needed.
5. The SQL is executed against the SQLite banking database.
6. The query result and original question are turned into a final natural language answer.

---

## API

### POST `/query`

Request:

```json
{
  "user_query": "Who has the highest deposit?"
}
```

Legacy clients may still send `query`, but `user_query` is the canonical request field.

Response:

```json
{
  "answer": "The customer with the highest deposit is Liu Chih-Chiang with a deposit amount of 9,200,000.",
  "generated_sql": "SELECT ...",
  "query_result": [
    {
      "customer_name": "Liu Chih-Chiang",
      "deposit_amount": 9200000.0
    }
  ]
}
```

`query_result` is normally returned and remains available for debugging or evaluation, but the response schema also allows it to be omitted when needed.

---

## Run Locally

### Windows

1. Create a virtual environment

```bash
python -m venv venv
```

2. Activate it

```bash
venv\Scripts\Activate.ps1
```

If PowerShell blocks execution:

```bash
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

3. Install dependencies

```bash
pip install -r requirements.txt
```

4. Initialize the database

```bash
python scripts\init_db.py
python scripts\seed_data.py
```

5. Start the API server

```bash
python -m uvicorn app.main:app --reload
```

6. Open API docs

```text
http://127.0.0.1:8000/docs
```

7. Test the API

```bash
python scripts\test_api.py
```

### macOS

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

python scripts/init_db.py
python scripts/seed_data.py

python -m uvicorn app.main:app --reload
```

---

## OpenSearch Support

OpenSearch vector DB support is implemented at the code level in this project.

What is already implemented:

- OpenSearch client integration for retrieval
- keyword retrieval against OpenSearch indices
- vector retrieval using embeddings and `knn_vector`
- hybrid retrieval with rank merging
- a local index build and ingestion script: `python -m scripts.build_vector_index`

Runtime notes:

- Local runtime can still use `RETRIEVAL_BACKEND=local`
- OpenSearch runtime verification requires a local Docker / OpenSearch environment
- If OpenSearch is unavailable, the retrieval layer falls back to local retrieval
- OpenSearch support should therefore be treated as implemented in code, but runtime verification is environment-dependent

### Optional OpenSearch Local Setup

Start OpenSearch:

```bash
docker compose up -d
```

Build the OpenSearch indices from the JSON knowledge base:

```bash
python -m scripts.build_vector_index
```

Switch retrieval backends with `RETRIEVAL_BACKEND` in `.env`:

```env
RETRIEVAL_BACKEND=local
```

or:

```env
RETRIEVAL_BACKEND=opensearch
```

---

## Offline / No OpenSearch Demo Mode

When OpenSearch is not available, use:

```env
RETRIEVAL_BACKEND=local
```

Then run:

```bash
python scripts/init_db.py
python scripts/seed_data.py
python -m uvicorn app.main:app --reload
python scripts/test_api.py
python -m scripts.run_eval
```

This mode can demo the complete Text-to-SQL pipeline:

- natural language query input
- retrieval
- SQL generation
- database execution
- natural language answer generation
- evaluation reporting

This mode does not mean OpenSearch runtime has been verified. It only confirms that the system can operate end-to-end with the local retrieval backend.

---

## Evaluation

Run:

```bash
python -m scripts.run_eval
```

### What it evaluates

- Exact SQL match
- Unordered query result match
- Ordered query result match
- Strict result match for `ORDER BY` queries
- Retrieval hit / recall reporting for schema, template, and business context

The generated evaluation report is written to:

```text
reports/eval_report.json
```

---

## AI Coding Tool Usage

This project uses `codexcli` as the required AI coding tool for:

- architecture review
- code analysis
- refactoring suggestions
- prompt engineering review
- evaluation improvement planning

ChatGPT or other LLM-based assistants may also be used for learning and design discussion, but `codexcli` is the primary AI coding tool used for project inspection and engineering support.

All AI-generated suggestions are reviewed manually before being applied.

---

## Current Limitations

- Local retrieval is still lightweight and rule-based compared with a production search stack
- OpenSearch runtime verification depends on having a local Docker / OpenSearch environment
- SQL generation may still fall back to template-based logic when LLM generation is unavailable
- Answer generation has a rule-based fallback path

---

## Future Improvements

- Improve local retrieval quality and retrieval ranking metrics
- Expand the SQL template library and business context coverage
- Improve SQL logical equivalence checking
- Expand the evaluation dataset
- Strengthen prompt engineering and structured output controls

---

## Summary

This project demonstrates a complete Text-to-SQL pipeline with:

- retrieval
- SQL generation
- execution
- answer generation
- evaluation

The current implementation is modular, laptop-friendly, and able to run in a local offline demo mode without requiring OpenSearch runtime.
