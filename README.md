# Persian Secure Company Assistant

A local Persian RAG assistant with role-based access control. Documents are filtered before retrieval, so unauthorized document text is never sent to the language model.

## Requirements

- Python 3.10+
- Ollama installed and running
- A model pulled: `ollama pull qwen2.5:3b`

## Run

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

Run `streamlit run app.py`, then log in. The initial demo users are:

| User | Password | Role |
|---|---|---|
| `admin` | `change-me-admin` | Upload and manage documents |
| `manager` | `change-me-manager` | Access all documents |
| `employee` | `change-me-employee` | Access employee-level documents |

Change these credentials before any real deployment. The admin can upload PDFs and assign a minimum role. Employees only receive chunks from documents permitted for their role. The assistant cites source pages and refuses unsupported answers.

Set `OLLAMA_MODEL`, `OLLAMA_URL`, or `EMBED_MODEL` in the environment if needed. For production, put the app behind HTTPS and company SSO, use a managed database, rotate secrets, and keep the database/data directory encrypted.

برای تنظیمات بدون تغییر کد، فایل `.env.example` را به `.env` کپی کنید و مقدارهای آن را تغییر دهید:

```powershell
Copy-Item .env.example .env
```

تنظیمات اصلی شامل `EMBED_MODEL`، `OLLAMA_MODEL`، `TOP_K`، `SUMMARY_TOP_K`، `CHUNK_SIZE`، `CHUNK_OVERLAP`، `OLLAMA_TEMPERATURE` و `OLLAMA_NUM_CTX` است.
