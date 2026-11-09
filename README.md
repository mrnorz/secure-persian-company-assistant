# Persian Secure Company Assistant

A local Persian RAG assistant with role-based access control. Documents are filtered before retrieval, so unauthorized document text is never sent to the language model.

## Workspace improvements

- Persian RTL interface with native theme tokens in `.streamlit/config.toml`.
- Select one or more authorized PDFs, or search all accessible documents.
- Quick actions for excerpt summaries and explicit tasks/deadlines.
- Streaming answers, expandable source text, elapsed time, and Markdown export.
- Lazy embedding loading: login and navigation do not rebuild the index.
- Summaries sample document sections; they are explicitly labeled as excerpt summaries, not full-document reviews.
- PDFs are parsed in memory, avoiding a shared temporary upload file.

Changing document selection applies to future questions. New conversation starts an empty thread while retaining earlier threads in the session.

## Updated workflow

- Switch between **گفتگو** and **کتابخانهٔ اسناد** in the sidebar.
- Search document titles and start a summary scoped to a specific file.
- Maintain up to 20 conversations in the current browser session; a new conversation preserves earlier ones. Signing out or losing document access clears them. This is session history, not permanent disk storage.
- Follow-up questions can refer to the previous question within the same document selection. Previous answers are never treated as document evidence.
- Retrieval combines multilingual embeddings and exact word overlap, suppresses duplicate excerpts, and favors coverage across pages.
- The selected index and up to 24 question embeddings are cached within the session. Document access is rechecked before using cached results.
- Copy or download a single answer, export source excerpts with the conversation, or retry a failed request.
- New uploads are checked for duplicate content and limited to 30 MB. Legacy uploads without a content hash remain intact.

The interface displays total answer time and retrieval time; no fixed speedup is guaranteed. Summaries still cover sampled excerpts rather than every paragraph.

## Validation

```powershell
python -m unittest test_workspace test_retrieval test_app_ui -v
```

UI tests use a temporary database, synthetic documents, and mocked Ollama responses. They do not download models or send real company documents to a model. Run real-answer quality checks separately against your own documents and hardware.

## Visual theme

The interface uses a navy sidebar, teal accents, a light workspace, and locally hosted Vazirmatn regular/bold fonts. Theme colors and typography are configured in `.streamlit/config.toml`. The opening illustration and brand mark are vector assets in `static/`; the font license is included in `static/fonts/OFL.txt`.

After updating to this theme, fully restart Streamlit to enable the static font routes, then refresh the browser. No third-party font requests are needed. Only public design assets belong under `static/`; keep company documents in `data/`.

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
