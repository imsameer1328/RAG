# DocuMindAI

DocuMindAI is a local URL question-answering app. It extracts text from public HTML pages, stores chunks in a persistent Chroma index, retrieves relevant passages, and asks a local Ollama model to answer with source citations.

## Requirements

- Python 3.10 or newer
- Ollama installed and running
- One Ollama chat model and one embedding model

The default model names are `llama3.2` and `nomic-embed-text`. Pull them if they are not already installed:

```powershell
ollama pull llama3.2
ollama pull nomic-embed-text
```

If you use different models, set `OLLAMA_CHAT_MODEL` and `OLLAMA_EMBED_MODEL` in `.env` to the exact names shown by `ollama list`.

## Setup on Windows PowerShell

```powershell
py -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
if (-not (Test-Path .env)) { Copy-Item .env.example .env }
```

If `.env` already exists, edit it instead of replacing it. Set model names there if needed.

## Run

Make sure Ollama is running, then start the app on the local machine:

```powershell
streamlit run app.py --server.address 127.0.0.1
```

Enter a public webpage URL, wait for indexing to finish, then ask a question. The Chroma index is saved under `data/chroma` and remains available after restarting the app.

## Notes

- The default scraper blocks hosts that resolve to private or local IP addresses, checks each redirect, and caps page downloads at 5 MB. Set `ALLOW_PRIVATE_URLS=true` only when you intentionally need to ingest an internal site.
- The app sends page excerpts to the configured local Ollama model. It does not call a hosted LLM API.
- The vector index and embedding model should remain consistent. If you change the embedding model, clear `data/chroma` and re-index your pages.
