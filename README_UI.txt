DocuMind AI — Knowledge Atelier UI (no sidebar)

1. Back up your current app.py.
2. Extract this ZIP into the root of your existing documind-ai project.
3. Keep your current agent/, ingestion/, retrieval/, and ChromaDB files.
4. The new app calls ingestion.scraper.scrape_website(start_url=..., max_pages=...). If your crawler is named crawl_website instead, change that import and call accordingly.
5. Run: streamlit run app.py
6. Check that Ollama is running before asking questions.

Requires your existing Streamlit, LangGraph and related dependencies; no new UI packages.

Notes:
- This is a presentation redesign, not a change to the backend.
- The client-side URL validation is only basic; public deployment requires URL redirect/DNS protection in the crawler itself.
- CSS uses some Streamlit internals and may need tweaking if Streamlit changes its DOM.
