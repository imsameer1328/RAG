"""DocuMind AI — Knowledge Atelier UI (no sidebar).

Uses the existing backend, unchanged:
- ingestion.scraper.scrape_website(start_url, max_pages)
- retrieval.chunking.chunk_documents(documents)
- retrieval.vector_store.create_vector_store(chunks)
- agent.graph.build_agent(vector_store)

Run from the documind-ai project root: streamlit run app.py
"""

import html
import ipaddress
import logging
from pathlib import Path
from urllib.parse import urlparse

import streamlit as st
from langgraph.errors import GraphRecursionError

from ingestion.scraper import scrape_website
from retrieval.chunking import chunk_documents
from retrieval.vector_store import create_vector_store
from agent.graph import build_agent


ROOT = Path(__file__).resolve().parent
CSS_FILE = ROOT / "assets" / "styles.css"

st.set_page_config(
    page_title="DocuMind AI | Knowledge Atelier",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="collapsed",
)

if hasattr(st, "html"):
    st.html(CSS_FILE)
else:
    st.markdown(
        f"<style>{CSS_FILE.read_text(encoding='utf-8')}</style>",
        unsafe_allow_html=True,
    )


def markup(content: str) -> None:
    """Only render trusted templates; escape variable values first."""
    if hasattr(st, "html"):
        st.html(content)
    else:
        st.markdown(content, unsafe_allow_html=True)


logger = logging.getLogger("documind")
logging.basicConfig(level=logging.INFO)

DEFAULTS = {
    "vector_store": None,
    "agent": None,
    "doc_title": None,
    "doc_url": None,
    "page_count": 0,
    "chunk_count": 0,
    "chat_history": [],
}
for name, default in DEFAULTS.items():
    if name not in st.session_state:
        st.session_state[name] = default


def validate_url(value: str) -> str:
    """Basic UI validation; the crawler still needs its own network safety checks."""
    value = value.strip()
    parsed = urlparse(value)
    if parsed.scheme not in ("https", "http") or not parsed.hostname:
        raise ValueError("Enter a complete documentation URL, starting with https://")
    if parsed.username or parsed.password:
        raise ValueError("URLs containing a username or password aren't supported.")
    hostname = parsed.hostname.lower()
    if hostname == "localhost" or hostname.endswith((".localhost", ".local")):
        raise ValueError("Please provide a public documentation website.")
    try:
        ip = ipaddress.ip_address(hostname)
    except ValueError:
        ip = None
    if ip is not None and not ip.is_global:
        raise ValueError("Private and local IP addresses aren't supported.")
    return value


def graph_state(question: str) -> dict:
    return {
        "question": question,
        "original_question": question,
        "documents": [],
        "generation": "",
        "sources": [],
        "relevant": False,
        "retries": 0,
        "answer_supported": False,
        "answer_retries": 0,
    }


def normalize_sources(values) -> list[dict]:
    seen = set()
    cleaned = []
    for value in values or []:
        if isinstance(value, dict):
            title = str(value.get("title") or "Documentation page")
            url = value.get("url") or value.get("source") or ""
        else:
            title, url = "Documentation page", str(value)
        if not isinstance(url, str):
            continue
        url = url.strip()
        parsed = urlparse(url)
        if parsed.scheme not in ("https", "http") or not parsed.netloc or url in seen:
            continue
        seen.add(url)
        cleaned.append({"title": title, "url": url})
    return cleaned


def render_sources(items, message_index: int) -> None:
    sources = normalize_sources(items)
    if not sources:
        return
    with st.expander(f"References  ·  {len(sources)} source(s)"):
        for i, source in enumerate(sources, start=1):
            left, right = st.columns([5, 1], vertical_alignment="center")
            with left:
                st.markdown(f"**{i:02d} · {source['title']}**")
                st.caption(source["url"])
            with right:
                st.link_button(
                    "Open ↗",
                    source["url"],
                    key=f"source_{message_index}_{i}",
                    use_container_width=True,
                )
            if i < len(sources):
                st.divider()


def render_diagnostics(message: dict) -> None:
    if message.get("error"):
        return
    with st.expander("Agent trace  ·  How this answer was produced"):
        a, b, c = st.columns(3)
        a.metric("Query rewrites", message.get("retries", 0))
        b.metric("Answer retries", message.get("answer_retries", 0))
        c.metric(
            "Grounding check",
            "Passed" if message.get("answer_supported") else "Not passed",
        )
        st.caption("The grounding check is an LLM assessment, not a guarantee of factual accuracy.")


def run_agent(question: str) -> dict:
    try:
        with st.status("Searching, reasoning and checking sources…", expanded=False) as status:
            result = st.session_state.agent.invoke(
                graph_state(question), config={"recursion_limit": 20}
            )
            status.update(label="Response ready", state="complete", expanded=False)
        return {
            "role": "assistant",
            "content": result.get("generation") or "I couldn't generate a reliable answer.",
            "sources": result.get("sources", []),
            "retries": result.get("retries", 0),
            "answer_retries": result.get("answer_retries", 0),
            "answer_supported": result.get("answer_supported", False),
        }
    except GraphRecursionError:
        logger.exception("Agent recursion limit reached")
        message = "The agent reached its execution limit. Try a more focused question."
    except Exception as exc:
        logger.exception("Agent execution failed")
        detail = str(exc).lower()
        if any(x in detail for x in ("connection refused", "connecterror", "connection error")):
            message = "I couldn't reach Ollama. Make sure Ollama is running, then try again."
        elif "timeout" in detail or "timed out" in detail:
            message = "The request timed out. Please try again."
        else:
            message = "Something went wrong while generating the answer. Check the terminal logs."
    return {"role": "assistant", "content": message, "sources": [], "error": True}


# ── NAVIGATION / BRAND ────────────────────────────────────────────────────────
status_text = "SOURCE CONNECTED" if st.session_state.agent else "READY TO INDEX"
markup(
    f"""
    <header class="dm-topbar">
      <div class="dm-brand">
        <span class="dm-brand-mark">✦</span>
        <span class="dm-brand-name">DocuMind<span class="dm-brand-ai"> AI</span></span>
        <span class="dm-brand-sub">/ KNOWLEDGE ATELIER</span>
      </div>
      <div class="dm-topbar-right">
        <span class="dm-engine-tag">LANGGRAPH ENGINE</span>
        <span class="dm-connection"><i></i>{status_text}</span>
      </div>
    </header>
    """
)

# ── HERO ─────────────────────────────────────────────────────────────────────
markup(
    """
    <section class="dm-hero">
      <div class="dm-hero-copy">
        <div class="dm-overline">YOUR PERSONAL DOCUMENTATION INTELLIGENCE</div>
        <h1>Make sense of<br><em>any documentation.</em></h1>
        <p>Turn complex technical references into clear, source-backed answers.
           Powered by a self-checking, agentic retrieval workflow.</p>
        <div class="dm-hero-tags">
          <span>01 · INGEST</span><span>02 · RETRIEVE</span><span>03 · VERIFY</span>
        </div>
      </div>
      <div class="dm-hero-art" aria-hidden="true">
        <div class="dm-orbit dm-orbit-one"></div>
        <div class="dm-orbit dm-orbit-two"></div>
        <div class="dm-art-core">✦</div>
        <span class="dm-art-label dm-art-label-a">Knowledge</span>
        <span class="dm-art-label dm-art-label-b">Reasoning</span>
        <span class="dm-art-label dm-art-label-c">Evidence</span>
      </div>
    </section>
    """
)

# ── KNOWLEDGE SOURCE / FORM ──────────────────────────────────────────────────
markup(
    """
    <div class="dm-section-title">
      <div><div class="dm-section-index">01 / SOURCE SETUP</div>
      <h2>Connect a knowledge source</h2></div>
      <p>Start with a docs homepage or section URL.</p>
    </div>
    """
)

with st.container(border=True):
    with st.form("ingestion_form", clear_on_submit=False):
        col_url, col_depth = st.columns([3.5, 1.05], gap="medium", vertical_alignment="bottom")
        with col_url:
            doc_url = st.text_input(
                "Documentation website",
                placeholder="https://docs.python.org/3/tutorial/",
                help="Enter the URL of the documentation you want to explore.",
            )
        with col_depth:
            max_pages = st.selectbox("Pages to crawl", options=[5, 10, 20, 30, 50], index=1)
        clicked = st.form_submit_button(
            "✦  Build my knowledge base", type="primary", use_container_width=True
        )
    st.caption("Public documentation sites only · One active knowledge base at a time")

if clicked:
    try:
        url = validate_url(doc_url)
        with st.status("Indexing your documentation…", expanded=True) as progress:
            st.write("01 · Discovering documentation pages")
            documents = scrape_website(start_url=url, max_pages=max_pages)
            if not documents:
                raise ValueError("No readable documentation pages were found.")
            st.write("02 · Creating searchable text chunks")
            chunks = chunk_documents(documents)
            if not chunks:
                raise ValueError("No chunks could be created from the scraped pages.")
            st.write("03 · Building vector search index")
            new_store = create_vector_store(chunks)
            st.write("04 · Initializing LangGraph workflow")
            new_agent = build_agent(new_store)
            progress.update(label="Documentation indexed successfully", state="complete", expanded=False)

        # Only switch active source if the whole pipeline succeeded.
        st.session_state.vector_store = new_store
        st.session_state.agent = new_agent
        st.session_state.doc_url = url
        st.session_state.doc_title = documents[0].metadata.get("title", "Documentation")
        st.session_state.page_count = len(documents)
        st.session_state.chunk_count = len(chunks)
        st.session_state.chat_history = []
        st.toast("Knowledge base connected!", icon="✅")
        st.rerun()
    except ValueError as exc:
        st.warning(str(exc))
    except Exception:
        logger.exception("Knowledge ingestion failed")
        st.error(
            "Couldn't index this documentation. The previous knowledge base is unchanged. "
            "Check the URL and terminal logs."
        )

# ── KNOWLEDGE STATUS ─────────────────────────────────────────────────────────
if st.session_state.agent:
    title = html.escape(str(st.session_state.doc_title or "Documentation"))
    url = html.escape(str(st.session_state.doc_url or ""), quote=True)
    hostname = html.escape(urlparse(st.session_state.doc_url or "").hostname or "Website")
    markup(
        f"""
        <div class="dm-source-status">
          <span class="dm-live-dot"></span>
          <div class="dm-source-text"><strong>{title}</strong><small>{hostname}</small></div>
          <span class="dm-source-number"><strong>{st.session_state.page_count}</strong><small>PAGES</small></span>
          <span class="dm-source-number"><strong>{st.session_state.chunk_count}</strong><small>CHUNKS</small></span>
          <span class="dm-source-ready">● INDEX READY</span>
        </div>
        """
    )
else:
    markup(
        """
        <div class="dm-source-empty">
          <span class="dm-source-empty-icon">◈</span>
          <span>No knowledge base connected yet. Index a website to unlock the chat.</span>
        </div>
        """
    )

# ── CHAT WORKSPACE ────────────────────────────────────────────────────────────
left, right = st.columns([4.5, 1], vertical_alignment="bottom")
with left:
    markup(
        """
        <div class="dm-section-title dm-chat-title">
           <div><div class="dm-section-index">02 / RESEARCH SPACE</div>
           <h2>Ask your documentation</h2></div>
        </div>
        """
    )
with right:
    if st.button(
        "↺  Clear chat",
        use_container_width=True,
        disabled=not st.session_state.chat_history,
        key="clear_chat",
    ):
        st.session_state.chat_history = []
        st.rerun()

if not st.session_state.agent:
    markup(
        """
        <div class="dm-empty-chat">
          <div class="dm-empty-symbol">✳</div>
          <h3>Answers are closer than you think.</h3>
          <p>Connect a documentation website above, then explore it with natural language.
             The agent retrieves evidence, rewrites weak searches and checks its answers.</p>
          <div class="dm-empty-steps"><span>① Connect</span><span>② Ask</span><span>③ Explore</span></div>
        </div>
        """
    )
else:
    prompts = (
        "Summarize the core concepts",
        "How do I get started?",
        "What are the key examples?",
    )
    if not st.session_state.chat_history:
        markup(
            """
            <div class="dm-welcome-chat">
               <div class="dm-welcome-icon">✦</div>
               <h3>Your documentation is ready.</h3>
               <p>Try one of these prompts or type your own question below.</p>
            </div>
            """
        )
        p1, p2, p3 = st.columns(3, gap="small")
        with p1:
            starter1 = st.button(prompts[0], key="sample_1", use_container_width=True)
        with p2:
            starter2 = st.button(prompts[1], key="sample_2", use_container_width=True)
        with p3:
            starter3 = st.button(prompts[2], key="sample_3", use_container_width=True)
    else:
        starter1 = starter2 = starter3 = False

    for message_index, message in enumerate(st.session_state.chat_history):
        with st.chat_message(message["role"]):
            if message.get("error"):
                st.error(message["content"])
            else:
                st.markdown(message["content"])
            if message["role"] == "assistant":
                render_sources(message.get("sources", []), message_index)
                render_diagnostics(message)

    typed_question = st.chat_input(
        "Ask anything about the connected docs…",
        max_chars=1000,
    )
    starter_question = next(
        (prompt for clicked, prompt in zip((starter1, starter2, starter3), prompts) if clicked),
        None,
    )
    question = (typed_question or starter_question or "").strip()

    if question:
        st.session_state.chat_history.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            reply = run_agent(question)
            st.session_state.chat_history.append(reply)
            if reply.get("error"):
                st.error(reply["content"])
            else:
                st.markdown(reply["content"])
            render_sources(reply.get("sources", []), len(st.session_state.chat_history) - 1)
            render_diagnostics(reply)

# ── FOOTER ────────────────────────────────────────────────────────────────────
markup(
    """
    <footer class="dm-footer">
      <div><strong>DocuMind AI</strong><span>·</span> KNOWLEDGE, MADE CLEAR.</div>
      <div>LANGGRAPH <span>·</span> CHROMADB <span>·</span> OLLAMA <span>·</span> STREAMLIT</div>
    </footer>
    """
)
