from ingestion.scraper import scrape_url
from retrieval.chunking import chunk_documents
from retrieval.vector_store import create_vector_store
from agent.graph import build_agent


# 1. Ingest documentation
url = "https://docs.python.org/3/tutorial/introduction.html"

document = scrape_url(url)
chunks = chunk_documents([document])

print("Documents scraped and chunked!")


# 2. Create vector database
vector_store = create_vector_store(chunks)

print("Vector database ready!")


# 3. Build LangGraph agent
agent = build_agent(vector_store)

print("LangGraph agent initialized!")


# 4. Ask a question
question = "What are Python lists and how do they work?"

initial_state = {
    "question": question,
    "original_question": question,
    "documents": [],
    "generation": "",
    "sources": [],
    "relevant": False,
    "retries": 0,
    "answer_supported": False,
    "answer_retries": 0
}


# 5. Execute the graph
result = agent.invoke(initial_state)


# 6. Display the final answer
print("\n" + "=" * 50)
print("DOCUMIND AI RESPONSE")
print("=" * 50)

print(result["generation"])

print("\nSOURCES:")

for source in result["sources"]:
    print(source)

print("\nREWRITES:", result["retries"])

print("\nANSWER SUPPORTED:", result["answer_supported"])
print("ANSWER RETRIES:", result["answer_retries"])
