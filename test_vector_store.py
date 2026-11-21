from ingestion.scraper import scrape_url
from retrieval.chunking import chunk_documents
from retrieval.vector_store import create_vector_store, search_documents

# 1. Scrape documentation
url = "https://docs.python.org/3/tutorial/introduction.html"

document = scrape_url(url)
print("Scraping completed!")

# 2. Chunk the document
chunks = chunk_documents([document])
print("Total chunks:", len(chunks))

# 3. Create embeddings and store chunks
vector_store = create_vector_store(chunks)
print("Documents stored in ChromaDB!")

# 4. Search using a natural-language question
query = "What are Python lists?"

results = search_documents(vector_store, query)

print("\nSEARCH RESULTS:")

for i, result in enumerate(results, start=1):
    print(f"\nResult {i}:")
    print(result.page_content[:500])
    print("Source:", result.metadata.get("source"))
