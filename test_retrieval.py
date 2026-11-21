from ingestion.scraper import scrape_website
from retrieval.chunking import chunk_documents
from retrieval.vector_store import create_vector_store


def main():

    # 1. Crawl website
    print("\n--- SCRAPING WEBSITE ---")

    url = "https://docs.python.org/3/tutorial/"

    documents = scrape_website(url, max_pages=10)

    print("Pages scraped:", len(documents))

    if not documents:
        print("No documents found. Check your URL or scraper.")
        return

    # 2. Split documents into chunks
    print("\n--- CHUNKING DOCUMENTS ---")

    chunks = chunk_documents(documents)

    print("Total chunks:", len(chunks))

    if not chunks:
        print("No chunks generated.")
        return

    # 3. Create vector store
    print("\n--- CREATING VECTOR STORE ---")

    vector_store = create_vector_store(chunks)

    print("Vector store created successfully!")

    # 4. Test similarity relevance scores
    print("\n--- RELEVANCE SCORE TEST ---")

    query = "How do Python functions work?"

    results = vector_store.similarity_search_with_relevance_scores(
        query,
        k=5
    )

    for index, (document, score) in enumerate(results, start=1):
        print(f"\nResult {index}")
        print("Score:", round(score, 4))
        print("Source:", document.metadata.get("source", "Unknown"))
        print("Content:", document.page_content[:200])

    # 5. Test MMR retrieval
    print("\n--- MMR RETRIEVAL TEST ---")

    queries = [
        "How do Python functions work?",
        "What are Python lists and tuples?",
        "How does exception handling work in Python?"
    ]

    retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 5,
            "fetch_k": 20,
            "lambda_mult": 0.5
        }
    )

    for query in queries:

        print("\n" + "=" * 60)
        print("QUERY:", query)
        print("=" * 60)

        retrieved_docs = retriever.invoke(query)

        print("Retrieved chunks:", len(retrieved_docs))

        for index, document in enumerate(retrieved_docs, start=1):

            print(f"\nResult {index}")
            print("Source:", document.metadata.get("source", "Unknown"))
            print("Title:", document.metadata.get("title", "Unknown"))
            print("Content:", document.page_content[:300])

    print("\n--- RETRIEVAL TESTING COMPLETED ---")


if __name__ == "__main__":
    main()