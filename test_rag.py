
from ingestion.scraper import scrape_url
from retrieval.chunking import chunk_documents
from retrieval.vector_store import create_vector_store
from retrieval.rag_chain import answer_question

# Step 1: Scrape documentation
url = "https://docs.python.org/3/tutorial/introduction.html"

document = scrape_url(url)
print("Scraping completed!")

# Step 2: Split into chunks
chunks = chunk_documents([document])
print("Chunking completed!")

# Step 3: Create vector database
vector_store = create_vector_store(chunks)
print("Vector database created!")

# Step 4: Ask a question
question = "What are Python lists?"

result = answer_question(vector_store, question)

# Step 5: Display answer
print("\nDOCUMIND AI ANSWER:")
print(result["answer"])

print("\nSOURCES:")
for source in result["sources"]:
    print(source)
