from ingestion.scraper import scrape_url
from retrieval.chunking import chunk_documents

# Step 1: Scrape documentation
url = "https://docs.python.org/3/tutorial/introduction.html"

document = scrape_url(url)

print("Original document length:", len(document.page_content))

# Step 2: Split into chunks
chunks = chunk_documents([document])

print("\nTotal chunks:", len(chunks))

# Step 3: Display the first chunk
print("\nFIRST CHUNK:")
print(chunks[0].page_content)

# Step 4: Display metadata
print("\nMETADATA:")
print(chunks[0].metadata)
