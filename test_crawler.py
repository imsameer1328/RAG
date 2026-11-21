from ingestion.scraper import scrape_website

url = "https://docs.python.org/3/tutorial/"

documents = scrape_website(
    start_url=url,
    max_pages=10
)

print("\nTOTAL PAGES:", len(documents))

for index, document in enumerate(documents, start=1):

    print(f"\nPAGE {index}")
    print("TITLE:", document.metadata["title"])
    print("SOURCE:", document.metadata["source"])
    print("CONTENT LENGTH:", len(document.page_content))
    print("PREVIEW:", document.page_content[:200])
