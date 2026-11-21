
"""Manual scraper demo: run with ``python test_scraper.py``."""

from ingestion.scraper import scrape_url


def main() -> None:
    url = "https://docs.python.org/3/tutorial/introduction.html"
    document = scrape_url(url)

    print("\nTITLE:")
    print(document.metadata["title"])
    print("\nSOURCE:")
    print(document.metadata["source"])
    print("\nEXTRACTED CONTENT:")
    print(document.page_content[:2000])
    print("\nTOTAL CHARACTERS:", len(document.page_content))


if __name__ == "__main__":
    main()
