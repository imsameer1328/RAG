
import requests
from bs4 import BeautifulSoup
from langchain_core.documents import Document
from urllib.parse import urljoin, urlparse, urldefrag
from collections import deque


HEADERS = {
    "User-Agent": "DocuMindAI/1.0 (documentation research tool)"
}


def scrape_url(url: str) -> Document:
    """Scrape one webpage and return it as a LangChain Document."""

    parsed_url = urlparse(url)

    if parsed_url.scheme not in ("http", "https") or not parsed_url.netloc:
        raise ValueError("Please provide a valid HTTP or HTTPS URL.")

    try:
        response = requests.get(
            url,
            headers=HEADERS,
            timeout=20
        )
        response.raise_for_status()

    except requests.RequestException as error:
        raise RuntimeError(f"Could not fetch {url}: {error}") from error

    soup = BeautifulSoup(response.text, "html.parser")

    title = soup.title.get_text(strip=True) if soup.title else "Untitled page"

    for element in soup([
        "script", "style", "nav", "footer",
        "header", "noscript"
    ]):
        element.decompose()

    content = soup.find("main") or soup.find("article") or soup.body

    if content is None:
        raise ValueError(f"No readable content found at {url}")

    text = content.get_text(separator="\n", strip=True)

    if not text:
        raise ValueError(f"No readable text found at {url}")

    return Document(
        page_content=text,
        metadata={
            "source": url,
            "title": title
        }
    )


def scrape_website(start_url: str, max_pages: int = 20):
    """
    Crawl pages on the same website, starting from start_url.

    Returns a list of LangChain Documents.
    """

    parsed_start = urlparse(start_url)

    if parsed_start.scheme not in ("http", "https") or not parsed_start.netloc:
        raise ValueError("Please provide a valid HTTP or HTTPS URL.")

    base_domain = parsed_start.netloc.lower()

    # Queue stores URLs waiting to be scraped
    queue = deque([start_url])

    # Prevent visiting the same URL multiple times
    visited = set()

    documents = []

    while queue and len(documents) < max_pages:

        current_url = queue.popleft()

        # Remove URL fragments
        current_url, _ = urldefrag(current_url)

        if current_url in visited:
            continue

        visited.add(current_url)

        try:
            response = requests.get(
                current_url,
                headers=HEADERS,
                timeout=20
            )
            response.raise_for_status()

            # Only process HTML pages
            content_type = response.headers.get("Content-Type", "")

            if "text/html" not in content_type:
                continue

            soup = BeautifulSoup(response.text, "html.parser")

            title = (
                soup.title.get_text(strip=True)
                if soup.title
                else "Untitled page"
            )

            # Find links before removing navigation elements
            for link in soup.find_all("a", href=True):

                absolute_url = urljoin(current_url, link["href"])
                absolute_url, _ = urldefrag(absolute_url)

                parsed_link = urlparse(absolute_url)

                # Only follow links on the same domain
                if parsed_link.netloc.lower() != base_domain:
                    continue

                if parsed_link.scheme not in ("http", "https"):
                    continue

                # Skip common non-documentation files
                if parsed_link.path.lower().endswith((
                    ".pdf", ".png", ".jpg", ".jpeg",
                    ".gif", ".svg", ".zip", ".css",
                    ".js", ".xml"
                )):
                    continue

                if absolute_url not in visited:
                    queue.append(absolute_url)

            # Remove unnecessary HTML elements
            for element in soup([
                "script", "style", "nav", "footer",
                "header", "noscript"
            ]):
                element.decompose()

            content = soup.find("main") or soup.find("article") or soup.body

            if content is None:
                continue

            text = content.get_text(separator="\n", strip=True)

            if not text:
                continue

            documents.append(
                Document(
                    page_content=text,
                    metadata={
                        "source": current_url,
                        "title": title
                    }
                )
            )

            print(f"Scraped ({len(documents)}/{max_pages}): {current_url}")

        except requests.RequestException as error:
            print(f"Skipping {current_url}: {error}")

    print(f"\nCrawling complete. Pages scraped: {len(documents)}")

    return documents
