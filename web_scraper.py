import asyncio
from playwright.async_api import async_playwright
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import re
import json
import os
import time


def clean_text_chunk(text):
    """Applies common text cleaning and normalization."""
    cleaned_text = re.sub(r'\s+', ' ', text).strip()
    return cleaned_text

def get_base_filename(url):
    """Generates a cleaned filename from a URL for saving purposes."""
    path = urlparse(url).path
    filename = path.strip('/').replace('/', '_')
    if not filename:
        return "index"
    return filename

async def scrape_and_preprocess_angelone_faqs(start_url_prefix, output_directory="angelone_faqs_data", delay_seconds=1.5):
    """
    Recursively scrapes and preprocesses Angel One support FAQ pages using Playwright,
    handling JavaScript-loaded content. It extracts Q&A pairs and general paragraphs,
    and stores them as structured chunks.

    Args:
        start_url_prefix (str): The base URL prefix to start crawling from and adhere to.
                                 e.g., "https://www.angelone.in/support/"
        output_directory (str): Directory to save the processed JSON chunks.
        delay_seconds (float): Delay in seconds between page visits to be polite.

    Returns:
        list: A list of dictionaries, where each dictionary represents a preprocessed chunk.
    """
    processed_chunks = []
    visited_urls = set()
    urls_to_visit = asyncio.Queue() # Use async queue for Playwright
    
    # Ensure the prefix ends with a slash for consistent comparison
    if not start_url_prefix.endswith('/'):
        start_url_prefix += '/'

    # Initial URL check and add to queue
    parsed_start_url = urlparse(start_url_prefix)
    normalized_start_url = parsed_start_url.scheme + "://" + parsed_start_url.netloc + parsed_start_url.path
    if "/hindi/" not in normalized_start_url: # Exclude Hindi pages from start
        await urls_to_visit.put(normalized_start_url)
    else:
        print(f"Skipping initial Hindi URL: {normalized_start_url}")
        return []

    print(f"Starting web scrape from: {normalized_start_url} (excluding /hindi/ paths)")

    os.makedirs(output_directory, exist_ok=True)

    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()

        while not urls_to_visit.empty():
            current_url = await urls_to_visit.get()

            parsed_current_url = urlparse(current_url)
            normalized_current_url = parsed_current_url.scheme + "://" + parsed_current_url.netloc + parsed_current_url.path

            if normalized_current_url in visited_urls:
                continue

            # Skip Hindi pages and pages outside the prefix
            if "/hindi/" in normalized_current_url:
                print(f"Skipping Hindi page: {normalized_current_url}")
                continue
            if not normalized_current_url.startswith(start_url_prefix):
                # This could happen if a non-prefix internal link was discovered but not filtered early
                print(f"Skipping URL outside prefix: {normalized_current_url}")
                continue
                
            visited_urls.add(normalized_current_url)
            print(f"Visiting: {normalized_current_url}")

            try:
                # Navigate and wait for network activity to be idle
                # This helps ensure JavaScript content has loaded
                await page.goto(normalized_current_url, wait_until="networkidle")
                

                html_content = await page.content() # Get the full HTML content after JS execution

                soup = BeautifulSoup(html_content, 'html.parser')

                # --- B. For Angel One Support FAQs (Web-scraped Data) ---
                # 1. HTML Parsing & Cleaning (using BeautifulSoup)
                # Identify main content area. This is highly site-specific.
                # Inspect Angel One's support page HTML to find appropriate selectors.
                
                main_content_div = soup.find('div', class_='article-content') # Placeholder selector
                if not main_content_div:
                    main_content_div = soup.find('main') # Fallback
                if not main_content_div:
                    main_content_div = soup.find('body') # Last resort

                if not main_content_div:
                    print(f"Warning: Could not find main content for {normalized_current_url}. Skipping content extraction.")
                    continue

                # Remove common noisy elements like nav, footer, script, style tags
                for tag in main_content_div(['nav', 'footer', 'script', 'style', 'header', 'aside', 'form', 'img']):
                    tag.decompose() # Remove the tag and its content

                # 2. Question-Answer Pair Extraction & 5. Chunking (Paragraph for Q&A)
                # This depends heavily on how Q&A are marked up on Angel One.
                
                # Strategy: Look for potential FAQ pairs or just chunk by paragraphs/headings
                current_section_title = None
                for element in main_content_div.find_all(['h1', 'h2', 'h3', 'h4', 'p', 'li', 'strong']):
                    text_content = clean_text_chunk(element.get_text())
                    if not text_content:
                        continue

                    chunk_type = "paragraph"
                    
                    if element.name in ['h1', 'h2', 'h3', 'h4']:
                        current_section_title = text_content
                        chunk_type = "heading"
                    elif element.name == 'strong' and len(text_content.split()) < 20: # Heuristic for potential question
                         # If it's a strong tag that could be a question, check next sibling
                        next_sibling_text = ""
                        next_sibling = element.find_next_sibling()
                        if next_sibling and next_sibling.name in ['p', 'div', 'ul', 'ol']:
                            next_sibling_text = clean_text_chunk(next_sibling.get_text())
                            if len(next_sibling_text.split()) > 10: # Ensure answer is substantial
                                processed_chunks.append({
                                    "content": f"Question: {text_content}\nAnswer: {next_sibling_text}",
                                    "metadata": {
                                        "source_url": normalized_current_url,
                                        "source_filename": get_base_filename(normalized_current_url),
                                        "page_title": soup.title.string if soup.title else "No Title",
                                        "section_title": current_section_title,
                                        "chunk_type": "qa_pair"
                                    }
                                })
                                # Skip processing the next sibling as it's part of this QA chunk
                                continue # For this simplified example, just process element by element
                        
                    # Default: process as a general paragraph chunk
                    if len(text_content.split()) > 10: # Length restriction for paragraph chunks
                        processed_chunks.append({
                            "content": text_content,
                            "metadata": {
                                "source_url": normalized_current_url,
                                "source_filename": get_base_filename(normalized_current_url),
                                "page_title": soup.title.string if soup.title else "No Title",
                                "section_title": current_section_title,
                                "chunk_type": chunk_type
                            }
                        })

                # --- Extract and Queue New Links ---
                for link in soup.find_all('a', href=True):
                    href = link['href']
                    absolute_url = urljoin(normalized_current_url, href)

                    parsed_absolute_url = urlparse(absolute_url)
                    normalized_absolute_url = parsed_absolute_url.scheme + "://" + parsed_absolute_url.netloc + parsed_absolute_url.path

                    # Filter for desired prefix and exclude Hindi pages
                    if normalized_absolute_url.startswith(start_url_prefix) \
                       and "/hindi/" not in normalized_absolute_url \
                       and normalized_absolute_url not in visited_urls:
                        # Avoid adding duplicates to queue, although visited_urls check will catch it.
                        await urls_to_visit.put(normalized_absolute_url) # Add to queue for next visit

            except Exception as e:
                print(f"Error processing {normalized_current_url}: {e}")
            
            await asyncio.sleep(delay_seconds) # Be polite

        await browser.close()
    
    # Save all processed chunks to a single JSON file
    output_path = os.path.join(output_directory, "angelone_faqs_chunks.json")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(processed_chunks, f, ensure_ascii=False, indent=2)
    print(f"\nSaved {len(processed_chunks)} chunks to {output_path}")

    return processed_chunks

# --- Example Usage ---
if __name__ == "__main__":
    angelone_support_url = "https://www.angelone.in/support/"
    output_dir = "angelone_scraped_data"

    # Run the asynchronous scraping function
    asyncio.run(scrape_and_preprocess_angelone_faqs(angelone_support_url, output_dir, delay_seconds=1.5))