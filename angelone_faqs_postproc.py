import json
import re
import os
from fuzzywuzzy import fuzz

# --- Helper Functions (re-used and slightly adapted for web context) ---

def load_chunks_from_json(file_path):
    """Loads chunks from a JSON file."""
    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        return []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            chunks = json.load(f)
        print(f"Loaded {len(chunks)} chunks from {file_path}")
        return chunks
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from {file_path}: {e}")
        return []
    except Exception as e:
        print(f"An unexpected error occurred while loading {file_path}: {e}")
        return []

def clean_text_chunk(text):
    """Applies common text cleaning and normalization."""
    # Remove common web artifacts like multiple spaces, tabs, newlines
    cleaned_text = re.sub(r'\s+', ' ', text).strip()
    return cleaned_text

# --- Specific Irrelevance Discernment Patterns for Web-Scraped FAQs ---
# These patterns target common elements found on website support/FAQ pages
# that are typically not part of the core answer content.
FAQ_IRRELEVANT_PATTERNS = [
    # Feedback prompts
    r"Was this (article|answer|information) helpful\?",
    r"Yes\s*[/\\-]\s*No",
    r"Rate this article",
    r"Give us feedback",
    r"Help us improve",

    # Contact/Support prompts
    r"Still have questions\?",
    r"Contact (support|us|customer service)",
    r"Reach out to us",
    r"Our team is ready to help",

    # Related articles/Navigation
    r"Related (articles|topics|questions)",
    r"More like this",
    r"You might also like",
    r"Previous Article",
    r"Next Article",
    r"Back to (top|home)",
    r"Go to (top|home)",
    r"View all (articles|FAQs|topics)",
    r"Browse (all|categories)",
    r"Table of Contents",

    # Social sharing
    r"Share this (article|page)",
    r"(Facebook|Twitter|LinkedIn|WhatsApp|Email|Print|Copy Link)", # Social media/share icons' text

    # Generic website footers/headers (can appear on FAQs)
    r"^\s*Copyright\s+\W*\d{4}.*$", # e.g., "Copyright © 2024"
    r"All rights reserved\.",
    r"Privacy Policy",
    r"Terms of Use",
    r"Disclaimer",
    r"Legal Notice",
    r"Cookie Policy",
    r"Sitemap",
    r"Website developed by",
    r"Powered by \w+",
    r"Contact Us",
    r"About Us",
    r"Careers",
    r"Support",
    r"Customer Service",
    r"Follow Us on",
    r"Connect with us",
    r"Visit our blog",
    r"Subscribe to our newsletter",
    r"Sign up for updates",
    r"Get Started",
    r"Learn More",
    r"Download our app",
    r"Mobile App",
    r"Download on the App Store",
    r"Get it on Google Play",
    r"©\s*\d{4}\s*[-–]\s*\d{4}",
    r"Return to homepage",
    r"Read more",
    r"Continue reading",
    r"See more",
    r"Skip to content",
    r"Top of Page",
    r"Back to main content",

    # Common article metadata (if not already stripped by scraper)
    r"Published on:",
    r"Author:",
    r"Posted by:",
    r"Last updated:",
    r"Last modified:",
    r"Effective date:",
    r"version \d+(\.\d+)*",
]


# --- Deduplication and Filtering Functions (re-used) ---

def remove_exact_duplicates(chunks):
    """Removes chunks with exact duplicate 'content'."""
    unique_contents = set()
    filtered_chunks = []
    duplicate_count = 0
    for chunk in chunks:
        content = chunk.get('content')
        if content is None: continue
        if content not in unique_contents:
            unique_contents.add(content)
            filtered_chunks.append(chunk)
        else:
            duplicate_count += 1
    print(f"Removed {duplicate_count} exact duplicate chunks.")
    return filtered_chunks

def remove_short_chunks(chunks, min_word_count=50):
    """Removes chunks whose 'content' has fewer than min_word_count words."""
    filtered_chunks = []
    short_chunk_count = 0
    for chunk in chunks:
        content = chunk.get('content')
        if content is None: continue
        word_count = len(content.split())
        if word_count >= min_word_count:
            filtered_chunks.append(chunk)
        else:
            short_chunk_count += 1
    print(f"Removed {short_chunk_count} short chunks (less than {min_word_count} words).")
    return filtered_chunks

def remove_irrelevant_chunks_by_patterns(chunks, patterns):
    """
    Removes chunks whose 'content' matches any of the provided patterns.
    This replaces the previous 'remove_boilerplate_chunks'.
    """
    filtered_chunks = []
    irrelevant_chunk_count = 0

    compiled_patterns = [re.compile(pattern, re.IGNORECASE | re.DOTALL) for pattern in patterns]

    for chunk in chunks:
        content = chunk.get('content')
        if content is None:
            continue

        is_irrelevant = False
        for pattern in compiled_patterns:
            if pattern.search(content):
                is_irrelevant = True
                break
        
        if not is_irrelevant:
            filtered_chunks.append(chunk)
        else:
            irrelevant_chunk_count += 1
            # Optional: print(f"Removed irrelevant chunk: {content[:50]}...")
    print(f"Removed {irrelevant_chunk_count} chunks matching irrelevant (FAQ-specific) patterns.")
    return filtered_chunks

def remove_near_duplicates_levenshtein(chunks, similarity_threshold=90):
    """
    Removes near-duplicate chunks based on Levenshtein distance (fuzzywuzzy ratio).
    Keeps the first occurrence of semantically unique content.
    """
    if not chunks: return []
    
    content_with_indices = [(i, chunk['content']) for i, chunk in enumerate(chunks) if chunk.get('content')]
    is_duplicate = [False] * len(chunks)
    near_duplicate_count = 0

    print(f"Starting near-duplicate detection using Levenshtein distance (threshold: {similarity_threshold}%).")

    for i in range(len(content_with_indices)):
        original_idx_i, content_i = content_with_indices[i]
        if is_duplicate[original_idx_i]: continue

        for j in range(i + 1, len(content_with_indices)):
            original_idx_j, content_j = content_with_indices[j]
            if is_duplicate[original_idx_j]: continue

            similarity = fuzz.ratio(content_i, content_j)
            
            if similarity >= similarity_threshold:
                is_duplicate[original_idx_j] = True
                near_duplicate_count += 1

    filtered_chunks = [chunk for i, chunk in enumerate(chunks) if not is_duplicate[i]]
    print(f"Removed {near_duplicate_count} near-duplicate chunks based on Levenshtein distance.")
    return filtered_chunks


# --- Main Execution Workflow for FAQ Cleaning ---

if __name__ == "__main__":
    # --- Configuration ---
    # Path to your input JSON file containing scraped FAQ chunks
    # Example: This would be the output from your Playwright scraping script,
    # e.g., "angelone_faqs_data/angelone_faqs_chunks.json"
    input_faq_json_file = r"D:\Alltius_Interview_Assignment\angelone_scraped_data\angelone_faqs_chunks.json" # Adjust this path

    # Path for the cleaned output JSON file
    cleaned_faq_json_file = r"D:\Alltius_Interview_Assignment\angelone_scraped_data\cleaned_angelone_faqs_chunks.json"

    # Filter parameters
    MIN_WORD_COUNT = 50 # Minimum words for a chunk to be considered meaningful
    LEVENSHTEIN_SIMILARITY_THRESHOLD = 90 # % similarity for near-duplicate (0-100)

    print("\n--- Starting FAQ Data Cleaning ---")
    
    # 1. Load the FAQ chunks
    faq_chunks = load_chunks_from_json(input_faq_json_file)

    if faq_chunks:
        # 2. Remove exact duplicates
        cleaned_chunks_step1 = remove_exact_duplicates(faq_chunks)
        
        # 3. Remove short chunks
        cleaned_chunks_step2 = remove_short_chunks(cleaned_chunks_step1, MIN_WORD_COUNT)
        
        # 4. Remove irrelevant chunks using FAQ-specific patterns
        cleaned_chunks_step3 = remove_irrelevant_chunks_by_patterns(cleaned_chunks_step2, FAQ_IRRELEVANT_PATTERNS)
        
        # 5. Remove near-duplicates using Levenshtein distance
        final_cleaned_chunks = remove_near_duplicates_levenshtein(cleaned_chunks_step3, LEVENSHTEIN_SIMILARITY_THRESHOLD)

        print(f"\nTotal final cleaned FAQ chunks: {len(final_cleaned_chunks)}")

        # Save final cleaned chunks
        with open(cleaned_faq_json_file, 'w', encoding='utf-8') as f:
            json.dump(final_cleaned_chunks, f, ensure_ascii=False, indent=2)
        print(f"Final cleaned FAQ chunks saved to {cleaned_faq_json_file}")
    else:
        print("No FAQ chunks to process. Please ensure your input JSON file exists and contains data.")

    print("\n--- FAQ Data Cleaning Complete ---")