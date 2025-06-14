import fitz  # PyMuPDF
import json
import re
from tabulate import tabulate
import pandas as pd
import docx  # python-docx
import os
import glob
from Levenshtein import ratio as levenshtein_ratio

# --- Post-Processing & Cleaning Configuration ---
MIN_CHUNK_WORD_COUNT = 10  # Min words for a chunk to be considered meaningful
SIMILARITY_THRESHOLD = 0.95  # For near-duplicate detection (95% similar)

# Regex patterns to identify and remove boilerplate/disclaimer text
BOILERPLATE_PATTERNS = [
    re.compile(r'^\s*page\s*\d+\s*of\s*\d+\s*$', re.IGNORECASE),
    re.compile(r'^\s*\d+\s*$', re.IGNORECASE), # Remove chunks that are only numbers
    re.compile(r'copyright\s*©\s*\d{4}', re.IGNORECASE),
    re.compile(r'all\s*rights\s*reserved', re.IGNORECASE),
    re.compile(r'confidential\s*and\s*proprietary', re.IGNORECASE),
    re.compile(r'for\s*internal\s*use\s*only', re.IGNORECASE)
]

def is_boilerplate(text):
    """Checks if a text chunk matches any predefined boilerplate patterns."""
    for pattern in BOILERPLATE_PATTERNS:
        if pattern.search(text):
            return True
    return False

def filter_and_clean_chunks(chunks):
    """
    Applies a series of cleaning steps to the extracted chunks:
    1. Removes short and boilerplate chunks.
    2. Removes exact and near-duplicate chunks.
    """
    # Step 1: Filter out short and boilerplate chunks
    print(f"Initial chunk count: {len(chunks)}")
    filtered_chunks = []
    for chunk in chunks:
        content = chunk['content']
        word_count = len(content.split())
        
        if word_count < MIN_CHUNK_WORD_COUNT:
            continue
            
        if is_boilerplate(content):
            continue
            
        filtered_chunks.append(chunk)
    
    print(f"Chunks after filtering short/boilerplate: {len(filtered_chunks)}")

    # Step 2: Remove exact and near-duplicates
    unique_chunks = []
    seen_contents = set()

    for chunk in filtered_chunks:
        content = chunk['content']
        # Normalize whitespace for more accurate duplicate checking
        normalized_content = ' '.join(content.split())
        
        # Skip exact duplicates
        if normalized_content in seen_contents:
            continue
        
        # Check for near-duplicates against already added unique chunks
        is_duplicate = False
        for unique_chunk in unique_chunks:
            if levenshtein_ratio(normalized_content, ' '.join(unique_chunk['content'].split())) > SIMILARITY_THRESHOLD:
                is_duplicate = True
                break
        
        if not is_duplicate:
            unique_chunks.append(chunk)
            seen_contents.add(normalized_content)

    print(f"Chunks after removing duplicates: {len(unique_chunks)}")
    return unique_chunks

def extract_and_preprocess_pdf(pdf_path, health_plan_name):
    """
    Performs comprehensive data preprocessing on a single PDF document.
    Processes only the first 5 pages.
    """
    processed_chunks = []
    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        print(f"Error opening or processing {pdf_path}: {e}")
        return []

    for page_num, page in enumerate(doc):
        if page_num >= 5:
            print(f"Stopping at page 5 for {os.path.basename(pdf_path)}.")
            break

        table_areas = []
        try:
            tables = page.find_tables()
            for table in tables:
                bbox = table.bbox
                table_areas.append(bbox)
                table_data = table.extract()
                if not table_data or len(table_data) <= 1:
                    continue
                
                headers = [str(h).strip() if h is not None else "" for h in table_data[0]]
                for r_idx, row in enumerate(table_data[1:]):
                    row_str_parts = [f"{h}: {str(c).strip()}" for h, c in zip(headers, row) if h and c and str(c).strip()]
                    if not row_str_parts:
                        continue
                    row_chunk_content = ". ".join(row_str_parts) + "."
                    processed_chunks.append({
                        "content": row_chunk_content,
                        "metadata": {"source_file": os.path.basename(pdf_path), "page_number": page_num + 1, "health_plan_name": health_plan_name, "chunk_type": "table_row"}
                    })
        except Exception as e:
            print(f"Error processing tables on page {page_num+1} of {pdf_path}: {e}")

        blocks = page.get_text("blocks")
        for block in blocks:
            x0, y0, x1, y1, text, _, block_type = block
            if block_type == 1 or not text.strip():
                continue
            if any(x0 >= t[0] and y0 >= t[1] and x1 <= t[2] and y1 <= t[3] for t in table_areas):
                continue
            cleaned_text = re.sub(r'\s+', ' ', text).strip()
            processed_chunks.append({
                "content": cleaned_text,
                "metadata": {"source_file": os.path.basename(pdf_path), "page_number": page_num + 1, "health_plan_name": health_plan_name, "chunk_type": "paragraph"}
            })
    doc.close()
    return processed_chunks

def extract_and_preprocess_docx(docx_path):
    """
    Extracts text from a DOCX file, chunks it by paragraph.
    """
    processed_chunks = []
    try:
        doc = docx.Document(docx_path)
        for para in doc.paragraphs:
            cleaned_text = re.sub(r'\s+', ' ', para.text).strip()
            if cleaned_text:
                processed_chunks.append({
                    "content": cleaned_text,
                    "metadata": {"source_file": os.path.basename(docx_path), "purpose": "User Eligibility Check", "chunk_type": "paragraph"}
                })
    except Exception as e:
        print(f"Error processing DOCX file {docx_path}: {e}")
    return processed_chunks

def main():
    """
    Main function to process all PDF and DOCX files in a directory,
    and apply advanced cleaning and filtering.
    """
    input_directory = "D:\Alltius_Interview_Assignment\Insurance PDFs"
    output_directory = "D:\Alltius_Interview_Assignment\jsons_from_docs"
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    all_pdf_chunks, all_docx_chunks = [], []

    # Process PDFs
    pdf_files = glob.glob(os.path.join(input_directory, '*.pdf'))
    print(f"\nFound {len(pdf_files)} PDF files to process...")
    for pdf_path in pdf_files:
        print(f"--- Processing PDF: {os.path.basename(pdf_path)} ---")
        plan_name = os.path.basename(pdf_path).replace('_', ' ').replace('.pdf', '')
        pdf_chunks = extract_and_preprocess_pdf(pdf_path, plan_name)
        if pdf_chunks:
            all_pdf_chunks.extend(pdf_chunks)
    
    # Process DOCX
    docx_files = glob.glob(os.path.join(input_directory, '*.docx'))
    print(f"\nFound {len(docx_files)} DOCX files to process...")
    for docx_path in docx_files:
        print(f"--- Processing DOCX: {os.path.basename(docx_path)} ---")
        docx_chunks = extract_and_preprocess_docx(docx_path)
        if docx_chunks:
            all_docx_chunks.extend(docx_chunks)

    # --- Clean and Save PDF Chunks ---
    if all_pdf_chunks:
        print("\n--- Cleaning all extracted PDF chunks ---")
        cleaned_pdf_chunks = filter_and_clean_chunks(all_pdf_chunks)
        pdf_output_path = os.path.join(output_directory, "all_pdf_chunks_cleaned.json")
        with open(pdf_output_path, 'w', encoding='utf-8') as f:
            json.dump(cleaned_pdf_chunks, f, ensure_ascii=False, indent=2)
        print(f"\nSuccessfully saved {len(cleaned_pdf_chunks)} cleaned PDF chunks to: {pdf_output_path}")
    else:
        print("\nNo PDF chunks were generated.")

    # --- Clean and Save DOCX Chunks ---
    if all_docx_chunks:
        print("\n--- Cleaning all extracted DOCX chunks ---")
        cleaned_docx_chunks = filter_and_clean_chunks(all_docx_chunks)
        docx_output_path = os.path.join(output_directory, "all_docx_chunks_cleaned.json")
        with open(docx_output_path, 'w', encoding='utf-8') as f:
            json.dump(cleaned_docx_chunks, f, ensure_ascii=False, indent=2)
        print(f"\nSuccessfully saved {len(cleaned_docx_chunks)} cleaned DOCX chunks to: {docx_output_path}")
    else:
        print("\nNo DOCX chunks were generated.")

if __name__ == '__main__':
    # Make sure you have the required library for near-duplicate detection
    # pip install python-Levenshtein
    main()