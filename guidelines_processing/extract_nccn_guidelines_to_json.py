import os
import sys
import json
import pymupdf
import pymupdf4llm


# Add the dummy_corpora folder to sys.path so we can import the dummy dictionary
current_dir = os.path.dirname(os.path.abspath(__file__))
dummy_corpora_dir = os.path.abspath(os.path.join(current_dir, '..', 'data', 'dummy_corpora'))
sys.path.append(dummy_corpora_dir)

from guideline_dictionary_dummy import guidelines_nccn_dict

# Output folder for JSON files
output_folder = os.path.abspath(os.path.join('..', 'data', 'dummy_corpora'))
os.makedirs(output_folder, exist_ok=True)

def remove_illegal_characters(text):
    """Keep only printable characters plus newline, carriage return, and tab."""
    return ''.join(c for c in text if c.isprintable() or c in "\n\r\t")

# Loop through NCCN dummy guidelines
for guideline_name, info in guidelines_nccn_dict.items():
    print(f"Processing NCCN guideline: {guideline_name}")

    pdf_path = os.path.abspath(os.path.join(current_dir, '..', 'data', 'dummy_corpora', info['path']))
    source_name = f"Synthetic {guideline_name} NCCN-like Guideline (Dummy)"

    # Load PDF as markdown docs
    reader = pymupdf4llm.LlamaMarkdownReader()
    try:
        docs = reader.load_data(pdf_path)
    except (FileNotFoundError, pymupdf.FileNotFoundError):
        print(f"\nERROR: PDF file not found: {pdf_path}")
        print("Please check that you inserted the correct path and filename in 'guideline_dictionary_dummy.py'.")
        print("Make sure the PDF exists in the folder 'data/dummy_corpora/guidelines'.\n")
        sys.exit(1)

    # Filter pages according to starting_page / final_page and skip images
    filtered_docs = [doc for i, doc in enumerate(docs)
                     if (i + 1) >= info['starting_page']
                     and (i + 1) <= info['final_page']
                     and (i + 1) not in info['images_to_save']]

    # Convert to JSON dicts
    json_data = []
    chunk_id_base = 500  # Start numbering for NCCN differently
    for i, doc in enumerate(filtered_docs):
        json_data.append({
            "chunk_id": chunk_id_base + i + 1,
            "source": source_name,
            "selected_corpora": 0,  # default 0
            "text": remove_illegal_characters(doc.text)
        })

    # Save JSON
    json_filename = f"dummy_NCCN_guidelines_{guideline_name.replace('-', '_')}.json"
    json_path = os.path.join(output_folder, json_filename)
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, ensure_ascii=False, indent=2)

    print(f"Saved {len(json_data)} chunks to {json_path}\n")
