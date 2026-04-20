import os
import sys
import json
import pymupdf
import pymupdf4llm

# Add the dummy_corpora folder to sys.path so we can import the dummy dictionary
current_dir = os.path.dirname(os.path.abspath(__file__))
dummy_corpora_dir = os.path.abspath(os.path.join(current_dir, '..', 'data', 'dummy_corpora'))
sys.path.append(dummy_corpora_dir)

from guideline_dictionary_dummy import guidelines_s3_dict

# Output folder for JSON files
output_folder = os.path.abspath(os.path.join('..', 'data', 'dummy_corpora'))
os.makedirs(output_folder, exist_ok=True)

def chunk_by_heading(text, heading_marker="### "):
    lines = text.splitlines()
    chunks = []
    current_chunk = []
    for line in lines:
        if line.startswith(heading_marker):
            if current_chunk:
                chunks.append("\n".join(current_chunk))
            current_chunk = [line]
        else:
            current_chunk.append(line)
    if current_chunk:
        chunks.append("\n".join(current_chunk))
    return chunks

def split_last_chunk(chunks, heading):
    if not chunks:
        return chunks
    last_chunk = chunks[-1]
    split_chunks = last_chunk.split(heading)
    new_chunks = chunks[:-1] + [c.strip() for c in split_chunks if c.strip()]
    return new_chunks[:-1]

# Loop through S3 dummy guidelines
for guideline_name, info in guidelines_s3_dict.items():
    print(f"Processing S3 guideline: {guideline_name}")

    pdf_path = os.path.abspath(os.path.join(current_dir, '..', 'data', 'dummy_corpora', info['path']))
    source_name = f"Synthetic {guideline_name} Guidelines (Dummy)"

    # Load PDF as markdown docs
    reader = pymupdf4llm.LlamaMarkdownReader()
    try:
        docs = reader.load_data(pdf_path)
    except (FileNotFoundError, pymupdf.FileNotFoundError):
        print(f"\nERROR: PDF file not found: {pdf_path}")
        print("Please check that you inserted the correct path and filename in 'guideline_dictionary_dummy.py'.")
        print("Make sure the PDF exists in the folder 'data/dummy_corpora/guidelines'.\n")
        sys.exit(1)

    # Combine all text from the starting page
    all_text = [doc.text for doc in docs[info['starting_page']-1:]]  # 0-based index
    joined_text = "\n".join(all_text)

    # Chunk by heading
    chunks = chunk_by_heading(joined_text, info['mark'])
    final_chunks = split_last_chunk(chunks, info['final_cleaned_chunk'])

    # Create list of dicts for JSON
    json_data = []
    chunk_id_base = 100  # Start numbering at 101
    for i, chunk_text in enumerate(final_chunks):
        json_data.append({
            "chunk_id": chunk_id_base + i + 1,
            "source": source_name,
            "selected_corpora": 0,  # default 0, can be updated manually later
            "text": chunk_text.strip()
        })

    # Save JSON
    json_filename = f"dummy_S3_guidelines_{guideline_name.replace('-', '_')}.json"
    json_path = os.path.join(output_folder, json_filename)
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, ensure_ascii=False, indent=2)

    print(f"Saved {len(json_data)} chunks to {json_path}\n")
