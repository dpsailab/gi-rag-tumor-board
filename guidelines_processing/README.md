# Dummy Guidelines PDF Extraction and JSON Generation

This repository contains scripts and dummy dictionaries to extract content from PDF guidelines (S3 and NCCN) and convert them into structured JSON format. The workflow is designed to support downstream processing, corpora selection, and manual curation.

You can find the guidelines at these sites:
- **S3 German Guidelines:** `https://www.leitlinienprogramm-onkologie.de/leitlinien/uebersicht` (`Ösophaguskarzinom`, `Magenkarzinom`, `Pankreaskarzinom`, `Kolorektales Karzinom`, and `HCC und biliäre Karzinome`)
- **NCCN Guidelines:** `https://www.nccn.org/guidelines/category_1` (`Esophageal and Esophagogastric Junction Cancers`, `Gastric Cancer`, `Pancreatic Adenocarcinoma`, `Colon Cancer`, `Rectal Cancer`, `Hepatocellular Carcinoma`, `Biliary Tract Cancers`)

## Step 1: Place PDFs guidelines in the correct folder

Place the PDFs of the German S3 and/or NCCN guidelines that you want to use in the folder → data/dummy_corpora/guidelines


## Step 2: Prepare the Dummy Dictionary

1. Open `data/dummy_corpora/guideline_dictionary_dummy.py`.
2. Replace the `"PATH OF YOUR PDF"` placeholders with the correct **relative paths** to your PDF files in the `data/dummy_corpora/guidelines` folder.
3. Make sure each guideline has the correct `starting_page`, `final_cleaned_chunk` (for S3), `mark` (heading marker, for S3), and `images_to_save` (pages with images to skip).

**Example:**

```python
guidelines_s3_dict = {
    'esophageal': {
        'path': "../data/dummy_corpora/guidelines/LL_Ösophaguskarzinom_Kurzversion_3.1.pdf",
        'starting_page': 21,
        'final_cleaned_chunk': "### 15 Tabellenverzeichnis",
        'mark': "### ",
        'images_to_save': [22, 29, 48],
    }
}
```


## Step 3: Extract JSON from PDFs

### S3 Guidelines

Given the structure of the German S3 Guidelines, we decided to create one chunk per chapters (determined by the specified mark).

Use:
```bash
pyton processing/extract_s3_guidelines_to_json.py
```
- Extracts text starting from starting_page and splits chunks using the specified mark.
- Uses starting_page and final_cleaned_chunk to remove trailing sections (e.g., accreditations, introduction, acknowledgements, references).
- Skips images_to_save pages.
- Output JSON files are saved in `data/dummy_corpora/`, with filenames based on the keys of guidelines_dict in `guideline_dictionary_dummy.py`, for example:

```text
    dummy_S3_guidelines_esophageal.json
    dummy_S3_guidelines_gastric.json
```

### NCCN Guidelines

Given the structure of the German S3 Guidelines, we decided to create one chunk per slide.

Use:
```bash
python processing/extract_nccn_guidelines_to_json.py
```

- Creates one chunk per page between starting_page and final_page, skipping images_to_save pages.
- Output JSON files are saved in `data/dummy_corpora/`, with filenames based on the keys of guidelines_nccn_dict in `guideline_dictionary_dummy.py`, for example:

```text
    dummy_NCCN_guidelines_esophageal.json
    dummy_NCCN_guidelines_gastric.json
```

PS: PS: Honestly, this step could be replaced by a simple copy-paste of the PDFs. Not elegant, but it would achieve the same goal.


## Step 4: Manual Curation

After generating the JSON files, manual curation is required:

- ### Images and Tables:
    
    Images should be rewritten in plain text. We usually enclose them in the tag `<chart>image description</chart>`.
    
    For the paper we convert every markdown table to a written JSON format. If this is absolutely necessary, it is to be questioned.
    
- ### Selected Corpora:
    
    Open the JSON files and set `"selected_corpora": 1` for chunks that should be included in the curated corpus.

    The default value is 0.

    **Practical Guidance:**
    
    1. **Full vs Curated Corpora**  
       - Two parallel corpora are maintained:
         1. **Full Corpus:** Contains the entire cleaned guideline text. Useful for comprehensive retrieval or reference.  
         2. **Curated Corpus:** Optimized for clinical decision support during tumor board meetings. Focuses only on actionable and treatment-relevant content.

    2. **Curation Criteria for the Tumor Board Corpus**  
       - Include only chunks that directly support treatment planning, staging, and surgical or systemic therapy decisions. This step must be performed with the support of board-certified oncologists or surgical oncologists.
       - Exclude repetitive or non-actionable sections, such as:  
         - Epidemiology or background statistics  
         - Prevention strategies  
         - General diagnostic criteria (unless they impact immediate therapeutic decisions)  

    3. **Inference Behavior**  
       - During RAG retrieval, the system will retrieve up to **five most relevant chunks** per query.  
       - Ensuring only actionable chunks are marked in the curated corpus reduces retrieval noise and improves relevance.

    **Summary Table:**

    | Corpus Type       | Contents                                                                                   | Selected Chunks (`selected_corpora = 1`) |
    |------------------|-------------------------------------------------------------------------------------------|-----------------------------------------|
    | Full Corpus       | Entire cleaned guideline text                                                              | Optional, for full reference            |
    | Curated Corpus    | Tumor board–optimized, actionable content only                                           | Mandatory for optimized retrieval       |
    
- ### Quality Check:
    
     Ensure text chunks are readable and headings are preserved where relevant.
    
     Remove any irrelevant footer or page numbering text if present.


## Step 5: Generate Embeddings

Once your curated JSON files are ready and cleaned:

Run the embedding script to compute vector embeddings for each guideline chunk:
```bash
python processing/corpora_embeddings.py
```
- Preset-Input: curated JSON files in data/dummy_corpora/ (e.g., dummy_guidelines.json)
- Preset-Output: JSON files with embeddings appended, e.g., dummy_guidelines_with_embeddings.json
  
Update the input and output filenames in `processing/corpora_embeddings.py` as required. These embeddings are required for local similarity search in the RAG pipeline.
When using processed guideline PDFs, keep input and output names identical to prevent conflicts when running `pipelines/framework_3_RAG.py`.

Note: If you try to run the RAG pipeline without embeddings, you may get a dimension mismatch error. Always run this step first.


## Notes

- chunk_id is automatically assigned:

```text
        S3 guidelines → start at 101
        NCCN guidelines → start at 501
```

- Ensure all paths in the dictionary are relative to the repository root.
- The mark variable is essential for S3 guidelines; NCCN guidelines usually work per-page without headings.
- The scripts rely on pymupdf4llm to read PDFs and convert them into markdown-like text.



# Workflow Summary

1. Place PDFs in the corresponding folders.
2. Update the dummy dictionary with correct PDF paths and page info.
3. Run S3 and/or NCCN extraction scripts.
4. Manually curate JSON outputs: update selected_corpora, rewrite images/tables, remove noise.
5. Use curated JSON with embeddings for downstream applications (e.g., RAG pipeline, NLP processing, etc.).


