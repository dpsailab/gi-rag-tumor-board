# Prompt Examples

This document illustrates how prompts were formulated across different configurations in our study. It is intended for transparency and methodological reproducibility.
Examples are based on a representative esophageal cancer case and demonstrate the four main prompt strategies used.


## Overview of Prompt Types

| Prompt Type | Purpose | Retrieval | Language Output |
|-------------|---------|-----------|-----------------|
| Rewriting | Convert German case to standardized English | None | English |
| Simple Request | Direct model query without guidelines | None | German |
| Assistant | Query with full guideline PDFs attached | ChatGPT Assistant | German |
| Custom RAG | Query with retrieved guideline chunks | Custom RAG system | German |

---

## 1. Rewriting Prompt

**Purpose:** Convert the original German tumor board case into a standardized, guideline-aligned English format for retrieval and model input.

**Prompt Template:**
```
I would like to use the following clinical case in a Retrieval-Augmented Generation (RAG) workflow, referencing oncological guidelines to determine the next therapeutic step.

Please rewrite the case in a concise and standardized format, using clinical terminology and structure typically found in oncology guidelines.

Return only the reformulated case, without additional commentary.

#########

Clinical case: [ORIGINAL CASE]
```

**Usage:** This prompt is sent to GPT-4o-mini to translate and restructure German tumor board reports into English text aligned with guideline terminology.

**Example Input:**
```
Clinical case: 64-jähriger Patient mit Adenokarzinom des Ösophagus, cT3 cN1 M0, Z.n. neoadjuvanter Radiochemotherapie...
```

**Example Output:**
```
64-year-old male patient with esophageal adenocarcinoma, staged cT3 cN1 M0. Prior treatment includes neoadjuvant chemoradiotherapy with carboplatin/paclitaxel...
```

---

## 2. Simple Request (No Retrieval)

**Purpose:** Direct query to the model without guideline retrieval - baseline configuration.

**Prompt Template:**
```
You are a multidisciplinary oncological board with surgeons, oncologists, radiologists, radiation oncologists, and pathologists. Determine the next therapeutic step for the patient. Patient details are enclosed within triple single quotation marks (''' ''').

If a clinical question (‘Fragestellung’)  is present, answer precisely; otherwise, respond based on expertise. Do not suggest referral or further discussion. Answer in German, 1–2 sentences, starting with 'Das Board empfiehlt'.

### Actual case:

'''[ORIGINAL CASE or REWRITTEN CASE]'''
```

**Usage:** This baseline prompt relies entirely on the model's training data without external guideline retrieval.

**Note:** `[ORIGINAL CASE]` = German tumor board report; `[REWRITTEN CASE]` = English standardized version from Prompt Type 1.

---

## 3. ChatGPT Assistant Prompt

**Purpose:** Query using the ChatGPT Assistant interface with full guideline PDFs uploaded to the assistant environment.

**Prompt Template:**
```
You are a multidisciplinary oncological board (surgeons, oncologists, radiologists, radiation oncologists, pathologists). Determine the next therapeutic step based on patient details and attached guidelines.

Patient details under 'Actual Case' (''' '''). Analyze the case in the context of the guidelines (attached files) and provide a concise German recommendation in 1–2 sentences, starting with 'Das Board empfiehlt'.

### Actual case:

'''[ORIGINAL CASE or REWRITTEN CASE]'''
```

**Usage:** This prompt is used with the OpenAI Assistant API, where the specific tumor-type guideline PDFs are pre-uploaded to the assistant's file storage. The assistant automatically retrieves relevant sections.

**Attached Guidelines:**
- German S3 Guidelines (Leitlinienprogramm Onkologie) - complete PDFs
- NCCN Clinical Practice Guidelines - complete PDFs

---

## 4. Custom RAG Prompt (Full or Selected Corpora)

**Purpose:** Retrieval-augmented prompt including guideline excerpts retrieved by our custom RAG system.

**Prompt Template:**
```
You are a multidisciplinary oncological board (surgeons, oncologists, radiologists, radiation oncologists, pathologists). Determine the next therapeutic step for the patient based on:

1. Patient clinical details under 'Actual Case' (''' ''')
2. Relevant guideline excerpts under 'Context' (''' ''')

Formulate a concise German recommendation in 1–2 sentences, starting with 'Das Board empfiehlt'.

### Actual case:

'''[ORIGINAL CASE or REWRITTEN CASE]'''

### Context:

'''[RETRIEVED GUIDELINE CHUNKS]'''
```

**Example with Retrieved Context:**
```
### Context [example from REWRITTEN CASE]:

'''
- Chunk 153 - NCCN Guidelines: Preoperative Chemoradiation for esophageal adenocarcinoma
  Patients with locally advanced esophageal adenocarcinoma (cT2-4a, N0-3, M0) should receive neoadjuvant chemoradiotherapy followed by surgery...

- Chunk 157 - NCCN Guidelines: Follow-up and surveillance
  Post-treatment surveillance includes clinical examination every 3-6 months for 1-2 years, then every 6-12 months...

- Chunk 151 - NCCN Guidelines: Tumor classification and primary treatment options
  Clinical staging determines treatment approach: early-stage (T1a) may be suitable for endoscopic resection...

- Chunk 42 - German S3 Guideline: Surgical principles and preoperative staging
  Complete surgical resection (R0) is the primary curative treatment. Preoperative staging must include EUS and CT...

- Chunk 83 - NCCN Guidelines: Survivorship considerations
  Long-term follow-up should address nutritional status, anastomotic strictures, and surveillance for recurrence...
'''
```

**Usage:** Our custom RAG system retrieves top-5 most relevant chunks based on semantic similarity (cosine distance using BAAI/bge-m3 embeddings). Retrieved chunks are inserted into the Context section before model execution.

**Corpus Variants:**
- **Full Corpora:** Complete guideline text including background, epidemiology, diagnostics
- **Selected Corpora:** Curated guidelines excluding non-treatment sections (epidemiology, prevention, basic diagnostics)

---

## Configuration Matrix

The 16 experimental configurations combine:

| Configuration | Input Type | Retrieval Method | Model |
|--------------|------------|------------------|-------|
| 1 | Original | No Retrieval | GPT-4o-mini |
| 2 | Original | No Retrieval | GPT-4o |
| 3 | Original | Assistant | GPT-4o-mini |
| 4 | Original | Assistant | GPT-4o |
| 5 | Original | RAG Full | GPT-4o-mini |
| 6 | Original | RAG Full | GPT-4o |
| 7 | Original | RAG Selected | GPT-4o-mini |
| 8 | Original | RAG Selected | GPT-4o |
| 9 | Rewritten | No Retrieval | GPT-4o-mini |
| 10 | Rewritten | No Retrieval | GPT-4o |
| 11 | Rewritten | Assistant | GPT-4o-mini |
| 12 | Rewritten | Assistant | GPT-4o |
| 13 | Rewritten | RAG Full | GPT-4o-mini |
| 14 | Rewritten | RAG Full | GPT-4o |
| 15 | Rewritten | RAG Selected | GPT-4o-mini |
| 16 | Rewritten | RAG Selected | GPT-4o |

---

## Implementation Notes

- **Temperature:** 0.2 (low randomness)
- **Top P:** 1.0 (full probability distribution)
- **Language:** Prompts explicitly request German output to match tumor board format in the paper
- **Fragestellung:** German term for "clinical question" - many tumor board reports include explicit questions

---

## Citation

If you use these prompts, please cite:
```bibtex

```
