# Pipelines Overview

This folder contains all the Python pipelines and accessory scripts for processing patient cases, integrating guideline corpora, and running single-request prompt or retrieval-augmented generation (RAG). 
- **Remember to insert your API_KEY_PROJECT and your Assistant ID** in `prompts/chatgpt.py`:

```python
# Insert your OpenAI project API key here or load it from environment variables
API_KEY_PROJECT = "YOUR PROJECT API KEY"

[...]

# Insert your Assistant ID here
ASSISTANT_ID = "YOUR ASSISTANT ID"
```

The folder contains **five main files**:

---

## 1. `embeddings.py`

This script contains functions to embed text using the chosen embedding model.  
It is used by the RAG pipeline (`framework_3_RAG.py`) to generate vector representations of patient cases and guideline chunks for local similarity search.  

---

## 2. `rewrite.py`

This script provides a pipeline to rewrite a patient case into a standardized, guideline-style format.  
It uses a prompt template and calls the ChatGPT API to produce a structured, clinical-case presentation.  
Other pipelines (`framework_1_simple_request.py`, `framework_2_chatgpt_assistant.py` and `framework_3_RAG.py`) optionally use this script to operate on rewritten cases.

---

## 3. `framework_1_simple_request.py`

This framework allows the user to run a **single-request prompt** on a patient case.  
- User selects the model at runtime.  
- Case can be original, rewritten or translated only.  
- Output is a single response from ChatGPT using the provided prompt configuration.  

---

## 4. `framework_2_chatgpt_assistant.py`

Runs a patient case through a **pre-configured ChatGPT Assistant**.  
- Uses the assistant’s internal model selection.  
- Optionally works on rewritten and translated only cases.  
- Guidelines are retrieved by the assistant; PDFs must be available in its environment.
- **Remember to insert your Assistant ID** in `prompts/chatgpt.py`:

```python
# Insert your Assistant ID here
ASSISTANT_ID = "YOUR ASSISTANT ID"
```
---

## 5. `framework_3_RAG.py`

Runs a **custom RAG pipeline** on a patient case:  
- Embeds the patient case using `embeddings.py`.  
- Loads precomputed guideline embeddings (dummy or real).  
- Allows optional filtering of chunks marked as selected.  
- Performs local similarity search to retrieve the top-k most relevant guideline chunks.  
- Constructs a prompt combining the case and retrieved chunks.  
- Calls ChatGPT for RAG inference.

Supports optional use of **rewritten** or **translated only cases** and **user-selected guideline corpora** (dummy S3 or real digested guidelines by organ/system).


---

> **Summary:**  
- **Accessory scripts:** `embeddings.py`, `rewrite.py`  
- **Frameworks:** `framework_1_simple_request.py`, `framework_2_chatgpt_assistant.py`, `framework_3_RAG.py`  
- Pipelines are designed to be modular, allowing you to run single prompts, assistant prompts, or a full RAG workflow depending on your use case.

