# gi-rag-tumor-board

Retrieval-Augmented Generation framework for gastrointestinal oncology tumor board decision support

This repository accompanies the submitted manuscript:
"[Title under submission]".

## Purpose

This repository provides full transparency of the methodological framework, prompting strategies, and retrieval pipelines used in the study.

Note: Due to patient privacy regulations and copyright restrictions, no real patient data or guideline documents are included.


## Repository Structure & Contents

| Folder / File           | Description                                                                                                                                                                                                                                                                                                                                                                                                                                 |
| ----------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `config`                | Contains hyperparameter files (`hyperparameters.yaml`) for experimental setups.                                                                                                                                                                                                                                                                                                                                                             |
| `data`                  | Contains **dummy patient cases** (`dummy_patients/`) and **synthetic guideline corpora** (`dummy_corpora/`), plus the guideline dictionary (`guideline_dictionary_dummy.py`).                                                                                                                                                                                                                                                                                   |
| `experiments`           | Includes configuration matrices and evaluation documentation for reproducing the 16 experimental setups (`configuration_matrix.yaml`, `evaluation.md`).                                                                                                                                                                                                                                                                                                                        |
| `guidelines processing` | Contains scripts for processing guidelines, chunking PDFs, and generating JSON corpora.                                                                                                                                                                                                                                                                                                                                                     |
| `pipelines`             | All pipeline scripts: <br>- `framework_1_simple_request.py` – runs a single request prompt on a patient case <br>- `framework_2_chatgpt_assistant.py` – runs an assistant-style prompt using a pre-configured ChatGPT Assistant <br>- `framework_3_RAG.py` – executes the custom RAG pipeline <br>- `rewrite.py` – rewrites patient cases into structured guideline-style format <br>- `embeddings.py` – embeds patient cases for retrieval |
| `prompts`               | Contains all prompt templates and examples for different experimental configurations (`README.md` inside the folder explains usage).                                                                                                                                                                                                                                                                                                        |
| `getting_started.md`    | Step-by-step instructions to install dependencies and run the pipelines, including how to switch between dummy and real guideline corpora.                                                                                                                                                                                                                                                                                                  |
| `.gitignore`            | Git ignore file.                                                                                                                                                                                                                                                                                                                                                                                                                            |
| `LICENSE`               | Repository license.                                                                                                                                                                                                                                                                                                                                                                                                                         |
| `README.md`             | Main repository overview, scope, and structure description.                                                                                                                                                                                                                                                                                                                                                                                 |
| `requirements.txt`      | All Python dependencies needed to run the pipelines.                                                                                                                                                                                                                                                                                                                                                                                        |


## Reproducibility

The repository enables methodological replication of the study design and prompt engineering strategies. Synthetic data demonstrate end-to-end execution of the pipelines.

- Dummy patient cases: data/dummy_patients/

- Dummy guideline corpora: data/dummy_corpora/

- Experimental configuration: experiments/configuration_matrix.yaml

You can run any of the three pipelines (framework_1, framework_2, framework_3) on the dummy data to reproduce retrieval, embedding, and RAG outputs.


## What Is Included

- Prompt templates and examples for all experimental setups

- Dummy patient cases and synthetic guideline corpora

- Guideline processing logic (PDF chunking → JSON → embeddings)

- Configuration matrix for the 16 experimental setups

- Retrieval and RAG orchestration logic

## What is Not Included

- Real patient-level clinical data

- Guideline texts (German S3, NCCN)

- API keys or credentials


## Disclaimer

This code is for research purposes only and is not intended for clinical use.


---

