"""
Run a CUSTOM RAG pipeline using:
- Embedding of the patient case
- Precomputed guideline embeddings (dummy or real)
- Local similarity search
- Prompt-based RAG inference via ChatCompletion

IMPORTANT:
- Guideline embeddings are fully synthetic/dummy for demonstration only
- Real guidelines must already have embeddings computed
"""

import os
import sys
import json
import numpy as np
from typing import List, Dict, Optional

# ------------------------------------------------------------------
# Path setup: allow imports from parallel folders
# ------------------------------------------------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))
prompts_dir = os.path.abspath(os.path.join(current_dir, '..', 'prompts'))
processing_dir = os.path.abspath(os.path.join(current_dir, '..', 'processing'))
dummy_corpora_dir = os.path.abspath(os.path.join(current_dir, '..', 'data', 'dummy_corpora'))

sys.path.append(prompts_dir)
sys.path.append(processing_dir)
sys.path.append(dummy_corpora_dir)

# ------------------------------------------------------------------
# Imports
# ------------------------------------------------------------------
from prompt_templates import get_prompt_for_configuration
from chatgpt import chatgpt_chat_completion
from embeddings import embed_text
from guideline_dictionary_dummy import guidelines_s3_dict

# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------
TOP_K = 5

CASE_PATH = os.path.abspath(
    os.path.join(current_dir, '..', 'data', 'dummy_patients', 'example_case_de.txt')
)

# ------------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------------
def load_case_text(path: str) -> str:
    with open(path, 'r', encoding='utf-8') as f:
        return f.read().strip()


def load_guideline_corpora(json_paths: List[str]) -> List[Dict]:
    merged = []
    for path in json_paths:
        with open(path, 'r', encoding='utf-8') as f:
            merged.extend(json.load(f))
    return merged


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return float(np.dot(vec1, vec2) / (norm1 * norm2))


def retrieve_top_k_chunks(
    query_embedding: np.ndarray,
    corpora: Optional[List[Dict]],
    top_k: int = 5
) -> List[Dict]:
    scored_chunks = []
    for chunk in corpora:
        chunk_embedding = np.array(chunk["embedding"], dtype=float)
        score = cosine_similarity(query_embedding, chunk_embedding)
        scored_chunks.append({
            "chunk_id": chunk.get("chunk_id"),
            "source": chunk.get("source"),
            "text": chunk.get("text"),
            "score": score
        })
    scored_chunks.sort(key=lambda x: x["score"], reverse=True)
    return scored_chunks[:top_k]

# ------------------------------------------------------------------
# Main pipeline
# ------------------------------------------------------------------
if __name__ == "__main__":

    print("Running local RAG pipeline\n")

    # -----------------------------
    # Load patient case
    # -----------------------------
    case_text = load_case_text(CASE_PATH)

    # -----------------------------
    # User configuration
    # -----------------------------
    # Select the model
    print("Select model (1 or 2):")
    print("1) gpt-4o-mini")
    print("2) gpt-4o")
    model_choice = input("Enter choice [1 or 2]: ").strip()
    MODEL_NAME = "gpt-4o-mini-2024-07-18" if model_choice == "1" else "gpt-4o-2024-11-20"
    print('\nSelected model: {}\n'.format(MODEL_NAME))


    # Optional rewriting or translating only
    use_rewritten = input("Use rewritten or translated only case? (y/n): ").strip().lower() == "y"

    if use_rewritten:
        use_translated_only = input(
            "Do you want to use the translated (tr) only case or the rewritten (rw) one? (tr/rw): ").strip().lower() == "tr"
    else:
        use_translated_only = False

    if use_translated_only:
        try:
            from rewrite import translate_case_from_txt

            case_text = translate_case_from_txt(CASE_PATH)
            print("\n=== Using translated only case ===\n")
        except ImportError:
            print("WARNING: Translate function not found. Using original case.\n")

    elif use_rewritten:
        try:
            from rewrite import rewrite_case_from_txt
            case_text = rewrite_case_from_txt(CASE_PATH)
            print("\n=== Using rewritten case ===\n")
        except ImportError:
            print("WARNING: Rewrite function not found. Using original case.\n")

    # -----------------------------
    # Select guideline corpora
    # -----------------------------
    print("\nWARNING: You are about to use the dummy guideline corpus for demonstration.")
    use_real = input("Do you want to use real guidelines that have already been digested? (y/n): ").strip().lower() == "y"

    selected_json_files = []
    if use_real:
        print("\nSelect organ/system for corpus:")
        print("Options: esophageal, gastric, hepatic, pancreatic, colorectal, abort")
        organ_choice = input("Enter your choice: ").strip().lower()

        if organ_choice in ["esophageal", "gastric", "hepatic", "pancreatic", "colorectal"]:
            # Search all JSON files in dummy_corpora folder containing the keyword
            selected_json_files = [
                os.path.join(dummy_corpora_dir, f)
                for f in os.listdir(dummy_corpora_dir)
                if f.endswith(".json") and organ_choice in f
            ]
            if not selected_json_files:
                print(f"No JSON files found for {organ_choice}, using dummy corpus instead.\n")
        else:
            use_real = False
            print("Abort or invalid choice, using dummy corpus.\n")


    if not selected_json_files:
        if use_real:
            # Fallback to dummy S3 guidelines
            selected_json_files = [
                os.path.join(dummy_corpora_dir, f"dummy_S3_guidelines_{k.replace('-', '_')}.json")
                for k in guidelines_s3_dict.keys()
            ]
        else:
            selected_json_files = [os.path.join(current_dir, '..', 'data', 'dummy_corpora', 'dummy_guidelines_with_embeddings.json')]

    # Load and merge JSON files
    corpora = load_guideline_corpora(selected_json_files)
    print(f"Loaded {len(corpora)} chunks from {len(selected_json_files)} JSON file(s).\n")

    # Optionally filter by selected_corpora
    use_selected_corpora = input("Use only selected corpora chunks? (y/n): ").strip().lower() == "y"
    if use_selected_corpora:
        corpora = [chunk for chunk in corpora if chunk.get("selected_corpora", 0) == 1]
        print(f"Using only selected corpora chunks: {len(corpora)} available\n")

    # -----------------------------
    # Embed patient case
    # -----------------------------
    query_embedding = np.array(embed_text(case_text), dtype=float)

    # -----------------------------
    # Retrieve top-k chunks
    # -----------------------------
    try:
        retrieved_chunks = retrieve_top_k_chunks(query_embedding, corpora, top_k=TOP_K)
    except ValueError as e:
        if "shapes" in str(e) and "not aligned" in str(e):
            print("ERROR: Embedding dimension mismatch detected.")
            print("RUN the corpora_embeddings.py script in the folder 'guidelines_processing' first!")
            sys.exit(1)
        else:
            raise

    # -----------------------------
    # Build RAG prompt
    # -----------------------------
    prompt = get_prompt_for_configuration(
        case_text=case_text,
        config_type="rag_full",
        retrieved_chunks=retrieved_chunks
    )

    # -----------------------------
    # Run model
    # -----------------------------
    response = chatgpt_chat_completion(prompt, model=MODEL_NAME)

    # -----------------------------
    # Output
    # -----------------------------
    print("=== Patient Case ===")
    print(case_text)

    print("\n=== Retrieved Guideline Chunks ===")
    for chunk in retrieved_chunks:
        print(f"- Chunk {chunk['chunk_id']} | Score: {chunk['score']:.4f}  | Text: {chunk['text']}")

    print("\n=== RAG Output ===")
    print(response)
