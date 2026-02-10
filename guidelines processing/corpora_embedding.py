# processing/embedding.py
import json
import os
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings, Document

# -------------------------------
# 1. Load embedding model
# -------------------------------
embed_model = Settings.embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-m3",
    model_kwargs={"revision": "5617a9f"}  # pin to exact commit
)
print("Embedding model loaded!")

# -------------------------------
# 2. Load corpora JSON
# -------------------------------
def load_corpora(file_path: str):
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

# -------------------------------
# 3. Embed corpora text
# -------------------------------
def embed_corpora(corpora):
    for chunk in corpora:
        text = chunk["text"]
        embedding = embed_model.get_text_embedding(text)
        chunk["embedding"] = embedding
    return corpora

# -------------------------------
# 4. Save back
# -------------------------------
def save_corpora(corpora, output_path):
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(corpora, f, ensure_ascii=False, indent=2)
    print(f"Saved embeddings to {output_path}")

# -------------------------------
# 5. Main
# -------------------------------
if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))

    INPUT_FILE = os.path.abspath(
        os.path.join(current_dir, '..', 'data', 'dummy_corpora', 'dummy_guidelines.json')
    )

    OUTPUT_FILE = os.path.abspath(
        os.path.join(current_dir, '..', 'data', 'dummy_corpora', 'dummy_guidelines_with_embeddings.json')
    )

    corpora = load_corpora(INPUT_FILE)
    corpora_with_embeddings = embed_corpora(corpora)
    save_corpora(corpora_with_embeddings, OUTPUT_FILE)
