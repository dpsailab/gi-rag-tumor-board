from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings
import os

# -------------------------------
# 1. Load embedding model
# -------------------------------
embed_model = Settings.embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-m3",
    model_kwargs={"revision": "5617a9f"}  # pin to exact commit
)
print("Embedding model loaded!")

# -------------------------------
# 2a. Embed from TEXT
# -------------------------------
def embed_text(text: str):
    return embed_model.get_text_embedding(text)

# -------------------------------
# 2b. Embed from FILE
# -------------------------------
def embed_text_from_file(txt_path: str):
    with open(txt_path, "r", encoding="utf-8") as f:
        text = f.read()
    return embed_text(text)

# -------------------------------
# 3. Example usage
# -------------------------------
# if __name__ == "__main__":
#     current_dir = os.path.dirname(os.path.abspath(__file__))
#     case_txt_path = os.path.abspath(
#         os.path.join(current_dir, '..', 'data', 'dummy_patients', 'example_case_de.txt')
#     )
# 
#     embedding = embed_text_from_file(case_txt_path)
#     print(f"Patient embedding (first 10 values): {embedding[:10]}")
