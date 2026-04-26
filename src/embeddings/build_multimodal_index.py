import faiss
import numpy as np
import json
import pickle
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os

MODEL_NAME = 'all-MiniLM-L6-v2'

# FIX 3: local_files_only=True — prevents internet lookup failure
model = SentenceTransformer(MODEL_NAME, local_files_only=True)

PARSED_DIR = "E:\\Anato-tutor\\data\\parsed"
OUTPUT_DIR = "E:\\Anato-tutor\\data\\indices"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ── Build text index ──────────────────────────────────────────
def build_text_index():
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=100
    )

    all_chunks  = []
    all_sources = []

    txt_files = [f for f in os.listdir(PARSED_DIR) if f.endswith(".txt")]
    if not txt_files:
        print(f"\n❌ No .txt files found in {PARSED_DIR}")
        print("   Run smart_parse.py first.")
        return

    for filename in txt_files:
        filepath = os.path.join(PARSED_DIR, filename)
        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read()

        chunks      = splitter.split_text(text)
        good_chunks = [c for c in chunks if len(c.strip()) > 100]
        all_chunks.extend(good_chunks)
        all_sources.extend([filename] * len(good_chunks))

        print(f"  {filename}: {len(good_chunks)} chunks")

    print(f"\nTotal text chunks: {len(all_chunks)}")

    # Generate embeddings
    print("Generating text embeddings...")
    embeddings = model.encode(
        all_chunks,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True   # FIX 2: must match search query normalization
    ).astype("float32")

    # FIX 1: IndexFlatIP (cosine similarity) instead of IndexFlatL2
    # multimodal_search.py uses normalize_embeddings=True on queries,
    # so the index must also use IP to get correct cosine scores.
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    # Save
    faiss.write_index(index, f"{OUTPUT_DIR}\\text_index.faiss")

    with open(f"{OUTPUT_DIR}\\text_chunks.pkl", "wb") as f:
        pickle.dump({"chunks": all_chunks, "sources": all_sources}, f)

    print(f"\n✅ Text index saved.")
    print(f"   Total chunks : {len(all_chunks)}")
    print(f"   Dimensions   : {dimension}")
    print(f"   Index type   : IndexFlatIP (cosine similarity)")


if __name__ == "__main__":
    print("=== Building Text Index ===")
    build_text_index()
    print("\n✅ Text index built successfully!")