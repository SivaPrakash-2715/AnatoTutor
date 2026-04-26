import faiss
import numpy as np
import json
import pickle
import os
from sentence_transformers import SentenceTransformer


# ── Configuration ─────────────────────────────────────────────
BASE          = "E:\\Anato-tutor"
CAPTIONS_JSON = f"{BASE}\\data\\image_captions.json"
OUTPUT_DIR    = f"{BASE}\\data\\indices"
MODEL_NAME    = "all-MiniLM-L6-v2"

os.makedirs(OUTPUT_DIR, exist_ok=True)


def build_image_index():
    """
    Build a FAISS vector index from image captions.

    Full pipeline:
      image_captions.json
            ↓
      Load captions (text strings)
            ↓
      Generate embeddings (MiniLM model)
            ↓
      Normalize embeddings (enable cosine similarity)
            ↓
      Build FAISS IndexFlatIP (Inner Product = cosine)
            ↓
      Save: image_index.faiss + image_metadata.pkl
    """

    # ── STEP 1: Load image captions ───────────────────────────
    print("=" * 55)
    print("  STEP 1: Loading Image Captions")
    print("=" * 55)

    # Validate file exists
    if not os.path.exists(CAPTIONS_JSON):
        print(f"\n❌ ERROR: image_captions.json not found!")
        print(f"   Expected at: {CAPTIONS_JSON}")
        print(f"   Please run caption_images.py first.")
        return

    with open(CAPTIONS_JSON, "r", encoding="utf-8") as f:
        captions_dict = json.load(f)

    # Filter out any empty/blank captions
    filenames = [k for k, v in captions_dict.items() if v.strip()]
    captions  = [captions_dict[k] for k in filenames]
    total     = len(captions)

    if total == 0:
        print("❌ ERROR: No valid captions found in image_captions.json")
        print("   Please re-run caption_images.py")
        return

    print(f"\n   ✅ Loaded {total} image captions")
    print(f"\n   Sample captions:")
    for i in range(min(3, total)):
        print(f"   [{i+1}] File   : {filenames[i]}")
        print(f"        Caption: {captions[i]}")
        print()


    # ── STEP 2: Load Embedding Model ──────────────────────────
    print("=" * 55)
    print("  STEP 2: Loading Sentence Embedding Model")
    print("=" * 55)
    print(f"\n   Model : {MODEL_NAME}")
    print(f"   This model converts text → numeric vectors (embeddings)")
    print(f"   Output: 384-dimensional vector per caption\n")

    model = SentenceTransformer(MODEL_NAME, local_files_only=True)
    print("   ✅ Model loaded\n")


    # ── STEP 3: Generate Embeddings ───────────────────────────
    print("=" * 55)
    print("  STEP 3: Generating Caption Embeddings")
    print("=" * 55)
    print(f"\n   Converting {total} captions into vectors...")
    print(f"   normalize_embeddings=True → enables cosine similarity\n")

    embeddings = model.encode(
        captions,
        batch_size           = 32,
        show_progress_bar    = True,
        convert_to_numpy     = True,
        normalize_embeddings = True    # Cosine similarity: measures meaning angle
    ).astype("float32")

    dimension = embeddings.shape[1]
    print(f"\n   ✅ Embeddings generated")
    print(f"   Shape: {embeddings.shape}  ({total} images × {dimension} dimensions)")


    # ── STEP 4: Build FAISS Index ─────────────────────────────
    print("\n" + "=" * 55)
    print("  STEP 4: Building FAISS Vector Index")
    print("=" * 55)
    print(f"\n   Index type : IndexFlatIP (Inner Product)")
    print(f"   Why IP?    : With normalized vectors, Inner Product")
    print(f"                = Cosine Similarity (standard for semantic search)")
    print(f"   Dimension  : {dimension}\n")

    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    print(f"   ✅ FAISS index built")
    print(f"   Total vectors in index: {index.ntotal}")


    # ── STEP 5: Save to Disk ──────────────────────────────────
    print("\n" + "=" * 55)
    print("  STEP 5: Saving Index to Disk")
    print("=" * 55)

    index_path    = f"{OUTPUT_DIR}\\image_index.faiss"
    metadata_path = f"{OUTPUT_DIR}\\image_metadata.pkl"

    # Save FAISS index
    faiss.write_index(index, index_path)
    print(f"\n   ✅ Saved: image_index.faiss")
    print(f"      Path : {index_path}")

    # Save metadata (filenames + captions) for lookup during retrieval
    with open(metadata_path, "wb") as f:
        pickle.dump({
            "filenames": filenames,   # to find the actual image file
            "captions" : captions     # to display caption in the UI
        }, f)

    print(f"\n   ✅ Saved: image_metadata.pkl")
    print(f"      Path : {metadata_path}")
    print(f"      Contains: filenames list + captions list")


    # ── Summary ───────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("  ✅ IMAGE INDEX BUILT SUCCESSFULLY")
    print("=" * 55)
    print(f"\n   Total images indexed : {total}")
    print(f"   Embedding dimensions : {dimension}")
    print(f"   Similarity method    : Cosine Similarity (IndexFlatIP)")
    print(f"   Index saved to       : {index_path}")
    print(f"   Metadata saved to    : {metadata_path}")
    print(f"\n   Next step: Run rag_pipeline.py to test image retrieval")
    print("=" * 55)


# ── Entry Point ───────────────────────────────────────────────
if __name__ == "__main__":
    build_image_index()