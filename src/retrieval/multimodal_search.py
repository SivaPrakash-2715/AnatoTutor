# ============================================================
# AnatoTutor — Multimodal Retrieval
#
# NEW vs PREVIOUS:
#   + retrieve_from_image() — takes uploaded image, calls
#     describe_image() to get keywords, then searches FAISS
#     the same way as retrieve() does for text queries
#   - retrieve() unchanged — text query path
# ============================================================

import faiss, numpy as np, pickle, os, re
from sentence_transformers import SentenceTransformer
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

BASE       = "E:\\Anato-tutor"
INDEX_DIR  = f"{BASE}\\data\\indices"
IMAGES_DIR = f"{BASE}\\data\\images"

print("Loading retrieval system...")
model = SentenceTransformer("E:\\Anato-tutor\\models\\all-MiniLM-L6-v2")
text_index = faiss.read_index(f"{INDEX_DIR}\\text_index.faiss")
with open(f"{INDEX_DIR}\\text_chunks.pkl", "rb") as f:
    text_data = pickle.load(f)

image_index = None
image_data  = None
if os.path.exists(f"{INDEX_DIR}\\image_index.faiss"):
    image_index = faiss.read_index(f"{INDEX_DIR}\\image_index.faiss")
    with open(f"{INDEX_DIR}\\image_metadata.pkl", "rb") as f:
        image_data = pickle.load(f)
    print("  ✅ Text + Image indices loaded")
else:
    print("  ✅ Text index loaded (no image index yet)")

print("Retrieval system ready.\n")


# ════════════════════════════════════════════════════════════
# SHARED HELPERS
# ════════════════════════════════════════════════════════════

def _expand_image_query(query: str) -> list:
    """Generate 3 anatomy-tuned search variants from a query."""
    clean = re.sub(
        r'\b(what|where|how|why|explain|describe|define|write|note on|give|'
        r'marks?|for|about|the|an|a|is|are|of|in|on|with)\b',
        '', query, flags=re.IGNORECASE
    )
    clean = re.sub(r'\[\d+\]|\(\d+\s*marks?\)', '', clean).strip().strip('?.,')
    return [
        f"{clean} anatomy diagram",
        f"{clean} labeled illustration",
        f"{clean} medical figure",
    ]


def _search_images(query_vector, image_top_k=4, score_threshold=0.20):
    """
    Search image FAISS index with a pre-computed query vector.
    Returns list of image result dicts, sorted by score.
    """
    if image_index is None or image_data is None:
        return []

    seen_ids  = set()
    candidates = []

    img_dist, img_idx = image_index.search(query_vector, image_top_k * 3)
    for idx, dist in zip(img_idx[0], img_dist[0]):
        if idx < 0 or idx in seen_ids or float(dist) < score_threshold:
            continue
        seen_ids.add(idx)
        filepath = os.path.join(IMAGES_DIR, image_data["filenames"][idx])
        candidates.append({
            "filename": image_data["filenames"][idx],
            "caption" : image_data["captions"][idx],
            "filepath": filepath,
            "score"   : float(dist)
        })

    candidates.sort(key=lambda x: x["score"], reverse=True)
    return candidates[:image_top_k]


# ════════════════════════════════════════════════════════════
# EXISTING — retrieve()   (text query, unchanged logic)
# ════════════════════════════════════════════════════════════

def retrieve(query: str, text_top_k: int = 6,
             image_top_k: int = 4,
             image_score_threshold: float = 0.20) -> tuple:
    """Retrieve textbook chunks + anatomy images for a text query."""

    query_vector = model.encode([query], normalize_embeddings=True).astype("float32")

    # ── Text chunks ───────────────────────────────────────
    distances, indices = text_index.search(query_vector, text_top_k * 3)
    text_results  = []
    seen_prefixes = set()
    source_counts = {}

    for idx, dist in zip(indices[0], distances[0]):
        if idx < 0: continue
        chunk  = text_data["chunks"][idx]
        source = text_data["sources"][idx]
        prefix = chunk[:80].strip()
        if prefix in seen_prefixes: continue
        seen_prefixes.add(prefix)
        if len(chunk.strip()) < 100: continue
        if source_counts.get(source, 0) >= 2: continue
        source_counts[source] = source_counts.get(source, 0) + 1
        text_results.append({"text": chunk, "source": source, "score": float(dist)})
        if len(text_results) >= text_top_k: break

    # ── Images: multi-query expansion ─────────────────────
    expanded = _expand_image_query(query)
    all_queries = [query] + expanded
    seen_img_ids = set()
    all_candidates = []

    for q in all_queries:
        qv = model.encode([q], normalize_embeddings=True).astype("float32")
        results = _search_images(qv, image_top_k, image_score_threshold)
        for r in results:
            if r["filename"] not in seen_img_ids:
                seen_img_ids.add(r["filename"])
                all_candidates.append(r)

    all_candidates.sort(key=lambda x: x["score"], reverse=True)
    image_results = all_candidates[:image_top_k]

    return text_results, image_results


# ════════════════════════════════════════════════════════════
# NEW — retrieve_from_image()
#
# What it does:
#   1. Calls describe_image() → gets short anatomy keyword string
#      e.g. "femur bone anterior view"
#   2. Uses that keyword string EXACTLY like retrieve() does for
#      text — encodes it, searches FAISS for related textbook
#      chunks and related anatomy images from our library
#   3. Returns (description, text_results, image_results)
#      where description is shown to user so they can see what
#      the system understood from their image
#
# Called by: app.py → ask_anatututor_image()
# ════════════════════════════════════════════════════════════

def retrieve_from_image(image,
                        text_top_k: int  = 6,
                        image_top_k: int = 4) -> tuple:
    """
    Retrieve textbook content relevant to an uploaded anatomy image.

    Pipeline:
      uploaded image → Groq vision → description string
                     → MiniLM embed → FAISS search
                     → text chunks + related images

    Args:
        image      : numpy array / PIL Image from Gradio
        text_top_k : number of textbook chunks to retrieve
        image_top_k: number of related library images to retrieve

    Returns:
        (description, text_results, image_results)
        description  — what Groq understood from the image
        text_results — list of textbook chunk dicts
        image_results— list of library image dicts
    """
    # Step 1: Get description from Groq vision
    from rag.llm_generator import describe_image
    description = describe_image(image)

    # Step 2: Use description to search FAISS (same as text retrieve)
    text_results, image_results = retrieve(
        query=description,
        text_top_k=text_top_k,
        image_top_k=image_top_k,
    )

    return description, text_results, image_results