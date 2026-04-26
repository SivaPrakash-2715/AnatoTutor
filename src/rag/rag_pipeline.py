# ============================================================
# AnatoTutor — Final RAG Pipeline
# Combines: Retrieval → Context Building → Groq LLM Generation
# Run this to test the full system end-to-end.
#
# CHANGES FROM PREVIOUS VERSION:
#   1. context cap raised 2000 → 3500 (matches llm_generator.py)
#   2. image_top_k raised 2 → 4 (matches multimodal_search.py)
#   3. marks parameter added to run_anatomotutor() and forwarded to generate_answer()
# ============================================================

import sys
import os
import time

# Add src to path so imports work
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from retrieval.multimodal_search import retrieve
from rag.llm_generator import generate_answer


def run_anatomotutor(query: str, marks: int = 0) -> dict:
    """
    Full AnatoTutor pipeline for a single query.

    Steps:
    1. Retrieve relevant text chunks from FAISS text index
    2. Retrieve related anatomy images from FAISS image index
    3. Build a clean context string from retrieved chunks
    4. Send context + question to Groq LLM (Llama 3)
    5. Return structured result

    Args:
        query : The student's anatomy question
        marks : Optional marks value (0 = auto-detect from question text)

    Returns:
        dict with keys: query, answer, sources, related_images, image_captions, latency
    """
    start_time = time.time()

    # Step 1 & 2 — Retrieve
    print("  🔍 Retrieving relevant knowledge...")

    # CHANGE 2: image_top_k raised 2 → 4
    text_results, image_results = retrieve(query, text_top_k=6, image_top_k=4)

    if not text_results:
        return {
            "query"         : query,
            "answer"        : "No relevant content found in the textbooks for this query. Please try rephrasing.",
            "sources"       : [],
            "related_images": [],
            "image_captions": [],
            "latency"       : 0
        }

    # Step 3 — Build context string
    context_parts = []
    for result in text_results:
        source = result["source"].replace(".txt", "").replace("_", " ").title()
        context_parts.append(f"[From: {source}]\n{result['text']}")

    context = "\n\n---\n\n".join(context_parts)

    # CHANGE 1: context cap raised 2000 → 3500 to match llm_generator.py
    context = context[:3500]

    # Step 4 — Generate with Groq (Llama 3)
    print("  🤖 Generating explanation with Llama 3 via Groq...")

    # CHANGE 3: pass marks so marks-aware depth scaling works
    answer = generate_answer(context, query, marks=marks)

    latency = round(time.time() - start_time, 2)

    # Step 5 — Return result
    return {
        "query"         : query,
        "answer"        : answer,
        "sources"       : list(set(r["source"] for r in text_results)),
        "related_images": [r["filepath"] for r in image_results],
        "image_captions": [r["caption"]  for r in image_results],
        "latency"       : latency
    }


def print_result(result: dict):
    """Pretty print the AnatoTutor result to the terminal."""
    print("\n" + "=" * 60)
    print("  ANATOMOTUTOR — ANATOMY EXPLANATION")
    print("=" * 60)

    print(f"\n❓ Question: {result['query']}\n")
    print(result["answer"])

    print("\n" + "-" * 60)
    print("📚 Textbook Sources Used:")
    for src in result["sources"]:
        label = src.replace(".txt", "").replace("_", " ").title()
        print(f"   • {label}")

    if result["related_images"]:
        print("\n🖼️  Related Anatomical Images:")
        for img, cap in zip(result["related_images"], result["image_captions"]):
            print(f"   • {os.path.basename(img)}")
            print(f"     Caption: {cap}")

    print(f"\n⏱️  Response time: {result['latency']} seconds")
    print("=" * 60)


# ── Interactive Terminal Mode ─────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  Welcome to AnatoTutor")
    print("  Powered by: FAISS Retrieval + Llama 3 (Groq)")
    print("  Type 'exit' to quit")
    print("=" * 60)

    while True:
        print()
        query = input("Enter your anatomy question: ").strip()

        if not query:
            continue

        if query.lower() in ("exit", "quit", "q"):
            print("Goodbye!")
            break

        result = run_anatomotutor(query)
        print_result(result)