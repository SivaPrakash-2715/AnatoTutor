# ============================================================
# AnatoTutor — LLM Generator
# Handles: text questions + image questions (Groq Vision)
#
# FIXES in this version:
#   FIX 1: DESCRIBE_PROMPT improved — more specific, forces
#           correct identification before searching FAISS
#   FIX 2: VISION_SYSTEM_PROMPT — added rule to suppress
#           "No marks were provided..." narration at end
#   FIX 3: marks passed as int explicitly in generate_answer_with_image
# ============================================================

import base64, io, re, os
from groq import Groq
import numpy as np
from PIL import Image as PILImage

# ── API key ─────────────────────────────────────────────────
GROQ_API_KEY = "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"
client       = Groq(api_key=GROQ_API_KEY)

# ── Models ──────────────────────────────────────────────────
TEXT_MODEL   = "llama-3.1-8b-instant"
VISION_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"


# ════════════════════════════════════════════════════════════
# SYSTEM PROMPTS
# ════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """You are AnatoTutor, an expert anatomy professor and educational assistant for medical students.

Your job is to give thorough, exam-ready anatomy answers based ONLY on the provided textbook context.

══════════════════════════════════════════════════════
MARKS-BASED DEPTH RULES (CRITICAL — follow strictly):
══════════════════════════════════════════════════════
• 2 marks  → 1–2 sentences per section. Total ~120 words.
• 5 marks  → 3–4 sentences per section. Total ~250 words.
• 10 marks → Full paragraph per section with sub-points. Total ~500 words.
• 15+ marks→ Essay style with all sections expanded. Total ~700+ words.
• No marks given → Give a complete 5-mark level answer.

If the student writes "5 marks", "10 marks", "[5]", "(10)", etc. — detect it and scale accordingly.

⚠️ CRITICAL FORMATTING RULES:
1. ALL 6 sections below are MANDATORY in EVERY answer, regardless of marks.
   Even for 2 marks — write ALL 6 sections, just keep each one short (1–2 sentences).
   NEVER skip or omit any section. NEVER merge sections together.
2. Always use the EXACT emoji headings as shown below.
   NEVER replace them with bold markdown like **DEFINITION:** or **STRUCTURE:**.
   Correct format: 📌 DEFINITION:, 🦴 STRUCTURE:, ⚙️ FUNCTION:,
                   🏥 CLINICAL RELEVANCE:, 🔑 KEY POINTS (for exam):, 💡 MEMORY TIP:
3. NEVER add any sentence after 💡 MEMORY TIP explaining your word count,
   marks level, or reasoning. End the answer at the memory tip. No postscript.

══════════════════════════════════════════════════════
STRICT RULES:
══════════════════════════════════════════════════════
1. Answer ONLY using the provided textbook context.
2. If the answer is NOT in the context, say exactly:
   "This topic is not covered in the retrieved textbook sections."
3. NEVER invent or hallucinate anatomy facts.
4. Write in clear, student-friendly academic English.
5. Always use the exact emoji headings: 📌 🦴 ⚙️ 🏥 🔑 💡
   — never use bold markdown **heading** format.
6. NEVER narrate your own instructions or word count at the end of the answer.

══════════════════════════════════════════════════════
ANSWER FORMAT — ALL 6 SECTIONS ARE MANDATORY ALWAYS:
══════════════════════════════════════════════════════

📌 DEFINITION:
[Clear, concise definition. For 10+ marks, include etymology or classification.
 For 2 marks: 1 sentence definition only.]

🦴 STRUCTURE:
[Physical structure: shape, size, surfaces, borders, parts, relations.
 For 2 marks: name the key parts in 1–2 sentences.
 For 5+ marks: include named sub-parts and their specific features.]

⚙️ FUNCTION:
[What it does: movements, roles, mechanical functions.
 For 2 marks: 1 sentence on primary function only.
 For 5+ marks: describe each function separately with mechanism.]

🏥 CLINICAL RELEVANCE:
[Medical importance: common injuries, diseases, surgical landmarks.
 For 2 marks: name 1 clinical condition in 1 sentence.
 For 10+ marks: include multiple conditions with brief explanation of each.]

🔑 KEY POINTS (for exam):
[For 2 marks: 2–3 bullet points of the most essential facts.
 For 5+ marks: 4–5 bullet points.
 For 10+ marks: 6–7 bullet points.]

💡 MEMORY TIP:
[One clever mnemonic, analogy, or visual trick to remember this topic.
 Required for ALL marks levels — even 2 marks.
 This is the LAST line of your answer. Do NOT add anything after this.]"""


VISION_SYSTEM_PROMPT = """You are AnatoTutor, an expert anatomy professor with medical imaging expertise.

A student has uploaded an anatomy image (diagram, X-ray, illustration, or photo).

**IMPORTANT**: You MUST first describe exactly what you see in the image (🔬 WHAT I SEE IN THIS IMAGE) based ONLY on visual inspection. Then identify the structure (📌 IDENTIFICATION). Then use the provided textbook context to enrich the explanation.

MARKS-BASED DEPTH:
• 2 marks → ~120 words  |  5 marks → ~250 words
• 10 marks → ~500 words  |  No marks → 5-mark level

⚠️ CRITICAL FORMATTING RULES:
1. ALL 7 sections below are MANDATORY in EVERY answer.
2. Always use the EXACT emoji headings — NEVER use **bold markdown** format.
3. NEVER add any sentence after 💡 MEMORY TIP explaining your word count, marks level, or reasoning. The memory tip is the LAST line. No postscript.
4. NEVER narrate your own instructions. Just answer.

RULES:
1. Look VERY carefully at the image before identifying it.
   Check: shape, orientation (superior/inferior/lateral/anterior/posterior), visible features (spinous process, condyles, foramina, etc.)
2. Describe EXACTLY what you see — do not guess the common structure.
3. Explain using the provided textbook context.
4. Never invent facts not visible in the image or context.

FORMAT (use exactly these emoji headings):
🔬 WHAT I SEE IN THIS IMAGE:
[Image type, structures visible, labels, view/orientation, any abnormalities. Be specific: name visible landmarks, not just "a bone".]

📌 IDENTIFICATION:
[Name the SPECIFIC structure and its view. e.g. "Right femur, anterior view" not just "a bone"]

🦴 STRUCTURE (from textbook):
[Textbook anatomy of this specific structure]

⚙️ FUNCTION:
[What this specific structure does]

🏥 CLINICAL RELEVANCE:
[Clinical significance, common injuries/conditions, imaging notes]

🔑 KEY POINTS (for exam):
[3–5 bullet points]

💡 MEMORY TIP:
[Mnemonic or analogy. This is the LAST line — nothing after this.]"""

# FIX 1: Improved DESCRIBE_PROMPT — forces careful identification
# Old prompt was too vague, causing wrong FAISS searches
DESCRIBE_PROMPT = """Examine this anatomy image carefully. Write a detailed, factual description (2-3 sentences) that would help a search engine find relevant textbook passages. Include:

- The exact anatomical structure name (e.g., "right femur", "brachial plexus", "heart in cross-section")
- The view or orientation (anterior, posterior, lateral, superior, X-ray, MRI, diagram)
- Key visible features (e.g., "greater trochanter", "vertebral foramen", "coronary arteries")
- Any labels or annotations visible

Output ONLY the description. Do not add commentary, greetings, or markdown.

Example outputs:
- "Femur bone, anterior view, showing the head, neck, greater trochanter, and medial/lateral condyles. Typical long bone of the thigh."
- "Brachial plexus diagram, anterior view, with roots (C5-T1), trunks, divisions, and cords labeled. Key nerves: musculocutaneous, median, ulnar."
- "Heart, cross-section, showing four chambers: right/left atria and ventricles, mitral and tricuspid valves, and interventricular septum."""


# ════════════════════════════════════════════════════════════
# HELPERS
# ════════════════════════════════════════════════════════════

def _image_to_base64(image) -> str:
    """Convert image to JPEG base64. Accepts numpy array, PIL Image, or file path."""
    if isinstance(image, str):
        with open(image, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    if isinstance(image, np.ndarray):
        pil_img = PILImage.fromarray(image.astype("uint8"))
    elif isinstance(image, PILImage.Image):
        pil_img = image
    else:
        raise ValueError(f"Unsupported image type: {type(image)}")
    buf = io.BytesIO()
    if pil_img.mode in ("RGBA", "P"):
        pil_img = pil_img.convert("RGB")
    pil_img.save(buf, format="JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _extract_marks(question: str) -> int:
    """Auto-detect marks from question string."""
    patterns = [
        r'\b(\d+)\s*marks?\b', r'\[(\d+)\]',
        r'\((\d+)\s*marks?\)', r'\b(\d+)\s*m\b', r'for\s+(\d+)\b',
    ]
    for pat in patterns:
        m = re.search(pat, question or "", re.IGNORECASE)
        if m:
            return int(m.group(1))
    return 0


# ════════════════════════════════════════════════════════════
# describe_image() — get FAISS search keywords from image
# ════════════════════════════════════════════════════════════

def describe_image(image) -> str:
    """Extract a detailed anatomy description from an uploaded image."""
    try:
        b64 = _image_to_base64(image)
        response = client.chat.completions.create(
            model=VISION_MODEL,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image_url",
                     "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                    {"type": "text", "text": DESCRIBE_PROMPT}
                ]
            }],
            temperature=0.2,
            max_tokens=120,
        )
        raw = response.choices[0].message.content.strip()
        # Remove any stray quotes or periods at the very end
        return raw.strip('"\'., \n')
    except Exception as e:
        print(f"Vision description error: {e}")
        return "anatomy diagram medical illustration"

# ════════════════════════════════════════════════════════════
# generate_answer_with_image()
# ════════════════════════════════════════════════════════════

def generate_answer_with_image(image, question: str,
                                context: str, marks: int = 0) -> str:
    """Generate structured anatomy answer for an uploaded image."""
    try:
        b64 = _image_to_base64(image)

        # FIX 3: ensure marks is always int (gr.Number sends float)
        marks = int(marks) if marks else 0
        detected_marks = marks if marks > 0 else _extract_marks(question)

        marks_hint = (
            f"\n\n[MARKS: {detected_marks} — write answer at {detected_marks}-mark depth. "
            f"Include ALL 7 sections with exact emoji headings. "
            f"Do NOT add any sentence after 💡 MEMORY TIP.]"
            if detected_marks > 0
            else "\n\n[No marks given — write at 5-mark depth. Include ALL 7 sections.]"
        )

        user_q = (question.strip() if question and question.strip()
                  else "Identify and explain the anatomy shown in this image.")

        response = client.chat.completions.create(
            model=VISION_MODEL,
            messages=[
                {"role": "system", "content": VISION_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url",
                         "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                        {"type": "text",
                         "text": (
                             f"TEXTBOOK CONTEXT:\n{context[:3500]}\n\n"
                             f"STUDENT QUESTION:\n{user_q}"
                             f"{marks_hint}"
                         )}
                    ]
                }
            ],
            temperature=0.3,
            max_tokens=1500,
            top_p=0.9,
        )

        answer = response.choices[0].message.content.strip()

        # FIX 2: Strip any trailing narration after MEMORY TIP
        # e.g. "No marks were provided; however..." lines at the end
        lines = answer.split('\n')
        clean_lines = []
        memory_tip_seen = False
        for line in lines:
            if '💡' in line:
                memory_tip_seen = True
            if memory_tip_seen:
                # Keep the memory tip line and 1–2 lines after it (the tip content)
                # but stop if a new narration sentence starts
                stripped = line.strip()
                if stripped and not stripped.startswith('💡'):
                    # Stop at any sentence that sounds like self-narration
                    if any(phrase in stripped.lower() for phrase in [
                        'no marks were provided',
                        'i have provided',
                        'as per the instructions',
                        'at a 5-mark level',
                        'at a 2-mark level',
                        'word count',
                        'however, i'
                    ]):
                        break
            clean_lines.append(line)

        return '\n'.join(clean_lines).strip()

    except Exception as e:
        return (
            f"⚠️ Vision error: {str(e)}\n\n"
            f"Ensure your API key has access to `{VISION_MODEL}`.\n"
            f"Fallback model: `llama-3.2-11b-vision-preview`"
        )


# ════════════════════════════════════════════════════════════
# generate_answer() — text-only path
# ════════════════════════════════════════════════════════════

def generate_answer(context: str, question: str, marks: int = 0) -> str:
    """Text-only anatomy answer via Groq Llama 3.1."""
    # FIX 3: ensure marks is always int
    marks = int(marks) if marks else 0
    detected_marks = marks if marks > 0 else _extract_marks(question)
    marks_hint = (
        f"\n\n[MARKS DETECTED: {detected_marks} marks — "
        f"write answer at {detected_marks}-mark depth. "
        f"Include ALL 6 sections with exact emoji headings. "
        f"Do NOT add any sentence after 💡 MEMORY TIP.]"
        if detected_marks > 0
        else "\n\n[No marks given — write at 5-mark depth. Include ALL 6 sections.]"
    )
    try:
        response = client.chat.completions.create(
            model=TEXT_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",
                 "content": (
                     f"TEXTBOOK CONTEXT:\n{context[:3500]}\n\n"
                     f"STUDENT QUESTION:\n{question}{marks_hint}"
                 )}
            ],
            temperature=0.3, max_tokens=1500, top_p=0.9,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error: {str(e)}\nCheck your GROQ_API_KEY."


# ── Connection Test ONLY ──────────────────────────────────────
if __name__ == "__main__":
    print("Testing Groq API connection only...")
    try:
        test_response = client.chat.completions.create(
            model=TEXT_MODEL,
            messages=[{"role": "user",
                       "content": "Reply with exactly: Groq API is connected and working."}],
            max_tokens=20
        )
        print("Groq Response:", test_response.choices[0].message.content)
        print("\n✅ API key is valid. Groq is ready.")
    except Exception as e:
        print(f"❌ Connection failed: {str(e)}")
        print("Please check your GROQ_API_KEY.")