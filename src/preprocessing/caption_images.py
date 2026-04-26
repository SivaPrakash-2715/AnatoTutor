import os
import re
import json
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration


# Load BLIP model once
print("Loading BLIP image captioning model...")
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model     = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
print("BLIP model loaded.\n")


def extract_keywords_from_filename(filename: str) -> str:
    """
    Extract source book info from the image filename.
    Examples:
      "Grays_anatomy_page274_img2.jpeg"  → "grays anatomy"
      "BD_Chaurasias_Human_Anatomy_page129_img2.jpeg" → "bd chaurasias human anatomy"
    """
    name = os.path.splitext(filename)[0]
    name = re.sub(r'_page\d+_img\d+', '', name)
    name = re.sub(r'[_\-\(\)]+', ' ', name)
    name = re.sub(r'\s+', ' ', name).strip()
    return name.lower()


def _is_repetitive(caption: str) -> bool:
    """
    FIX 1: Detect looping/repetitive captions like:
      "liver, liver, liver, liver, liver, liver, liver"
      "the bone and the bone and the bone"

    Method: count unique words vs total words.
    If unique words < 40% of total → repetitive → reject.
    """
    words = caption.lower().split()
    if len(words) == 0:
        return True
    unique_ratio = len(set(words)) / len(words)
    return unique_ratio < 0.4


def _score_caption(caption: str) -> int:
    """
    FIX 2: Score caption by number of UNIQUE words.
    This picks the most informative caption, not just the longest.

    Example:
      "liver, liver, liver, liver, liver"  → score 1  (1 unique word)
      "diagram of the femur bone anterior" → score 6  (6 unique words)
    """
    return len(set(caption.lower().split()))


def caption_image(image_path: str) -> str:
    """
    Generate a rich, searchable caption for an anatomical image.

    Strategy:
    1. Run BLIP with 3 different prompts
    2. Filter out repetitive/looping captions (FIX 1)
    3. Pick caption with most unique words (FIX 2)
    4. Append filename keywords for source context
    """
    try:
        image    = Image.open(image_path).convert("RGB")
        filename = os.path.basename(image_path)

        file_keywords = extract_keywords_from_filename(filename)

        prompts = [
            "A medical anatomy diagram showing",
            "This anatomical illustration shows",
            "An anatomy textbook image of",
        ]

        best_caption = ""
        best_score   = -1

        for prompt_text in prompts:
            inputs  = processor(image, text=prompt_text, return_tensors="pt")
            output  = model.generate(**inputs, max_new_tokens=80)
            caption = processor.decode(output[0], skip_special_tokens=True).strip()

            # FIX 1: Skip repetitive captions
            if _is_repetitive(caption):
                continue

            # FIX 2: Pick caption with most unique words
            score = _score_caption(caption)
            if score > best_score:
                best_score   = score
                best_caption = caption

        # If ALL 3 prompts gave repetitive captions, use a fallback
        if not best_caption:
            best_caption = f"An anatomy textbook image"

        final_caption = f"{best_caption}. Source: {file_keywords}."
        return final_caption

    except Exception as e:
        print(f"  Warning: Could not caption {os.path.basename(image_path)}: {e}")
        return ""


def caption_all_images(images_dir: str, output_json: str):
    """
    Caption all anatomy images and save to JSON.

    FIX 3: Resume support — if output_json already exists,
    load existing captions and skip already-processed images.
    A crash at image 800/1543 will resume from 801, not restart from 0.
    """
    image_files = sorted([
        f for f in os.listdir(images_dir)
        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))
    ])

    total = len(image_files)
    print(f"Found {total} images total.\n")

    # FIX 3: Load existing captions if file exists (resume support)
    if os.path.exists(output_json):
        with open(output_json, "r", encoding="utf-8") as f:
            captions = json.load(f)
        print(f"Resuming — {len(captions)} images already captioned, skipping them.\n")
    else:
        captions = {}

    skipped   = 0
    processed = 0
    failed    = 0

    for i, filename in enumerate(image_files):

        # FIX 3: Skip already-captioned images
        if filename in captions:
            skipped += 1
            continue

        path    = os.path.join(images_dir, filename)
        caption = caption_image(path)

        if caption:
            captions[filename] = caption
            processed += 1
        else:
            failed += 1

        # Save every 20 images so a crash doesn't lose all progress
        if processed % 20 == 0 and processed > 0:
            with open(output_json, "w", encoding="utf-8") as f:
                json.dump(captions, f, indent=2)

        # Show progress every 10 images
        if (i + 1) % 10 == 0 or (i + 1) == total:
            print(f"  [{i+1}/{total}] {filename}")
            print(f"  Caption: {caption}\n")

    # Final save
    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(captions, f, indent=2)

    print(f"\n✅ Captions saved: {output_json}")
    print(f"   Total in file : {len(captions)}")
    print(f"   Newly captioned: {processed}")
    print(f"   Skipped (already done): {skipped}")
    print(f"   Failed: {failed}")


# Run
if __name__ == "__main__":
    BASE = "E:\\Anato-tutor"

    caption_all_images(
        images_dir  = f"{BASE}\\data\\images",
        output_json = f"{BASE}\\data\\image_captions.json"
    )