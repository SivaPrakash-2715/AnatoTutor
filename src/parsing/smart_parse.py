import fitz 
import os
import re

def is_good_page(text: str) -> bool:
    """Returns True if this page has useful anatomy content."""
    text = text.strip()
    
    # Skip nearly empty pages (figures, images only)
    if len(text) < 150:
        return False
    
    # Skip pages that are mostly numbers (index pages)
    words = text.split()
    digit_ratio = sum(1 for w in words if w.replace('.', '').isdigit()) / max(len(words), 1)
    if digit_ratio > 0.35:
        return False
    
    # Skip table of contents (lots of dots "......")
    if text.count('...') > 10:
        return False
    
    # Skip bibliography / references sections
    if re.search(r'^(references|bibliography|index)\b', text[:100].lower()):
        return False
    
    return True


def clean_text(text: str) -> str:
    """Remove common textbook noise from extracted text."""
    # Remove isolated page numbers like "206" on their own line
    text = re.sub(r'\n\s*\d{1,4}\s*\n', '\n', text)
    
    # Remove running headers/footers (short lines that repeat)
    lines = text.split('\n')
    cleaned = [line for line in lines if len(line.strip()) > 3 or line.strip() == '']
    text = '\n'.join(cleaned)
    
    # Normalize whitespace
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r' {2,}', ' ', text)
    
    return text.strip()


def parse_pdf(pdf_path: str, output_path: str, 
              skip_first_pages: int = 15, 
              skip_last_pages: int = 30):
    """
    Parse a PDF textbook and save clean text output.
    
    skip_first_pages: Skip title/copyright/TOC at start
    skip_last_pages:  Skip index/bibliography at end
    """
    doc = fitz.open(pdf_path)
    total_pages = len(doc)
    
    print(f"Total pages: {total_pages}")
    print(f"Processing pages {skip_first_pages + 1} to {total_pages - skip_last_pages}")
    
    parsed_text = ""
    skipped = 0
    kept = 0
    
    for page_num in range(skip_first_pages, total_pages - skip_last_pages):
        page = doc[page_num]
        text = page.get_text("text")
        
        if is_good_page(text):
            cleaned = clean_text(text)
            parsed_text += f"\n--- Page {page_num + 1} ---\n"
            parsed_text += cleaned
            kept += 1
        else:
            skipped += 1
    
    doc.close()
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(parsed_text)
    
    print(f"Done! Kept: {kept} pages | Skipped (noise): {skipped} pages")
    print(f"Output saved: {output_path}")


# ── Run for all books ──────────────────────────────────────────────────
if __name__ == "__main__":
    books = [
        {
            "input": "E:\\Anato-tutor\\data\\textbooks\\Gray's anatomy for students ( PDFDrive ).pdf",
            "output": "E:\\Anato-tutor\\data\\parsed\\grays_anatomy.txt",
            "skip_first": 30,   # Gray's has a long preface
            "skip_last": 50,    # Has a big index
        },
        {
            "input": "E:\\Anato-tutor\\data\\textbooks\\BD_Chaurasias_Human_Anatomy_Volume-1_8th_ED.pdf",
            "output": "E:\\Anato-tutor\\data\\parsed\\bd_chaurasia.txt",
            "skip_first": 20,
            "skip_last": 30,
        },
        {
            "input": "E:\\Anato-tutor\\data\\textbooks\\Vishram-Singh-Textbook-of-Anatomy-Upper-Limb-and-Thorax..pdf",
            "output": "E:\\Anato-tutor\\data\\parsed\\vishram_upper.txt",
            "skip_first": 15,
            "skip_last": 25,
        },
        {
            "input": "E:\\Anato-tutor\\data\\textbooks\\Vishram-Singh-Textbook-of-Anatomy-Head-Neck-and-Brain..pdf",
            "output": "E:\\Anato-tutor\\data\\parsed\\vishram_head.txt",
            "skip_first": 15,
            "skip_last": 25,
        },
        {
            "input": "E:\\Anato-tutor\\data\\textbooks\\AnatomyAndPhysiology-LR.pdf",
            "output": "E:\\Anato-tutor\\data\\parsed\\openstax.txt",
            "skip_first": 10,
            "skip_last": 20,
        },
    ]
    
    import os
    os.makedirs("E:\\Anato-tutor\\data\\parsed", exist_ok=True)
    
    for book in books:
        print(f"\nParsing: {book['input']}")
        parse_pdf(
            book["input"], 
            book["output"],
            book["skip_first"],
            book["skip_last"]
        )