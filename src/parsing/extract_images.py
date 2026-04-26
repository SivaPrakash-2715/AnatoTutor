# src/parsing/extract_images.py
import fitz
import os
from pathlib import Path

def extract_images_from_pdf(pdf_path: str, output_dir: str, 
                             min_width: int = 200, min_height: int = 200):
    """
    Extract all images from a PDF that are large enough to be real diagrams.
    Small images are likely logos, icons, or decorations.
    """
    doc = fitz.open(pdf_path)
    book_name = Path(pdf_path).stem
    os.makedirs(output_dir, exist_ok=True)
    
    count = 0
    for page_num, page in enumerate(doc):
        images = page.get_images(full=True)
        
        for img_idx, img in enumerate(images):
            xref = img[0]
            base_img = doc.extract_image(xref)
            
            width = base_img["width"]
            height = base_img["height"]
            
            # Only keep images that are real diagrams (not tiny icons)
            if width >= min_width and height >= min_height:
                img_bytes = base_img["image"]
                ext = base_img["ext"]
                
                filename = f"{book_name}_page{page_num+1}_img{img_idx+1}.{ext}"
                filepath = os.path.join(output_dir, filename)
                
                with open(filepath, "wb") as f:
                    f.write(img_bytes)
                count += 1
    
    doc.close()
    print(f"Extracted {count} images from {book_name}")


# Run for all books
if __name__ == "__main__":
    books_dir = "E:\\Anato-tutor\\data\\textbooks"
    images_dir = "E:\\Anato-tutor\\data\\images"
    
    for filename in os.listdir(books_dir):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(books_dir, filename)
            extract_images_from_pdf(pdf_path, images_dir)