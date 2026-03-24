import pdfplumber
import sys

def extract(pdf_path, num_pages=10):
    try:
        with pdfplumber.open(pdf_path) as pdf:
            pages = pdf.pages[:num_pages]
            text = [page.extract_text() for page in pages]
        for i, t in enumerate(text):
            print(f"--- Page {i+1} ---")
            if t:
                print(t[:1000])  # printing up to 1000 chars per page to avoid overload
    except Exception as e:
        print(f"Error reading {pdf_path}: {e}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        extract(sys.argv[1])
    else:
        print("Provide a PDF path")
