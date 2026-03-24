import pdfplumber
import sys

def extract(pdf_path, out_path):
    with pdfplumber.open(pdf_path) as pdf:
        text = "\n".join([page.extract_text() or "" for page in pdf.pages])
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(text)

if __name__ == "__main__":
    extract(sys.argv[1], sys.argv[2])
