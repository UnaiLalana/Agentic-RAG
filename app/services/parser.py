import io
import pdfplumber
from docx import Document


class ParserError(Exception):
    """Raised when document parsing fails."""
    pass


def parse_pdf(file_data: bytes) -> str:
    """
    Extract text from a PDF file.

    Raises ParserError if the PDF appears to be scanned (no extractable text).
    """
    text_parts = []
    with pdfplumber.open(io.BytesIO(file_data)) as pdf:
        for page_num, page in enumerate(pdf.pages, 1):
            page_text = page.extract_text()
            if page_text:
                text_parts.append(page_text)

    full_text = "\n\n".join(text_parts).strip()

    if not full_text:
        raise ParserError(
            "No extractable text found in PDF. "
            "This appears to be a scanned/image-only PDF. "
            "Please upload a text-based PDF or a DOCX file."
        )

    return full_text


def parse_docx(file_data: bytes) -> str:
    """
    Extract text from a DOCX file.

    Raises ParserError if the document contains no text.
    """
    doc = Document(io.BytesIO(file_data))
    paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
    full_text = "\n\n".join(paragraphs).strip()

    if not full_text:
        raise ParserError(
            "No text content found in the DOCX file. "
            "The document appears to be empty."
        )

    return full_text


def parse_document(filename: str, file_data: bytes) -> str:
    """
    Parse a document based on its file extension.

    Supports: .pdf, .docx
    Returns the extracted text.
    """
    ext = filename.lower().rsplit(".", 1)[-1] if "." in filename else ""

    if ext == "pdf":
        return parse_pdf(file_data)
    elif ext == "docx":
        return parse_docx(file_data)
    else:
        raise ParserError(
            f"Unsupported file format: .{ext}. "
            "Please upload a PDF or DOCX file."
        )
