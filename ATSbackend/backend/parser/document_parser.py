
import os
import pdfplumber
from docx import Document
import logging

logger = logging.getLogger(__name__)

def parse_document(path: str) -> str:
    """
    Parse document and extract text.
    Supports: PDF (with OCR fallback), DOCX, TXT
    """
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        return _parse_pdf(path)
    if ext == ".docx":
        return _parse_docx(path)
    if ext == ".txt":
        return _parse_txt(path)
    raise ValueError(f"Unsupported file type: '{ext}'. Use PDF, DOCX, or TXT.")


def _parse_pdf(path):
    """
    Extract text from PDF with OCR fallback.
    
    Primary: pdfplumber (fast, text-based PDFs)
    Fallback: tesseract OCR (for image-heavy or scanned PDFs)
    """
    parts = []
    
    # Try primary extraction with pdfplumber
    try:
        with pdfplumber.open(path) as pdf:
            for i, page in enumerate(pdf.pages):
                text = page.extract_text()
                if text and text.strip():
                    parts.append(text)
    except Exception as e:
        logger.warning(f"pdfplumber extraction failed: {e}")
    
    # If primary extraction yielded text, return it
    if parts:
        result = "\n".join(parts)
        if result.strip():
            logger.debug(f"Extracted {len(result)} chars from PDF via pdfplumber")
            return result
    
    # Fallback: Try OCR extraction with pytesseract
    # Before attempting OCR (which requires system tesseract), try a pure-Python
    # text extraction using PyPDF2 as a best-effort fallback. This avoids
    # forcing a heavy system dependency during tests and in lightweight envs.
    try:
        from PyPDF2 import PdfReader
        reader = PdfReader(path)
        text_parts = []
        for p in reader.pages:
            try:
                t = p.extract_text() or ""
            except Exception:
                t = ""
            if t and t.strip():
                text_parts.append(t)
        if text_parts:
            result = "\n".join(text_parts)
            logger.debug(f"Extracted {len(result)} chars from PDF via PyPDF2")
            return result
    except Exception as e:
        logger.debug(f"PyPDF2 fallback not available or failed: {e}")

    logger.info(f"PDF extraction empty, attempting OCR fallback for {path}")
    return _parse_pdf_ocr(path)


def _parse_pdf_ocr(path):
    """
    Extract text from PDF using OCR (tesseract).
    Converts PDF pages to images and runs OCR.
    """
    try:
        from pdf2image import convert_from_path
        import pytesseract
    except ImportError:
        logger.error(
            "OCR fallback requires pdf2image and pytesseract. "
            "Install: pip install pdf2image pytesseract. "
            "Also install tesseract: https://github.com/UB-Mannheim/tesseract/wiki"
        )
        return ""
    
    try:
        images = convert_from_path(path)
        parts = []
        
        for i, image in enumerate(images):
            text = pytesseract.image_to_string(image)
            if text and text.strip():
                parts.append(text)
                logger.debug(f"OCR extracted page {i+1}")
        
        if parts:
            result = "\n".join(parts)
            logger.info(f"OCR extracted {len(result)} chars from {len(images)} pages")
            return result
        else:
            logger.warning(f"OCR returned empty text for {path}")
            return ""
    
    except Exception as e:
        logger.error(f"OCR extraction failed: {e}")
        return ""


def _parse_docx(path):
    """Extract text from DOCX."""
    try:
        doc = Document(path)
        text = "\n".join(p.text for p in doc.paragraphs if p.text.strip())
        logger.debug(f"Extracted {len(text)} chars from DOCX")
        return text
    except Exception as e:
        logger.error(f"DOCX parsing failed: {e}")
        return ""


def _parse_txt(path):
    """Extract text from TXT."""
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
        logger.debug(f"Extracted {len(text)} chars from TXT")
        return text
    except Exception as e:
        logger.error(f"TXT parsing failed: {e}")
        return ""
