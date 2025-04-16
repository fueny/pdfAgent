"""
PDF Processor Module for PDF Analysis System

This module is responsible for reading PDF files and extracting text content.
It includes validation for file size and handles edge cases like scanned PDFs.
"""

import os
import pdfplumber
import fitz  # PyMuPDF
import warnings
import sys
import io
from contextlib import contextmanager
from typing import Dict, List, Optional, Tuple

# 抑制 PyPDF2 的警告信息
warnings.filterwarnings("ignore", category=UserWarning, module="PyPDF2")
warnings.filterwarnings("ignore", category=UserWarning, module="pdfminer")
warnings.filterwarnings("ignore", category=UserWarning, module="pdfplumber")

# 创建一个上下文管理器来隐藏 stderr 输出
@contextmanager
def suppress_stderr():
    """A context manager that redirects stderr to devnull"""
    stderr = sys.stderr
    sys.stderr = io.StringIO()
    try:
        yield
    finally:
        sys.stderr = stderr

# 添加项目根目录到 Python 路径
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import from config
import config
from utils.file_utils import ensure_directory_exists


class PDFProcessor:
    """
    Class for processing PDF files and extracting text content.

    Attributes:
        max_file_size_mb (int): Maximum allowed file size in MB
    """

    def __init__(self, max_file_size_mb: int = None):
        """
        Initialize the PDF processor with size constraints.

        Args:
            max_file_size_mb (int, optional): Maximum allowed file size in MB.
                If None, uses value from config.
        """
        # Get max file size from config or use provided value
        self.max_file_size_mb = max_file_size_mb if max_file_size_mb is not None else config.MAX_PDF_SIZE_MB

    def validate_file_size(self, file_path: str) -> Tuple[bool, str]:
        """
        Validate if the file size is within the allowed limit.

        Args:
            file_path (str): Path to the PDF file

        Returns:
            Tuple[bool, str]: (success, message)
        """
        if not os.path.exists(file_path):
            return False, f"File not found: {file_path}"

        # Get file size in MB
        file_size_mb = os.path.getsize(file_path) / (1024 * 1024)

        if file_size_mb > self.max_file_size_mb:
            return False, f"File size ({file_size_mb:.2f} MB) exceeds the maximum allowed size ({self.max_file_size_mb} MB)"

        return True, f"File size ({file_size_mb:.2f} MB) is within the allowed limit"

    def extract_text_with_pdfplumber(self, file_path: str) -> str:
        """
        Extract text from PDF using pdfplumber.

        Args:
            file_path (str): Path to the PDF file

        Returns:
            str: Extracted text content
        """
        text = ""
        try:
            # 使用上下文管理器隐藏 stderr 输出
            with suppress_stderr():
                with pdfplumber.open(file_path) as pdf:
                    for page in pdf.pages:
                        page_text = page.extract_text() or ""
                        text += page_text + "\n\n"
        except Exception as e:
            print(f"Error extracting text with pdfplumber: {str(e)}")
            return ""

        return text

    def extract_text_with_pymupdf(self, file_path: str) -> str:
        """
        Extract text from PDF using PyMuPDF (fitz).

        Args:
            file_path (str): Path to the PDF file

        Returns:
            str: Extracted text content
        """
        text = ""
        try:
            # 使用上下文管理器隐藏 stderr 输出
            with suppress_stderr():
                with fitz.open(file_path) as doc:
                    for page in doc:
                        text += page.get_text() + "\n\n"
        except Exception as e:
            print(f"Error extracting text with PyMuPDF: {str(e)}")
            return ""

        return text

    def extract_text(self, file_path: str) -> str:
        """
        Extract text from PDF using multiple methods for robustness.

        Args:
            file_path (str): Path to the PDF file

        Returns:
            str: Extracted text content
        """
        # Try pdfplumber first
        text = self.extract_text_with_pdfplumber(file_path)

        # If pdfplumber fails or returns empty text, try PyMuPDF
        if not text.strip():
            text = self.extract_text_with_pymupdf(file_path)

        return text

    def process_pdf(self, file_path: str) -> Tuple[bool, str, Optional[str]]:
        """
        Process a PDF file: validate size and extract text.

        Args:
            file_path (str): Path to the PDF file

        Returns:
            Tuple[bool, str, Optional[str]]: (success, message, extracted_text)
        """
        # Validate file size
        is_valid, message = self.validate_file_size(file_path)
        if not is_valid:
            return False, message, None

        # Extract text
        text = self.extract_text(file_path)

        if not text.strip():
            return False, "Failed to extract text from PDF. The file might be scanned or corrupted.", None

        return True, "PDF processed successfully", text

    def process_multiple_pdfs(self, file_paths: List[str]) -> Dict[str, str]:
        """
        Process multiple PDF files.

        Args:
            file_paths (List[str]): List of paths to PDF files

        Returns:
            Dict[str, str]: Dictionary mapping file names to extracted text
        """
        results = {}

        for file_path in file_paths:
            # Get file name from path
            file_name = os.path.basename(file_path)

            # Process PDF
            success, message, text = self.process_pdf(file_path)

            if success and text:
                results[file_name] = text
            else:
                print(f"Failed to process {file_name}: {message}")

        return results


# LangGraph node for PDF processing
def pdf_processor_node(state):
    """
    LangGraph node for processing PDF files.

    Args:
        state: The current state containing file_paths

    Returns:
        Updated state with extracted_texts
    """
    file_paths = state.get("file_paths", [])

    if not file_paths:
        return {
            "status": "error",
            "message": "No PDF files provided",
            "extracted_texts": {}
        }

    processor = PDFProcessor()
    extracted_texts = processor.process_multiple_pdfs(file_paths)

    if not extracted_texts:
        return {
            "status": "error",
            "message": "Failed to extract text from any of the provided PDFs",
            "extracted_texts": {}
        }

    return {
        "status": "success",
        "message": f"Successfully processed {len(extracted_texts)} PDF files",
        "extracted_texts": extracted_texts
    }
