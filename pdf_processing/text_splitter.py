"""
Text Splitter Module for PDF Analysis System

This module is responsible for splitting extracted PDF text into semantic chunks
with appropriate metadata for further processing and analysis.
"""

import re
from typing import Dict, List, Any

# 添加项目根目录到 Python 路径
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import from config
import config
from utils.text_utils import clean_text


class PDFTextSplitter:
    """
    Class for splitting PDF text into semantic chunks with metadata.

    Attributes:
        chunk_size (int): Target size of each text chunk in tokens
        chunk_overlap (int): Number of tokens to overlap between chunks
    """

    def __init__(self, chunk_size: int = None, chunk_overlap: int = None):
        """
        Initialize the text splitter with chunk parameters.

        Args:
            chunk_size (int, optional): Target size of each text chunk in tokens.
                If None, uses value from config.
            chunk_overlap (int, optional): Number of tokens to overlap between chunks.
                If None, uses value from config.
        """
        # Get chunk parameters from config or use provided values
        self.chunk_size = chunk_size if chunk_size is not None else config.CHUNK_SIZE
        self.chunk_overlap = chunk_overlap if chunk_overlap is not None else config.CHUNK_OVERLAP

        # Import here to avoid circular imports
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        # Initialize the RecursiveCharacterTextSplitter with appropriate parameters
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

    def estimate_page_number(self, text: str, chunk_text: str, avg_chars_per_page: int = 2000) -> int:
        """
        Estimate the page number for a chunk based on its position in the text.

        Args:
            text (str): Full text of the PDF
            chunk_text (str): Text of the chunk
            avg_chars_per_page (int, optional): Average characters per page, defaults to 2000

        Returns:
            int: Estimated page number (1-based)
        """
        # Find the position of the chunk in the text
        chunk_start = text.find(chunk_text)

        if chunk_start == -1:
            # If chunk not found exactly, try with cleaned text
            clean_chunk = clean_text(chunk_text)
            clean_full = clean_text(text)
            chunk_start = clean_full.find(clean_chunk)

            if chunk_start == -1:
                # If still not found, return page 1
                return 1

        # Estimate page number based on character position
        estimated_page = (chunk_start // avg_chars_per_page) + 1

        return max(1, estimated_page)

    def split_text(self, text: str, file_name: str) -> List[Dict[str, Any]]:
        """
        Split text into chunks with metadata.

        Args:
            text (str): Text to split
            file_name (str): Name of the file

        Returns:
            List[Dict[str, Any]]: List of dictionaries containing text chunks and metadata
        """
        # Get raw chunks from the splitter
        raw_chunks = self.splitter.split_text(text)

        # Add metadata to each chunk
        chunks_with_metadata = []
        for i, chunk in enumerate(raw_chunks):
            # Estimate page number
            estimated_page = self.estimate_page_number(text, chunk)

            # Create chunk with metadata
            chunk_with_metadata = {
                "text": chunk,
                "metadata": {
                    "file_name": file_name,
                    "chunk_index": i,
                    "estimated_page": estimated_page,
                    "chunk_size": len(chunk)
                }
            }
            chunks_with_metadata.append(chunk_with_metadata)

        return chunks_with_metadata

    def split_multiple_texts(self, texts: Dict[str, str]) -> List[Dict[str, Any]]:
        """
        Split multiple texts into chunks with metadata.

        Args:
            texts (Dict[str, str]): Dictionary mapping file names to texts

        Returns:
            List[Dict[str, Any]]: List of dictionaries containing text chunks and metadata
        """
        all_chunks = []

        for file_name, text in texts.items():
            chunks = self.split_text(text, file_name)
            all_chunks.extend(chunks)

        return all_chunks


# LangGraph node for text splitting
def text_splitter_node(state):
    """
    LangGraph node for splitting PDF texts into chunks.

    Args:
        state: The current state containing extracted_texts

    Returns:
        Updated state with text_chunks
    """
    extracted_texts = state.get("extracted_texts", {})

    if not extracted_texts:
        return {
            "status": "error",
            "message": "No extracted texts to split",
            "text_chunks": []
        }

    # Create text splitter with default parameters
    splitter = PDFTextSplitter()

    # Split all texts
    text_chunks = splitter.split_multiple_texts(extracted_texts)

    if not text_chunks:
        return {
            "status": "error",
            "message": "Failed to split texts into chunks",
            "text_chunks": []
        }

    return {
        "status": "success",
        "message": f"Successfully split texts into {len(text_chunks)} chunks",
        "text_chunks": text_chunks
    }
