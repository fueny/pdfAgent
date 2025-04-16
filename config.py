"""
Configuration Module for PDF Analysis System

This module centralizes all configuration settings and environment variables.
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# LLM API Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4-turbo")
GROK_API_KEY = os.getenv("GROK_API_KEY", "")
GROK_MODEL = os.getenv("GROK_MODEL", "grok-3")

# LLM Service Configuration
DEFAULT_LLM_SERVICE = os.getenv("DEFAULT_LLM_SERVICE", "auto")  # 可选值: "auto", "openai", "grok"

# PDF Processing Configuration
MAX_PDF_SIZE_MB = int(os.getenv("MAX_PDF_SIZE_MB", 50))

# Text Splitting Configuration
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 800))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 200))

# Vector Database Configuration
VECTOR_DB_PATH = os.getenv("VECTOR_DB_PATH", "./vector_db")
VECTOR_MODEL_NAME = os.getenv("VECTOR_MODEL_NAME", "all-MiniLM-L6-v2")

# Memory Management Configuration
MEMORY_DIR = os.getenv("MEMORY_DIR", "./memory")

# Logging Configuration
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# Output Configuration
OUTPUT_FILE = os.getenv("OUTPUT_FILE", "pdf_analysis_results.md")

# Check if OpenAI API key is valid
def is_openai_api_key_valid():
    """Check if the OpenAI API key is valid (not empty and not the placeholder)"""
    return OPENAI_API_KEY and OPENAI_API_KEY != "your_openai_api_key_here"

# Check if Grok API key is valid
def is_grok_api_key_valid():
    """Check if the Grok API key is valid (not empty and not the placeholder)"""
    return GROK_API_KEY and GROK_API_KEY != "your_grok_api_key_here"
