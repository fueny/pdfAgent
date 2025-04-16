"""
Text Utility Functions for PDF Analysis System

This module provides utility functions for text processing.
"""

import re


def clean_text(text):
    """
    Clean text by removing extra whitespace and normalizing line breaks.
    
    Args:
        text (str): Text to clean
        
    Returns:
        str: Cleaned text
    """
    if not text:
        return ""
        
    # Replace multiple whitespace with a single space
    text = re.sub(r'\s+', ' ', text)
    
    # Remove leading/trailing whitespace
    text = text.strip()
    
    return text


def truncate_text(text, max_length=1000, add_ellipsis=True):
    """
    Truncate text to a maximum length.
    
    Args:
        text (str): Text to truncate
        max_length (int, optional): Maximum length, defaults to 1000
        add_ellipsis (bool, optional): Whether to add ellipsis, defaults to True
        
    Returns:
        str: Truncated text
    """
    if not text or len(text) <= max_length:
        return text
        
    truncated = text[:max_length]
    
    if add_ellipsis:
        truncated += "..."
        
    return truncated


def extract_sentences(text, min_length=10, max_length=200):
    """
    Extract sentences from text.
    
    Args:
        text (str): Text to extract sentences from
        min_length (int, optional): Minimum sentence length, defaults to 10
        max_length (int, optional): Maximum sentence length, defaults to 200
        
    Returns:
        List[str]: List of sentences
    """
    if not text:
        return []
        
    # Simple sentence splitting by period, question mark, or exclamation mark
    # followed by a space or newline
    sentence_pattern = r'[.!?][\s\n]+'
    sentences = re.split(sentence_pattern, text)
    
    # Clean and filter sentences
    cleaned_sentences = []
    for sentence in sentences:
        sentence = clean_text(sentence)
        if sentence and min_length <= len(sentence) <= max_length:
            cleaned_sentences.append(sentence)
            
    return cleaned_sentences
