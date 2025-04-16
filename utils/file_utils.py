"""
File Utility Functions for PDF Analysis System

This module provides utility functions for file operations.
"""

import os
import json


def ensure_directory_exists(directory_path):
    """
    Ensure that a directory exists, creating it if necessary.

    Args:
        directory_path (str): Path to the directory
    """
    os.makedirs(directory_path, exist_ok=True)


def save_json(data, file_path):
    """
    Save data as JSON to a file.

    Args:
        data (Any): Data to save
        file_path (str): Path to the file

    Returns:
        bool: Success status
    """
    try:
        # Ensure directory exists
        directory = os.path.dirname(file_path)
        if directory:
            ensure_directory_exists(directory)

        # Save data
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        return True
    except Exception as e:
        print(f"Error saving JSON data: {str(e)}")
        return False


def load_json(file_path, default=None):
    """
    Load JSON data from a file.

    Args:
        file_path (str): Path to the file
        default (Any, optional): Default value to return if file doesn't exist or loading fails

    Returns:
        Any: Loaded data or default value
    """
    if not os.path.exists(file_path):
        return default

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading JSON data: {str(e)}")
        return default


def save_text(text, file_path):
    """
    Save text to a file.

    Args:
        text (str): Text to save
        file_path (str): Path to the file

    Returns:
        bool: Success status
    """
    try:
        # Ensure directory exists
        directory = os.path.dirname(file_path)
        if directory:
            ensure_directory_exists(directory)

        # Save text
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(text)
        return True
    except Exception as e:
        print(f"Error saving text: {str(e)}")
        return False


def load_text(file_path, default=""):
    """
    Load text from a file.

    Args:
        file_path (str): Path to the file
        default (str, optional): Default value to return if file doesn't exist or loading fails

    Returns:
        str: Loaded text or default value
    """
    if not os.path.exists(file_path):
        return default

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        print(f"Error loading text: {str(e)}")
        return default
