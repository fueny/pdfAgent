"""
Memory Manager Module for PDF Analysis System

This module is responsible for managing persistent memory across sessions,
storing and retrieving analysis results, and tracking document history.
"""

import os
import time
import json
import pickle
from typing import Dict, List, Any, Optional

# 添加项目根目录到 Python 路径
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import from config
import config
from utils.file_utils import ensure_directory_exists, save_json, load_json


class MemoryManager:
    """
    Class for managing persistent memory across sessions.

    Attributes:
        memory_dir (str): Directory for storing memory files
        document_history (Dict[str, List[Dict]]): History of analyzed documents
        analysis_cache (Dict[str, Dict]): Cache of analysis results
    """

    def __init__(self, memory_dir: str = None):
        """
        Initialize the memory manager.

        Args:
            memory_dir (str, optional): Directory for storing memory files.
                If None, uses value from config.
        """
        # Set memory directory from config or use provided value
        self.memory_dir = memory_dir if memory_dir is not None else config.MEMORY_DIR
        ensure_directory_exists(self.memory_dir)

        # Initialize document history
        self.document_history = self._load_document_history()

        # Initialize analysis cache
        self.analysis_cache = {}

    def _load_document_history(self) -> Dict[str, List[Dict]]:
        """
        Load document history from disk.

        Returns:
            Dict[str, List[Dict]]: Document history
        """
        history_path = os.path.join(self.memory_dir, "document_history.json")
        return load_json(history_path, {})

    def _save_document_history(self) -> bool:
        """
        Save document history to disk.

        Returns:
            bool: Success status
        """
        history_path = os.path.join(self.memory_dir, "document_history.json")
        return save_json(self.document_history, history_path)

    def add_document(self, file_path: str, metadata: Dict[str, Any] = None) -> bool:
        """
        Add a document to the history.

        Args:
            file_path (str): Path to the document
            metadata (Dict[str, Any], optional): Additional metadata

        Returns:
            bool: Success status
        """
        # Get file name from path
        file_name = os.path.basename(file_path)

        # Create entry
        entry = {
            "file_name": file_name,
            "file_path": file_path,
            "timestamp": time.time(),
            "date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "metadata": metadata or {}
        }

        # Add to history
        if file_name not in self.document_history:
            self.document_history[file_name] = []

        self.document_history[file_name].append(entry)

        # Save history
        return self._save_document_history()

    def get_document_history(self, file_name: str = None) -> Dict[str, List[Dict]]:
        """
        Get document history.

        Args:
            file_name (str, optional): Name of the document. If None, returns all history.

        Returns:
            Dict[str, List[Dict]]: Document history
        """
        if file_name:
            return {file_name: self.document_history.get(file_name, [])}
        else:
            return self.document_history

    def save_analysis_result(self, file_name: str, result: Dict[str, Any]) -> bool:
        """
        Save analysis result for a document.

        Args:
            file_name (str): Name of the document
            result (Dict[str, Any]): Analysis result

        Returns:
            bool: Success status
        """
        # Create result directory
        result_dir = os.path.join(self.memory_dir, "results")
        ensure_directory_exists(result_dir)

        # Create file name with timestamp
        timestamp = int(time.time())
        result_path = os.path.join(result_dir, f"{file_name}_{timestamp}.json")

        # Add timestamp to result
        result_with_meta = result.copy()
        result_with_meta["_timestamp"] = timestamp
        result_with_meta["_date"] = time.strftime("%Y-%m-%d %H:%M:%S")

        # Save result
        success = save_json(result_with_meta, result_path)

        # Update cache
        if success:
            self.analysis_cache[file_name] = result

        return success

    def get_analysis_result(self, file_name: str, use_cache: bool = True) -> Optional[Dict[str, Any]]:
        """
        Get the latest analysis result for a document.

        Args:
            file_name (str): Name of the document
            use_cache (bool, optional): Whether to use cached result, defaults to True

        Returns:
            Optional[Dict[str, Any]]: Analysis result or None if not found
        """
        # Check cache first
        if use_cache and file_name in self.analysis_cache:
            return self.analysis_cache[file_name]

        # Get result directory
        result_dir = os.path.join(self.memory_dir, "results")
        if not os.path.exists(result_dir):
            return None

        # Find all result files for the document
        result_files = []
        for file in os.listdir(result_dir):
            if file.startswith(f"{file_name}_") and file.endswith(".json"):
                result_files.append(file)

        if not result_files:
            return None

        # Sort by timestamp (newest first)
        result_files.sort(reverse=True)

        # Load the latest result
        latest_result_path = os.path.join(result_dir, result_files[0])
        result = load_json(latest_result_path)

        # Update cache
        if result:
            self.analysis_cache[file_name] = result

        return result

    def get_all_analysis_results(self) -> Dict[str, Dict[str, Any]]:
        """
        Get all analysis results.

        Returns:
            Dict[str, Dict[str, Any]]: Dictionary mapping file names to analysis results
        """
        # Get result directory
        result_dir = os.path.join(self.memory_dir, "results")
        if not os.path.exists(result_dir):
            return {}

        # Group result files by document
        result_files_by_doc = {}
        for file in os.listdir(result_dir):
            if file.endswith(".json"):
                parts = file.split("_")
                if len(parts) >= 2:
                    doc_name = "_".join(parts[:-1])
                    if doc_name not in result_files_by_doc:
                        result_files_by_doc[doc_name] = []
                    result_files_by_doc[doc_name].append(file)

        # Get latest result for each document
        results = {}
        for doc_name, files in result_files_by_doc.items():
            # Sort by timestamp (newest first)
            files.sort(reverse=True)

            # Load the latest result
            latest_result_path = os.path.join(result_dir, files[0])
            result = load_json(latest_result_path)

            if result:
                results[doc_name] = result

        return results

    def clear_cache(self) -> None:
        """
        Clear the analysis cache.
        """
        self.analysis_cache = {}


# LangGraph nodes for memory management
def initialize_memory_node(state):
    """
    LangGraph node for initializing memory.

    Args:
        state: The current state

    Returns:
        Updated state with memory_manager
    """
    # Create memory manager
    memory_manager = MemoryManager()

    # Add documents to history
    file_paths = state.get("file_paths", [])
    for file_path in file_paths:
        memory_manager.add_document(file_path)

    return {
        "status": "success",
        "message": "Memory initialized",
        "memory_manager": memory_manager
    }


def save_results_node(state):
    """
    LangGraph node for saving analysis results.

    Args:
        state: The current state containing analysis results

    Returns:
        Updated state
    """
    memory_manager = state.get("memory_manager")
    if not memory_manager:
        return {
            "status": "error",
            "message": "Memory manager not initialized"
        }

    # Get results to save
    summaries = state.get("summaries", {})
    golden_sentences = state.get("golden_sentences", {})
    similarity_analysis = state.get("similarity_analysis")

    # Create result object
    result = {
        "status": state.get("status", "unknown"),
        "message": state.get("message", ""),
        "summaries": summaries,
        "golden_sentences": golden_sentences,
        "similarity_analysis": similarity_analysis
    }

    # Save results for each file
    for file_name in summaries.keys():
        memory_manager.save_analysis_result(file_name, result)

    # Generate output file
    from utils.formatting import format_analysis_results
    output_text = format_analysis_results(result)

    # Save output file
    output_path = config.OUTPUT_FILE
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(output_text)

    return {
        "status": "success",
        "message": f"Results saved to {output_path}",
        "output_path": output_path
    }


# Combined node for memory management
def memory_manager_node(state):
    """
    Combined LangGraph node for memory management.

    Args:
        state: The current state

    Returns:
        Updated state
    """
    # Initialize memory if needed
    if "memory_manager" not in state:
        memory_state = initialize_memory_node(state)
        state = {**state, **memory_state}

    # Save results if analysis is complete
    if "summaries" in state and "golden_sentences" in state:
        result_state = save_results_node(state)
        state = {**state, **result_state}

    return state
