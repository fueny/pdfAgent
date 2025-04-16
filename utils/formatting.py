"""
Formatting Utilities for PDF Analysis System

This module provides utility functions for formatting output.
"""


def format_analysis_results(results):
    """
    Format analysis results as a readable markdown string.
    
    Args:
        results (Dict[str, Any]): Analysis results
        
    Returns:
        str: Formatted results
    """
    output = []
    
    # Add header
    output.append("# PDF Analysis Results\n")
    
    # Add status
    status = results.get("status", "unknown")
    message = results.get("message", "No message provided")
    output.append(f"**Status:** {status}")
    output.append(f"**Message:** {message}\n")
    
    # Add summaries
    summaries = results.get("summaries", {})
    if summaries:
        output.append("## PDF Summaries\n")
        for file_name, summary in summaries.items():
            output.append(f"### {file_name}\n")
            output.append(summary)
            output.append("\n")
    
    # Add golden sentences
    golden_sentences = results.get("golden_sentences", {})
    if golden_sentences:
        output.append("## Golden Sentences\n")
        for file_name, sentences in golden_sentences.items():
            output.append(f"### {file_name}\n")
            for i, sentence in enumerate(sentences):
                output.append(f"#### Golden Sentence {i+1}\n")
                output.append(f"**Sentence:** {sentence['sentence']}\n")
                output.append(f"**Context:** {sentence['context']}\n")
                output.append(f"**Reason:** {sentence['reason']}\n")
            output.append("\n")
    
    # Add similarity analysis
    similarity_analysis = results.get("similarity_analysis")
    if similarity_analysis:
        output.append("## Cross-PDF Similarity Analysis\n")
        
        # Add common themes
        common_themes = similarity_analysis.get("common_themes", [])
        if common_themes:
            output.append("### Common Themes\n")
            for theme in common_themes:
                output.append(f"- {theme}\n")
            output.append("\n")
        
        # Add similar points
        similar_points = similarity_analysis.get("similar_points", [])
        if similar_points:
            output.append("### Similar Points\n")
            for point in similar_points:
                output.append(f"- **Point:** {point.get('point', '')}\n")
                output.append(f"  **Documents:** {', '.join(point.get('documents', []))}\n")
            output.append("\n")
        
        # Add analysis
        analysis = similarity_analysis.get("analysis", "")
        if analysis:
            output.append("### Analysis\n")
            output.append(analysis)
            output.append("\n")
    
    return "\n".join(output)
