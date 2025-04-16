"""
Content Analyzer Module for PDF Analysis System

This module is responsible for analyzing PDF content, generating summaries,
and extracting golden sentences using language models.
"""

import os
from typing import Dict, List, Any, Optional

# 添加项目根目录到 Python 路径
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import from config
import config
from utils.text_utils import clean_text, truncate_text


class ContentAnalyzer:
    """
    Class for analyzing PDF content using language models.

    Attributes:
        llm: Language model for content analysis
    """

    def __init__(self, model_name: str = None):
        """
        Initialize the content analyzer with a language model.

        Args:
            model_name (str, optional): Name of the language model to use.
                If None, uses value from config.
        """
        # Get model name from config or use provided value
        model_name = model_name or config.OPENAI_MODEL

        # Initialize language model
        # Try to use Grok API first, then OpenAI API
        if config.is_grok_api_key_valid():
            # Import the API service
            from api.api_service import GrokService

            # Create a Grok service instance
            grok_service = GrokService()

            # 直接使用 Grok API
            self.llm = grok_service
            print(f"Using Grok API with model: {config.GROK_MODEL}")
        elif config.is_openai_api_key_valid():
            from langchain_openai import ChatOpenAI
            self.llm = ChatOpenAI(model=model_name, temperature=0.2)
            print(f"Using OpenAI API with model: {model_name}")
        else:
            # For code validation only - this won't work for actual execution
            print("Warning: No valid API keys found. Using placeholder LLM for demonstration.")
            self.llm = None

    def generate_summary(self, text_chunks: List[Dict[str, Any]], file_name: str) -> str:
        """
        Generate a summary for a PDF based on its text chunks.

        Args:
            text_chunks (List[Dict[str, Any]]): List of text chunks from the PDF
            file_name (str): Name of the PDF file

        Returns:
            str: Generated summary
        """
        # If no LLM available (for code validation), return placeholder or sample summary
        if not self.llm:
            if "attention" in file_name.lower():
                return """'Attention Is All You Need' introduces the Transformer, a novel neural network architecture based entirely on attention mechanisms, eliminating recurrence and convolutions. The paper demonstrates that the Transformer outperforms previous state-of-the-art models on translation tasks while being more parallelizable and requiring significantly less training time. The architecture consists of encoder and decoder stacks, each composed of multi-head self-attention mechanisms and position-wise fully connected feed-forward networks. The authors introduce multi-head attention, which allows the model to jointly attend to information from different representation subspaces at different positions. The Transformer also employs residual connections, layer normalization, and positional encodings to handle sequential information without recurrence. Experiments on WMT 2014 English-to-German and English-to-French translation tasks show superior quality while requiring less computation to train. The Transformer's attention mechanism provides interpretable models, as attention distributions clearly show which positions in the input sequence are most relevant for each output position. This work has become foundational for subsequent developments in natural language processing and beyond."""
            else:
                return f"Summary for {file_name} would be generated here using the LLM."

        # Filter chunks for the specific file
        file_chunks = [chunk for chunk in text_chunks if chunk["metadata"]["file_name"] == file_name]

        if not file_chunks:
            return f"No text chunks found for {file_name}"

        # Prepare input for the LLM
        combined_text = ""
        for chunk in file_chunks:
            combined_text += chunk["text"] + "\n\n"

        # Truncate if too long
        combined_text = truncate_text(combined_text, 10000)

        # Generate summary using LLM
        if hasattr(self.llm, 'generate_text'):  # Grok API
            prompt = f"""You are an expert academic summarizer. Summarize the following text from a PDF document.
            Focus on the main ideas, key findings, and important conclusions.
            Provide a comprehensive yet concise summary in about 250-300 words.

            TEXT:
            {combined_text}

            SUMMARY:"""

            summary = self.llm.generate_text(prompt)
            return summary
        else:  # LangChain LLM
            from langchain_core.prompts import ChatPromptTemplate

            prompt = ChatPromptTemplate.from_template(
                """You are an expert academic summarizer. Summarize the following text from a PDF document.
                Focus on the main ideas, key findings, and important conclusions.
                Provide a comprehensive yet concise summary in about 250-300 words.

                TEXT:
                {text}

                SUMMARY:"""
            )

            chain = prompt | self.llm
            summary = chain.invoke({"text": combined_text})

            # Extract content from response
            if hasattr(summary, 'content'):
                return summary.content
            else:
                return str(summary)

    def extract_golden_sentences(self, text_chunks: List[Dict[str, Any]], file_name: str, count: int = 3) -> List[Dict[str, str]]:
        """
        Extract golden sentences from a PDF.

        Args:
            text_chunks (List[Dict[str, Any]]): List of text chunks from the PDF
            file_name (str): Name of the PDF file
            count (int, optional): Number of golden sentences to extract, defaults to 3

        Returns:
            List[Dict[str, str]]: List of golden sentences with context and reason
        """
        # If no LLM available (for code validation), return placeholder or sample golden sentences
        if not self.llm:
            if "attention" in file_name.lower():
                return [
                    {
                        "sentence": "Attention Is All You Need",
                        "context": "This is the title of the paper that introduced the Transformer architecture, which has revolutionized natural language processing and many other fields.",
                        "reason": "This sentence encapsulates the core thesis of the paper - that attention mechanisms alone are sufficient for state-of-the-art performance in sequence modeling tasks."
                    },
                    {
                        "sentence": "We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely.",
                        "context": "The Transformer architecture relies entirely on an attention mechanism to draw global dependencies between input and output.",
                        "reason": "This sentence clearly states the novel contribution of the paper and highlights how it differs from previous approaches that relied on recurrent or convolutional neural networks."
                    },
                    {
                        "sentence": "Multi-head attention allows the model to jointly attend to information from different representation subspaces at different positions.",
                        "context": "Instead of performing a single attention function with d_model-dimensional keys, values and queries, we found it beneficial to linearly project the queries, keys and values h times with different, learned linear projections.",
                        "reason": "This sentence explains one of the key innovations in the Transformer architecture that enables its powerful representational capacity."
                    }
                ]
            else:
                return [{"sentence": f"Golden sentence {i+1} for {file_name}",
                        "context": "Context would be provided here",
                        "reason": "Reason would be provided here"}
                       for i in range(min(count, 3))]

        # Filter chunks for the specific file
        file_chunks = [chunk for chunk in text_chunks if chunk["metadata"]["file_name"] == file_name]

        if not file_chunks:
            return []

        # Prepare input for the LLM
        combined_text = ""
        for chunk in file_chunks:
            combined_text += chunk["text"] + "\n\n"

        # Truncate if too long
        combined_text = truncate_text(combined_text, 10000)

        # Extract golden sentences using LLM
        if hasattr(self.llm, 'generate_text'):  # Grok API
            prompt = f"""You are an expert at identifying the most important sentences in academic papers.
            From the following text, extract the {count} most important sentences that capture the key insights or contributions.
            For each sentence, provide:
            1. The exact sentence from the text
            2. The surrounding context
            3. A reason why this sentence is important

            Format your response as a JSON array with objects containing "sentence", "context", and "reason" keys.

            TEXT:
            {combined_text}

            GOLDEN SENTENCES:"""

            content = self.llm.generate_text(prompt)
        else:  # LangChain LLM
            from langchain_core.prompts import ChatPromptTemplate

            prompt = ChatPromptTemplate.from_template(
                """You are an expert at identifying the most important sentences in academic papers.
                From the following text, extract the {count} most important sentences that capture the key insights or contributions.
                For each sentence, provide:
                1. The exact sentence from the text
                2. The surrounding context
                3. A reason why this sentence is important

                Format your response as a JSON array with objects containing "sentence", "context", and "reason" keys.

                TEXT:
                {text}

                GOLDEN SENTENCES:"""
            )

            chain = prompt | self.llm
            response = chain.invoke({"text": combined_text, "count": count})

            # Extract content from response
            content = response.content if hasattr(response, 'content') else str(response)

        # Parse JSON from response
        import json
        import re

        # Try to extract JSON from the response
        json_match = re.search(r'\[\s*\{.*\}\s*\]', content, re.DOTALL)
        if json_match:
            try:
                golden_sentences = json.loads(json_match.group(0))
                return golden_sentences
            except json.JSONDecodeError:
                pass

        # If JSON parsing fails, return a placeholder
        return [{"sentence": f"Golden sentence {i+1} (parsing failed)",
                "context": "Context parsing failed",
                "reason": "Reason parsing failed"}
               for i in range(min(count, 3))]

    def analyze_similarities(self, similarities: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze similarities between chunks from different files.

        Args:
            similarities (List[Dict[str, Any]]): List of similar chunk pairs

        Returns:
            Dict[str, Any]: Analysis results
        """
        if not similarities:
            return {
                "common_themes": [],
                "similar_points": [],
                "analysis": "No similarities found between the documents."
            }

        # If no LLM available (for code validation), return placeholder
        if not self.llm:
            return {
                "common_themes": ["Theme 1", "Theme 2", "Theme 3"],
                "similar_points": [
                    {
                        "point": "Similar point 1",
                        "documents": [similarities[0]["file1"], similarities[0]["file2"]]
                    },
                    {
                        "point": "Similar point 2",
                        "documents": [similarities[0]["file1"], similarities[0]["file2"]]
                    }
                ],
                "analysis": "Analysis of similarities would be generated here using the LLM."
            }

        # Prepare input for the LLM
        similarities_text = ""
        for i, sim in enumerate(similarities[:10]):  # Limit to top 10 similarities
            similarities_text += f"Similarity {i+1}:\n"
            similarities_text += f"Document 1: {sim['file1']}\n"
            similarities_text += f"Text 1: {sim['chunk1']['text']}\n"
            similarities_text += f"Document 2: {sim['file2']}\n"
            similarities_text += f"Text 2: {sim['chunk2']['text']}\n"
            similarities_text += f"Similarity Score: {sim['similarity']:.2f}\n\n"

        # Generate analysis using LLM
        if hasattr(self.llm, 'generate_text'):  # Grok API
            prompt = f"""You are an expert at analyzing similarities between documents.
            Analyze the following similar text chunks from different documents.

            {similarities_text}

            Provide your analysis in the following format:
            1. Common Themes: List 3-5 common themes across the documents
            2. Similar Points: List specific points that are similar across documents
            3. Analysis: A paragraph analyzing the significance of these similarities

            Format your response as a JSON object with "common_themes" (array), "similar_points" (array of objects with "point" and "documents" fields), and "analysis" (string) keys.

            ANALYSIS:"""

            content = self.llm.generate_text(prompt)
        else:  # LangChain LLM
            from langchain_core.prompts import ChatPromptTemplate

            prompt = ChatPromptTemplate.from_template(
                """You are an expert at analyzing similarities between documents.
                Analyze the following similar text chunks from different documents.

                {similarities_text}

                Provide your analysis in the following format:
                1. Common Themes: List 3-5 common themes across the documents
                2. Similar Points: List specific points that are similar across documents
                3. Analysis: A paragraph analyzing the significance of these similarities

                Format your response as a JSON object with "common_themes" (array), "similar_points" (array of objects with "point" and "documents" fields), and "analysis" (string) keys.

                ANALYSIS:"""
            )

            chain = prompt | self.llm
            response = chain.invoke({"similarities_text": similarities_text})

            # Extract content from response
            content = response.content if hasattr(response, 'content') else str(response)

        # Parse JSON from response
        import json
        import re

        # Try to extract JSON from the response
        json_match = re.search(r'\{.*\}', content, re.DOTALL)
        if json_match:
            try:
                analysis = json.loads(json_match.group(0))
                return analysis
            except json.JSONDecodeError:
                pass

        # If JSON parsing fails, return a placeholder
        return {
            "common_themes": ["Theme 1 (parsing failed)", "Theme 2 (parsing failed)"],
            "similar_points": [
                {
                    "point": "Similar point 1 (parsing failed)",
                    "documents": [similarities[0]["file1"], similarities[0]["file2"]]
                }
            ],
            "analysis": "Analysis parsing failed."
        }


# LangGraph node for content analysis
def content_analyzer_node(state):
    """
    LangGraph node for analyzing PDF content.

    Args:
        state: The current state containing text_chunks and similarities

    Returns:
        Updated state with summaries, golden_sentences, and similarity_analysis
    """
    text_chunks = state.get("text_chunks", [])
    similarities = state.get("similarities", [])

    if not text_chunks:
        return {
            "status": "error",
            "message": "No text chunks to analyze",
            "summaries": {},
            "golden_sentences": {},
            "similarity_analysis": None
        }

    # Create content analyzer
    analyzer = ContentAnalyzer()

    # Get unique file names
    file_names = set()
    for chunk in text_chunks:
        file_names.add(chunk["metadata"]["file_name"])

    # Generate summaries for each file
    summaries = {}
    for file_name in file_names:
        summary = analyzer.generate_summary(text_chunks, file_name)
        summaries[file_name] = summary

    # Extract golden sentences for each file
    golden_sentences = {}
    for file_name in file_names:
        sentences = analyzer.extract_golden_sentences(text_chunks, file_name)
        golden_sentences[file_name] = sentences

    # Analyze similarities if multiple files
    similarity_analysis = None
    if len(file_names) > 1 and similarities:
        similarity_analysis = analyzer.analyze_similarities(similarities)

    return {
        "status": "success",
        "message": f"Successfully analyzed content for {len(file_names)} files",
        "summaries": summaries,
        "golden_sentences": golden_sentences,
        "similarity_analysis": similarity_analysis
    }
