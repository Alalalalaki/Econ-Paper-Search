"""
Simplified semantic search - receives pre-filtered data and embeddings.
"""

import numpy as np
import pandas as pd
import streamlit as st
from sentence_transformers import SentenceTransformer


@st.cache_resource
def load_semantic_model():
    """Load sentence transformer model once and cache"""
    return SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')


def perform_semantic_search(query, filtered_df, filtered_embeddings, min_similarity=0.0):
    """
    Perform semantic search on pre-filtered papers and their embeddings.

    Args:
        query: Search query string
        filtered_df: Already filtered dataframe
        filtered_embeddings: Embeddings corresponding to filtered_df (same order!)
        min_similarity: Minimum similarity threshold

    Returns:
        DataFrame with results sorted by similarity score
    """
    if len(filtered_df) == 0 or not query.strip():
        return pd.DataFrame()

    # Load model
    model = load_semantic_model()

    # Encode query
    query_embedding = model.encode([query], normalize_embeddings=True)[0]

    # Compute similarities for all filtered embeddings
    similarities = np.dot(filtered_embeddings, query_embedding)

    # Filter by minimum similarity
    mask = similarities >= min_similarity

    if not mask.any():
        return pd.DataFrame()

    # Create results with proper index handling
    results = filtered_df[mask].copy()
    results['similarity'] = similarities[mask]

    # Sort by similarity (descending)
    results = results.sort_values('similarity', ascending=False)

    return results
