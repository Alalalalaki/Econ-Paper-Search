"""
Shared data processing module to ensure consistency between embedding generation and app.
This module MUST be used by both generate_embedding.py and app.py to ensure
the exact same papers are loaded and processed.
"""

import pandas as pd
import numpy as np
import os
import logging

logger = logging.getLogger(__name__)


def load_single_file(file_path):
    """Load a single CSV file with consistent settings - exactly as in app.py"""
    # Define dtypes for consistency
    dtype_dict = {
        "year": "Int16",
        "journal": "category"
    }

    # Load with specific columns
    usecols = ["title", "authors", "abstract", "url", "journal", "year"]

    # Read CSV in chunks to reduce memory
    chunks = []
    for chunk in pd.read_csv(file_path, dtype=dtype_dict, usecols=usecols, chunksize=10000):
        # Convert object columns to categories where possible
        for col in chunk.select_dtypes(['object']).columns:
            if chunk[col].nunique() / len(chunk) < 0.5:  # If less than 50% unique values
                chunk[col] = chunk[col].astype('category')
        chunks.append(chunk)

    return pd.concat(chunks, ignore_index=True)


def clean_papers(df):
    """
    Apply consistent cleaning to papers dataframe.
    This exactly matches the cleaning in the original app.py
    """
    initial_count = len(df)

    # 1. Remove papers with missing years
    df = df[~df.year.isna()]
    after_year = len(df)
    logger.info(f"Removed {initial_count - after_year} papers with missing years")

    # 2. Convert to string dtypes (must be done before string operations)
    df['title'] = df['title'].astype('string')
    df['authors'] = df['authors'].astype('string')
    df['abstract'] = df['abstract'].astype('string')

    # 3. Drop book reviews (exactly as in original)
    masks = [~df.title.str.contains(i, case=False, regex=False) for i in ["pp.", " p."]]
    mask = np.vstack(masks).all(axis=0)
    df = df.loc[mask]
    after_reviews = len(df)
    logger.info(f"Removed {after_year - after_reviews} book reviews")

    # 4. Clean titles (exactly as in original - no na parameter)
    df.title = df.title.str.replace(r'\n\[.*?\]', '', regex=True)
    df.title = df.title.str.replace(r'\n', ' ', regex=True)

    # 5. Drop duplicates
    before_dup = len(df)
    df = df[~df.duplicated(['title', 'url']) | df.url.isna()]
    logger.info(f"Removed {before_dup - len(df)} duplicates")

    # 6. Replace broken links to None
    broken_links = ["http://hdl.handle.net/", "https://hdl.handle.net/"]
    df.loc[df.url.isin(broken_links), "url"] = None

    # Don't reset index here - let the caller decide
    logger.info(f"Total papers after cleaning: {len(df)} (removed {initial_count - len(df)})")

    return df


def load_all_papers(data_dir="Data"):
    """
    Load and clean all papers in the correct order.
    This exactly replicates the load_data_and_combine() function from app.py
    """
    args = {
        "dtype": {
            "year": "Int16",  # Using smaller integer type
            "journal": "category"  # Store journal as category
        },
        "usecols": ["title", "authors", "abstract", "url", "journal", "year"]
    }

    # Load files one by one to prevent memory spike
    dfs = []
    file_periods = ['b2000', '2000s', '2010s', '2015s', '2020s']

    for period in file_periods:
        file_path = f"{data_dir}/papers_{period}.csv"
        if os.path.exists(file_path):
            logger.info(f"Loading {period}...")
            df = load_single_file(file_path)
            logger.info(f"Loaded {len(df)} papers from {period}")
            dfs.append(df)
        else:
            logger.warning(f"File not found: {file_path}")

    df = pd.concat(dfs, ignore_index=True)
    logger.info(f"Total papers before cleaning: {len(df)}")

    # Clean data - exactly as in app.py
    df = df[~df.year.isna()]

    # Use string dtype instead of object
    df['title'] = df['title'].astype('string')
    df['authors'] = df['authors'].astype('string')
    df['abstract'] = df['abstract'].astype('string')

    # drop book reviews (not perfect)
    masks = [~df.title.str.contains(i, case=False, regex=False) for i in ["pp.", " p."]]  # "pages," " pp "
    mask = np.vstack(masks).all(axis=0)
    df = df.loc[mask]

    # clean titles
    df.title = df.title.str.replace(r'\n\[.*?\]', '', regex=True)
    df.title = df.title.str.replace(r'\n', ' ', regex=True)

    # drop some duplicates due to weird strings in authors and abstract
    df = df[~df.duplicated(['title', 'url']) | df.url.isna()]

    # replace broken links to None
    broken_links = ["http://hdl.handle.net/", "https://hdl.handle.net/"]
    df.loc[df.url.isin(broken_links), "url"] = None

    logger.info(f"Total papers after cleaning: {len(df)}")

    return df
