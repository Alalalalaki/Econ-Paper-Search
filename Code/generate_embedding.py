"""
Generate embeddings for all economics papers.

This script creates semantic embeddings for paper search functionality.
Uses the shared data_processing module to ensure consistency with app.py.
"""

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import os
import sys
from datetime import datetime
import json
from tqdm import tqdm
import logging

from data_processing import load_all_papers

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)


class EmbeddingGenerator:
    """Generate and save embeddings for economics papers"""

    def __init__(self, model_name='sentence-transformers/all-MiniLM-L6-v2'):
        """Initialize the embedding generator"""
        logger.info(f"Loading model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        logger.info(f"Model loaded. Embedding dimension: {self.embedding_dim}")

    def create_embeddings(self, df, batch_size=32):
        """Create embeddings for all papers"""
        # Combine title and abstract (same as search logic in app.py)
        texts = (df['title'] + ' ' + df['abstract'].fillna('')).tolist()

        logger.info(f"Creating embeddings for {len(texts)} papers...")

        # Generate embeddings with progress bar
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True  # Pre-normalize for cosine similarity
        )

        return embeddings


def main():
    """Main function to generate all embeddings"""

    # Check if we're in the right directory
    if not os.path.exists('../Data'):
        logger.error("Error: Cannot find Data directory. Please run this script from the Code directory.")
        logger.error("Current directory: " + os.getcwd())
        sys.exit(1)

    # Configuration
    MODEL_NAME = 'sentence-transformers/all-MiniLM-L6-v2'
    BATCH_SIZE = 32

    # Create Embeddings directory
    embeddings_dir = '../Embeddings'
    os.makedirs(embeddings_dir, exist_ok=True)
    logger.info(f"Output directory: {embeddings_dir}")

    # Initialize generator
    generator = EmbeddingGenerator(MODEL_NAME)

    # Load all papers using shared processing
    logger.info("Loading all papers using shared data processing...")
    df_all = load_all_papers(data_dir="../Data")
    logger.info(f"Total papers loaded: {len(df_all)}")

    # Dictionary to store all metadata
    all_metadata = {
        'model_name': MODEL_NAME,
        'embedding_dim': generator.embedding_dim,
        'creation_date': datetime.now().isoformat(),
        'last_update': datetime.now().isoformat(),
        'total_papers': len(df_all),
        'note': 'b2000 embeddings are split by index (not year) to maintain order and stay under 50MB',
        'embedding_structure': ['b2000_part1', 'b2000_part2', '2000s', '2010s', '2015s', '2020s'],
        'files': {}
    }

    # Process b2000 papers (all papers before 2000)
    # Split by index to maintain order, not by year!
    df_b2000 = df_all[df_all.year < 2000]

    if len(df_b2000) > 0:
        logger.info(f"\nProcessing pre-2000 papers (total: {len(df_b2000)})")

        # Split in the middle by index to keep files under 50MB
        split_index = len(df_b2000) // 2

        # First half
        df_b2000_part1 = df_b2000.iloc[:split_index]
        logger.info(f"Generating embeddings for b2000_part1 ({len(df_b2000_part1)} papers)...")
        embeddings_part1 = generator.create_embeddings(df_b2000_part1, batch_size=BATCH_SIZE)
        embeddings_part1_float16 = embeddings_part1.astype(np.float16)
        np.save('../Embeddings/embeddings_b2000_part1.npy', embeddings_part1_float16)

        file_size_part1 = os.path.getsize('../Embeddings/embeddings_b2000_part1.npy') / (1024 * 1024)
        all_metadata['files']['b2000_part1'] = {
            'num_papers': len(df_b2000_part1),
            'file_size_mb': file_size_part1,
            'year_range': f'{df_b2000_part1.year.min()}-{df_b2000_part1.year.max()}',
            'index_range': f'0-{split_index-1} of b2000'
        }
        logger.info(f"Saved embeddings_b2000_part1.npy: {file_size_part1:.1f} MB")

        # Second half
        df_b2000_part2 = df_b2000.iloc[split_index:]
        logger.info(f"Generating embeddings for b2000_part2 ({len(df_b2000_part2)} papers)...")
        embeddings_part2 = generator.create_embeddings(df_b2000_part2, batch_size=BATCH_SIZE)
        embeddings_part2_float16 = embeddings_part2.astype(np.float16)
        np.save('../Embeddings/embeddings_b2000_part2.npy', embeddings_part2_float16)

        file_size_part2 = os.path.getsize('../Embeddings/embeddings_b2000_part2.npy') / (1024 * 1024)
        all_metadata['files']['b2000_part2'] = {
            'num_papers': len(df_b2000_part2),
            'file_size_mb': file_size_part2,
            'year_range': f'{df_b2000_part2.year.min()}-{df_b2000_part2.year.max()}',
            'index_range': f'{split_index}-{len(df_b2000)-1} of b2000'
        }
        logger.info(f"Saved embeddings_b2000_part2.npy: {file_size_part2:.1f} MB")

    # Process other periods - these maintain order naturally since each CSV is one decade
    periods = [
        ('2000s', 2000, 2010),
        ('2010s', 2010, 2015),
        ('2015s', 2015, 2020),
        ('2020s', 2020, 3000)  # Large upper bound
    ]

    for period_name, year_start, year_end in periods:
        df_period = df_all[(df_all.year >= year_start) & (df_all.year < year_end)]

        if len(df_period) > 0:
            logger.info(f"\nGenerating embeddings for {period_name} ({len(df_period)} papers)...")
            embeddings = generator.create_embeddings(df_period, batch_size=BATCH_SIZE)
            embeddings_float16 = embeddings.astype(np.float16)
            output_path = f'../Embeddings/embeddings_{period_name}.npy'
            np.save(output_path, embeddings_float16)

            file_size = os.path.getsize(output_path) / (1024 * 1024)
            all_metadata['files'][period_name] = {
                'num_papers': len(df_period),
                'file_size_mb': file_size,
                'year_range': f'{df_period.year.min()}-{df_period.year.max()}'
            }
            logger.info(f"Saved {output_path}: {file_size:.1f} MB")

    # Calculate total size
    all_metadata['total_size_mb'] = sum(f['file_size_mb'] for f in all_metadata['files'].values())

    # Save overall metadata
    with open('../Embeddings/overall_metadata.json', 'w') as f:
        json.dump(all_metadata, f, indent=2)

    # Print summary
    logger.info(f"\n{'='*60}")
    logger.info("EMBEDDING GENERATION COMPLETE!")
    logger.info(f"{'='*60}")
    logger.info(f"Total papers processed: {all_metadata['total_papers']:,}")
    logger.info(f"Total size: {all_metadata['total_size_mb']:.1f} MB")
    logger.info(f"Model used: {MODEL_NAME}")
    logger.info(f"Output format: NPY (float16)")
    logger.info(f"Embedding files: b2000_part1, b2000_part2, 2000s, 2010s, 2015s, 2020s")
    logger.info(f"{'='*60}")

    # Check file sizes
    logger.info("\nFile sizes:")
    for period, meta in all_metadata['files'].items():
        size_mb = meta['file_size_mb']
        status = "✅" if size_mb < 50 else "⚠️"
        logger.info(f"  {status} {period}: {size_mb:.1f} MB")


if __name__ == "__main__":
    main()
