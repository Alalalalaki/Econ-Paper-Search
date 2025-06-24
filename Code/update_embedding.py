"""
Update embeddings for modified data files only.

This script checks which CSV files have been modified and only regenerates
embeddings for those files, making monthly updates efficient.

Special handling for b2000.csv:
- If b2000.csv is modified, both embeddings_b2000_part1.npy and embeddings_b2000_part2.npy are regenerated
- Split is done by index (50/50) not by year
"""

import os
import sys
import json
from datetime import datetime
import logging
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# Add parent directory to path to import data_processing
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_processing import load_single_file, clean_papers

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)


class EmbeddingUpdater:
    """Update embeddings for modified papers"""

    def __init__(self, model_name=None):
        """Initialize the updater, loading model name from metadata if not provided"""
        # Load model name from existing metadata
        if model_name is None:
            metadata_path = '../Embeddings/overall_metadata.json'
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                model_name = metadata.get('model_name', 'sentence-transformers/all-MiniLM-L6-v2')
            else:
                model_name = 'sentence-transformers/all-MiniLM-L6-v2'

        logger.info(f"Loading model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name
        self.embedding_dim = self.model.get_sentence_embedding_dimension()

    def load_and_clean_papers(self, file_path):
        """Load and clean papers using shared data processing logic"""
        logger.info(f"Loading papers from {file_path}")

        # Use shared loading function
        df = load_single_file(file_path)

        # Use shared cleaning function
        df = clean_papers(df)

        # Reset index after cleaning
        df = df.reset_index(drop=True)

        logger.info(f"Loaded and cleaned {len(df)} papers")
        return df

    def create_embeddings(self, df, batch_size=32):
        """Create embeddings for papers"""
        texts = (df['title'] + ' ' + df['abstract'].fillna('')).tolist()
        logger.info(f"Creating embeddings for {len(texts)} papers...")

        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        return embeddings


def get_file_modification_time(filepath):
    """Get file modification time as ISO string"""
    if os.path.exists(filepath):
        return datetime.fromtimestamp(os.path.getmtime(filepath)).isoformat()
    return None


def check_which_files_need_update():
    """
    Check which CSV files have been modified since last embedding generation.
    Returns a list of CSV periods that need updating.
    """

    # Load existing metadata
    metadata_path = '../Embeddings/overall_metadata.json'
    if not os.path.exists(metadata_path):
        logger.error("No embeddings found. Please run generate_embeddings.py first.")
        return []

    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    files_to_update = []

    # CSV file periods (what actually exists in Data/)
    csv_periods = ['b2000', '2000s', '2010s', '2015s', '2020s']

    logger.info("Checking for modified files...")
    logger.info("="*60)

    for csv_period in csv_periods:
        csv_path = f'../Data/papers_{csv_period}.csv'

        if not os.path.exists(csv_path):
            logger.warning(f"CSV file not found: {csv_path}")
            continue

        # Get CSV modification time
        csv_mod_time = datetime.fromisoformat(get_file_modification_time(csv_path))

        # For b2000, we need to check both part1 and part2 embeddings
        if csv_period == 'b2000':
            # Check if either part needs update
            needs_update = False

            for part in ['b2000_part1', 'b2000_part2']:
                if part in metadata['files']:
                    emb_creation_time = datetime.fromisoformat(
                        metadata['files'][part].get('creation_date', metadata['creation_date'])
                    )
                    if csv_mod_time > emb_creation_time:
                        needs_update = True
                        break
                else:
                    needs_update = True
                    break

            if needs_update:
                logger.info(f"✓ {csv_period}.csv has been modified - will update part1 + part2 embeddings")
                files_to_update.append(csv_period)
            else:
                logger.info(f"  {csv_period}.csv is up to date")

        else:
            # Standard check for other files
            if csv_period in metadata['files']:
                emb_creation_time = datetime.fromisoformat(
                    metadata['files'][csv_period].get('creation_date', metadata['creation_date'])
                )
                if csv_mod_time > emb_creation_time:
                    logger.info(f"✓ {csv_period}.csv has been modified - will update embeddings")
                    files_to_update.append(csv_period)
                else:
                    logger.info(f"  {csv_period}.csv is up to date")
            else:
                logger.info(f"✓ {csv_period}.csv has no embeddings yet - will create")
                files_to_update.append(csv_period)

    logger.info("="*60)

    return files_to_update


def update_embeddings(csv_files_to_update):
    """Update embeddings for the specified CSV files"""

    if not csv_files_to_update:
        logger.info("All embeddings are up to date!")
        return

    logger.info(f"\nWill update embeddings for: {', '.join(csv_files_to_update)}")

    # Initialize updater
    updater = EmbeddingUpdater()

    # Load existing metadata
    with open('../Embeddings/overall_metadata.json', 'r') as f:
        metadata = json.load(f)

    # Track what we updated
    updated_embeddings = []

    # Process each CSV file
    for csv_period in csv_files_to_update:
        logger.info(f"\n{'='*60}")
        logger.info(f"Updating {csv_period}...")
        logger.info(f"{'='*60}")

        # Load papers
        csv_path = f'../Data/papers_{csv_period}.csv'
        df = updater.load_and_clean_papers(csv_path)

        if csv_period == 'b2000':
            # Special handling: split by index and create two embedding files
            logger.info("Splitting b2000 by index (50/50)...")

            # Find where to split
            split_index = len(df) // 2

            # Split dataframe
            df_part1 = df.iloc[:split_index].reset_index(drop=True)
            df_part2 = df.iloc[split_index:].reset_index(drop=True)

            logger.info(f"  Part 1: {len(df_part1):,} papers (indices 0-{split_index-1})")
            logger.info(f"  Part 2: {len(df_part2):,} papers (indices {split_index}-{len(df)-1})")

            # Update part1
            logger.info("\nUpdating embeddings_b2000_part1.npy...")
            embeddings_part1 = updater.create_embeddings(df_part1)
            embeddings_part1_float16 = embeddings_part1.astype(np.float16)
            np.save('../Embeddings/embeddings_b2000_part1.npy', embeddings_part1_float16)

            file_size_part1 = os.path.getsize('../Embeddings/embeddings_b2000_part1.npy') / (1024 * 1024)
            metadata['files']['b2000_part1'] = {
                'num_papers': len(df_part1),
                'file_size_mb': file_size_part1,
                'year_range': f'{df_part1.year.min()}-{df_part1.year.max()}',
                'index_range': f'0-{split_index-1} of b2000',
                'source_csv': 'papers_b2000.csv',
                'creation_date': datetime.now().isoformat()
            }
            updated_embeddings.append('b2000_part1')
            logger.info(f"✓ Updated embeddings_b2000_part1.npy: {file_size_part1:.1f} MB")

            # Update part2
            logger.info("\nUpdating embeddings_b2000_part2.npy...")
            embeddings_part2 = updater.create_embeddings(df_part2)
            embeddings_part2_float16 = embeddings_part2.astype(np.float16)
            np.save('../Embeddings/embeddings_b2000_part2.npy', embeddings_part2_float16)

            file_size_part2 = os.path.getsize('../Embeddings/embeddings_b2000_part2.npy') / (1024 * 1024)
            metadata['files']['b2000_part2'] = {
                'num_papers': len(df_part2),
                'file_size_mb': file_size_part2,
                'year_range': f'{df_part2.year.min()}-{df_part2.year.max()}',
                'index_range': f'{split_index}-{len(df)-1} of b2000',
                'source_csv': 'papers_b2000.csv',
                'creation_date': datetime.now().isoformat()
            }
            updated_embeddings.append('b2000_part2')
            logger.info(f"✓ Updated embeddings_b2000_part2.npy: {file_size_part2:.1f} MB")

        else:
            # Standard update for other periods
            embeddings = updater.create_embeddings(df)
            embeddings_float16 = embeddings.astype(np.float16)
            output_path = f'../Embeddings/embeddings_{csv_period}.npy'
            np.save(output_path, embeddings_float16)

            file_size = os.path.getsize(output_path) / (1024 * 1024)
            metadata['files'][csv_period] = {
                'num_papers': len(df),
                'file_size_mb': file_size,
                'year_range': f'{df.year.min()}-{df.year.max()}',
                'source_csv': f'papers_{csv_period}.csv',
                'creation_date': datetime.now().isoformat()
            }
            updated_embeddings.append(csv_period)
            logger.info(f"✓ Updated {output_path}: {file_size:.1f} MB")

    # Update totals
    metadata['total_papers'] = sum(f['num_papers'] for f in metadata['files'].values())
    metadata['total_size_mb'] = sum(f['file_size_mb'] for f in metadata['files'].values())
    metadata['last_update'] = datetime.now().isoformat()

    # Save updated metadata
    with open('../Embeddings/overall_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    # Summary
    logger.info(f"\n{'='*60}")
    logger.info("UPDATE COMPLETE!")
    logger.info(f"{'='*60}")
    logger.info(f"Updated embeddings: {', '.join(updated_embeddings)}")
    logger.info(f"Total papers: {metadata['total_papers']:,}")
    logger.info(f"Total size: {metadata['total_size_mb']:.1f} MB")
    logger.info(f"{'='*60}")


def main():
    """Main function"""

    # Check if we're in the right directory
    if not os.path.exists('../Data') or not os.path.exists('../Embeddings'):
        logger.error("Error: Cannot find Data or Embeddings directories.")
        logger.error("Please run this script from the Code directory.")
        logger.error("Make sure you've run generate_embeddings.py first.")
        return

    logger.info("="*60)
    logger.info("EMBEDDING UPDATE CHECK")
    logger.info("="*60)

    # Check which files need updating
    files_to_update = check_which_files_need_update()

    if files_to_update:
        logger.info(f"\nFound {len(files_to_update)} file(s) to update")
        update_embeddings(files_to_update)
    else:
        logger.info("\n✅ All embeddings are up to date!")

    logger.info("\nTip: This script is perfect for monthly updates after adding new papers.")


if __name__ == "__main__":
    main()
