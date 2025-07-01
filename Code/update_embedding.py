"""
Update embeddings for modified data files.

For efficiency, when only papers_2020s.csv is modified (the common case for monthly updates),
the script generates embeddings only for newly added papers rather than regenerating all embeddings.
This works because update.py prepends new papers to the beginning of papers_2020s.csv.

For other files or multiple file changes, the script performs full regeneration to ensure
consistency with cross-file duplicate removal.
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

from data_processing import load_all_papers

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


def efficient_update_2020s(updater, metadata):
    """
    Update embeddings for 2020s papers by processing only newly added entries.

    This function leverages the fact that update.py prepends new papers to the
    beginning of papers_2020s.csv. It loads existing embeddings and only generates
    new ones for the papers that were added, then concatenates them in the correct order.

    Args:
        updater: EmbeddingUpdater instance
        metadata: Dictionary containing embedding metadata

    Returns:
        bool: True if update was performed, False if no new papers found
    """
    logger.info("\nPerforming efficient update for 2020s papers...")

    # Load all papers to ensure consistent duplicate removal across files
    df_all = load_all_papers(data_dir="../Data")

    # Extract papers for the specific year range
    df_2020s = df_all[df_all.year >= 2020].copy()
    current_count = len(df_2020s)

    # Get previous count from metadata
    previous_count = metadata['files'].get('2020s', {}).get('num_papers', 0)

    if current_count == previous_count:
        logger.info("No new papers in 2020s. Skipping update.")
        return False

    new_paper_count = current_count - previous_count
    logger.info(f"Found {new_paper_count} new papers in 2020s")

    # Load existing embeddings
    embeddings_path = '../Embeddings/embeddings_2020s.npy'
    if os.path.exists(embeddings_path) and previous_count > 0:
        logger.info("Loading existing embeddings...")
        existing_embeddings = np.load(embeddings_path).astype(np.float32)

        # New papers are at the beginning of the dataframe due to update.py's prepend logic
        df_new_papers = df_2020s.iloc[:new_paper_count]

        # Generate embeddings only for new papers
        logger.info(f"Generating embeddings for {new_paper_count} new papers...")
        new_embeddings = updater.create_embeddings(df_new_papers)

        # Maintain correct order: new embeddings first, then existing
        all_embeddings = np.vstack([new_embeddings, existing_embeddings])

        logger.info(f"Combined embeddings shape: {all_embeddings.shape}")
    else:
        # All papers are new - generate embeddings for entire dataset
        logger.info("No existing embeddings found. Generating all embeddings...")
        all_embeddings = updater.create_embeddings(df_2020s)

    # Save updated embeddings
    embeddings_float16 = all_embeddings.astype(np.float16)
    np.save(embeddings_path, embeddings_float16)

    # Update metadata with additional tracking for efficient updates
    file_size = os.path.getsize(embeddings_path) / (1024 * 1024)
    metadata['files']['2020s'] = {
        'num_papers': current_count,
        'file_size_mb': file_size,
        'year_range': f'{df_2020s.year.min()}-{df_2020s.year.max()}',
        'source_csv': 'papers_2020s.csv',
        'creation_date': datetime.now().isoformat(),
        'last_efficient_update': datetime.now().isoformat(),  # Track when efficient update was used
        'papers_added': new_paper_count  # Track incremental additions
    }

    logger.info(f"✓ Updated embeddings_2020s.npy: {file_size:.1f} MB")
    logger.info(f"  Previous papers: {previous_count}")
    logger.info(f"  Current papers: {current_count}")
    logger.info(f"  New papers added: {new_paper_count}")

    return True


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

    # Determine if only 2020s needs updating (common case for monthly updates)
    if csv_files_to_update == ['2020s']:
        # Use efficient update path for 2020s
        if efficient_update_2020s(updater, metadata):
            updated_embeddings.append('2020s')
    else:
        # Full regeneration required when multiple files changed or non-2020s files modified
        logger.info("\nPerforming full regeneration (non-2020s files were modified)...")

        # Load ALL papers together to handle cross-file duplicate removal properly
        # This ensures consistency with how the app loads data
        df_all = load_all_papers(data_dir="../Data")
        logger.info(f"Total papers after duplicate removal: {len(df_all)}")

        # Process each CSV file that needs updating
        for csv_period in csv_files_to_update:
            logger.info(f"\n{'='*60}")
            logger.info(f"Updating {csv_period}...")
            logger.info(f"{'='*60}")

            if csv_period == 'b2000':
                # Filter to b2000 papers from the full dataset
                df_b2000 = df_all[df_all.year < 2000]

                if len(df_b2000) == 0:
                    logger.warning(f"No papers found for {csv_period}")
                    continue

                # Process papers maintaining the same split index as original generation
                split_index = len(df_b2000) // 2

                # Part 1
                df_part1 = df_b2000.iloc[:split_index]
                logger.info(f"\nUpdating embeddings_b2000_part1.npy ({len(df_part1)} papers)...")
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

                # Part 2
                df_part2 = df_b2000.iloc[split_index:]
                logger.info(f"\nUpdating embeddings_b2000_part2.npy ({len(df_part2)} papers)...")
                embeddings_part2 = updater.create_embeddings(df_part2)
                embeddings_part2_float16 = embeddings_part2.astype(np.float16)
                np.save('../Embeddings/embeddings_b2000_part2.npy', embeddings_part2_float16)

                file_size_part2 = os.path.getsize('../Embeddings/embeddings_b2000_part2.npy') / (1024 * 1024)
                metadata['files']['b2000_part2'] = {
                    'num_papers': len(df_part2),
                    'file_size_mb': file_size_part2,
                    'year_range': f'{df_part2.year.min()}-{df_part2.year.max()}',
                    'index_range': f'{split_index}-{len(df_b2000)-1} of b2000',
                    'source_csv': 'papers_b2000.csv',
                    'creation_date': datetime.now().isoformat()
                }
                updated_embeddings.append('b2000_part2')
                logger.info(f"✓ Updated embeddings_b2000_part2.npy: {file_size_part2:.1f} MB")

            else:
                # Filter papers for this period from the full dataset
                year_ranges = {
                    '2000s': (2000, 2010),
                    '2010s': (2010, 2015),
                    '2015s': (2015, 2020),
                    '2020s': (2020, 3000)
                }

                if csv_period in year_ranges:
                    year_start, year_end = year_ranges[csv_period]
                    df_period = df_all[(df_all.year >= year_start) & (df_all.year < year_end)]

                    if len(df_period) == 0:
                        logger.warning(f"No papers found for {csv_period}")
                        continue

                    # The efficient path for 2020s applies even in full regeneration mode
                    if csv_period == '2020s':
                        # Still use efficient method if possible
                        if efficient_update_2020s(updater, metadata):
                            updated_embeddings.append('2020s')
                        continue

                    # Create embeddings for other periods
                    embeddings = updater.create_embeddings(df_period)
                    embeddings_float16 = embeddings.astype(np.float16)
                    output_path = f'../Embeddings/embeddings_{csv_period}.npy'
                    np.save(output_path, embeddings_float16)

                    file_size = os.path.getsize(output_path) / (1024 * 1024)
                    metadata['files'][csv_period] = {
                        'num_papers': len(df_period),
                        'file_size_mb': file_size,
                        'year_range': f'{df_period.year.min()}-{df_period.year.max()}',
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

    # Verify total count consistency between embeddings and papers
    df_all = load_all_papers(data_dir="../Data")
    logger.info(f"Embeddings match app data: {metadata['total_papers'] == len(df_all)} ✓")
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

if __name__ == "__main__":
    main()
