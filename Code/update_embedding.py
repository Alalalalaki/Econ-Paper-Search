"""
Update embeddings for modified data files.

For each modified CSV, the script generates embeddings only for newly added papers
rather than regenerating all embeddings. This works because update.py prepends new
papers to the beginning of each CSV file, so new papers appear at the start of each
period's subset after deduplication.
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
    csv_periods = ['b2000', '2000s', '2010s', '2015s', '2020s', '2025s']

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


YEAR_RANGES = {
    'b2000': (None, 2000),
    '2000s': (2000, 2010),
    '2010s': (2010, 2015),
    '2015s': (2015, 2020),
    '2020s': (2020, 2025),
    '2025s': (2025, None),
}


def filter_period(df_all, period_name):
    """Filter the full dataframe to a specific year-range period."""
    year_start, year_end = YEAR_RANGES[period_name]
    if year_start is None:
        return df_all[df_all.year < year_end]
    elif year_end is None:
        return df_all[df_all.year >= year_start]
    else:
        return df_all[(df_all.year >= year_start) & (df_all.year < year_end)]


def efficient_update_period(updater, metadata, df_all, period_name):
    """
    Efficiently update embeddings for a single period by generating embeddings
    only for newly added papers. Works for any period, not just 2025s.

    New papers are at the beginning of each period's subset because update.py
    prepends them to the CSV files.

    For b2000 (split into two .npy files): generates new embeddings, combines
    with existing part1+part2, and re-splits at the new midpoint — no existing
    embeddings are regenerated, just reshuffled.

    Returns:
        list: embedding file names that were updated, or empty list if nothing changed
    """
    df_period = filter_period(df_all, period_name)
    current_count = len(df_period)

    if current_count == 0:
        logger.warning(f"No papers found for {period_name}")
        return []

    # Get previous count from metadata
    if period_name == 'b2000':
        previous_count = (metadata['files'].get('b2000_part1', {}).get('num_papers', 0)
                          + metadata['files'].get('b2000_part2', {}).get('num_papers', 0))
    else:
        previous_count = metadata['files'].get(period_name, {}).get('num_papers', 0)

    if current_count == previous_count:
        logger.info(f"  {period_name}: no new papers, skipping")
        return []

    new_paper_count = current_count - previous_count
    logger.info(f"\n{period_name}: {new_paper_count} new papers ({previous_count} → {current_count})")

    # New papers are at the beginning due to prepend convention
    df_new = df_period.iloc[:new_paper_count]
    new_embeddings = updater.create_embeddings(df_new)

    now = datetime.now().isoformat()

    if period_name == 'b2000':
        # Load existing part1 and part2, combine with new, re-split
        existing_part1 = np.load('../Embeddings/embeddings_b2000_part1.npy').astype(np.float32)
        existing_part2 = np.load('../Embeddings/embeddings_b2000_part2.npy').astype(np.float32)
        all_embeddings = np.vstack([new_embeddings, existing_part1, existing_part2])

        split_index = len(all_embeddings) // 2
        np.save('../Embeddings/embeddings_b2000_part1.npy', all_embeddings[:split_index].astype(np.float16))
        np.save('../Embeddings/embeddings_b2000_part2.npy', all_embeddings[split_index:].astype(np.float16))

        for part, count, idx_start in [('b2000_part1', split_index, 0),
                                        ('b2000_part2', len(all_embeddings) - split_index, split_index)]:
            file_size = os.path.getsize(f'../Embeddings/embeddings_{part}.npy') / (1024 * 1024)
            metadata['files'][part] = {
                'num_papers': count,
                'file_size_mb': file_size,
                'year_range': f'{df_period.year.min()}-{df_period.year.max()}',
                'index_range': f'{idx_start}-{idx_start + count - 1} of b2000',
                'source_csv': 'papers_b2000.csv',
                'creation_date': now,
                'papers_added': new_paper_count,
            }
            logger.info(f"  ✓ embeddings_{part}.npy: {count} papers, {file_size:.1f} MB")

        return ['b2000_part1', 'b2000_part2']

    else:
        # Single-file period: stack [new, existing] and save
        embeddings_path = f'../Embeddings/embeddings_{period_name}.npy'
        if os.path.exists(embeddings_path) and previous_count > 0:
            existing = np.load(embeddings_path).astype(np.float32)
            all_embeddings = np.vstack([new_embeddings, existing])
        else:
            all_embeddings = new_embeddings

        np.save(embeddings_path, all_embeddings.astype(np.float16))

        file_size = os.path.getsize(embeddings_path) / (1024 * 1024)
        metadata['files'][period_name] = {
            'num_papers': current_count,
            'file_size_mb': file_size,
            'year_range': f'{df_period.year.min()}-{df_period.year.max()}',
            'source_csv': f'papers_{period_name}.csv',
            'creation_date': now,
            'papers_added': new_paper_count,
        }
        logger.info(f"  ✓ embeddings_{period_name}.npy: {current_count} papers, {file_size:.1f} MB")

        return [period_name]


def update_embeddings(csv_files_to_update):
    """Update embeddings for the specified CSV files using efficient incremental updates."""

    if not csv_files_to_update:
        logger.info("All embeddings are up to date!")
        return

    logger.info(f"\nWill update embeddings for: {', '.join(csv_files_to_update)}")

    updater = EmbeddingUpdater()

    with open('../Embeddings/overall_metadata.json', 'r') as f:
        metadata = json.load(f)

    # Load all papers once for consistent cross-file deduplication
    df_all = load_all_papers(data_dir="../Data")
    logger.info(f"Total papers after deduplication: {len(df_all)}")

    # Efficient update for each changed period
    updated_embeddings = []
    for period in csv_files_to_update:
        updated_embeddings.extend(efficient_update_period(updater, metadata, df_all, period))

    # Update totals
    metadata['total_papers'] = sum(f['num_papers'] for f in metadata['files'].values())
    metadata['total_size_mb'] = sum(f['file_size_mb'] for f in metadata['files'].values())
    metadata['last_update'] = datetime.now().isoformat()

    with open('../Embeddings/overall_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    # Summary
    logger.info(f"\n{'='*60}")
    logger.info("UPDATE COMPLETE!")
    logger.info(f"{'='*60}")
    logger.info(f"Updated embeddings: {', '.join(updated_embeddings)}")
    logger.info(f"Total papers: {metadata['total_papers']:,}")
    logger.info(f"Total size: {metadata['total_size_mb']:.1f} MB")
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
