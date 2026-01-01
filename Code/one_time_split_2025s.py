"""
ONE-TIME SCRIPT: Update Econ-Paper-Search for the 2025s data split

Purpose:
    This script performs a one-time update after the CSV split in Econ-Paper-Scrape.
    It copies the new data files and regenerates embeddings for 2020s and 2025s.

Prerequisites:
    1. Run one_time_split_2025s.py in Econ-Paper-Scrape first
    2. Ensure the split was successful (papers_2020s.csv has 2020-2024, papers_2025s.csv has 2025+)

Usage:
    cd Code/
    python one_time_split_2025s.py

Created: 2026-01-01
Note: This script should only be run ONCE after the corresponding Econ-Paper-Scrape script.
"""

import os
import sys
import json
import shutil
import numpy as np
import pandas as pd
from datetime import datetime
from sentence_transformers import SentenceTransformer

# Add parent directory for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_processing import load_all_papers


# Configuration
SCRAPE_DATA_PATH = "../../Econ-Paper-Scrape/Data/"
SEARCH_DATA_PATH = "../Data/"
EMBEDDINGS_PATH = "../Embeddings/"
BACKUP_SUFFIX = f"_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


def step1_verify_scrape_split():
    """Verify that Econ-Paper-Scrape has the split data files."""
    print("Step 1: Verifying Econ-Paper-Scrape data split...")

    # Check papers_2020s.csv exists and has correct year range
    file_2020s = SCRAPE_DATA_PATH + "papers_2020s.csv"
    if not os.path.exists(file_2020s):
        print(f"  ERROR: {file_2020s} not found!")
        return False

    df_2020s = pd.read_csv(file_2020s, dtype={"year": "Int16"})
    if df_2020s.year.max() > 2024:
        print(f"  ERROR: papers_2020s.csv contains years > 2024 (max: {df_2020s.year.max()})")
        print("  Please run one_time_split_2025s.py in Econ-Paper-Scrape first!")
        return False

    # Check papers_2025s.csv exists
    file_2025s = SCRAPE_DATA_PATH + "papers_2025s.csv"
    if not os.path.exists(file_2025s):
        print(f"  ERROR: {file_2025s} not found!")
        print("  Please run one_time_split_2025s.py in Econ-Paper-Scrape first!")
        return False

    df_2025s = pd.read_csv(file_2025s, dtype={"year": "Int16"})
    if df_2025s.year.min() < 2025:
        print(f"  ERROR: papers_2025s.csv contains years < 2025 (min: {df_2025s.year.min()})")
        return False

    print(f"  papers_2020s.csv: {len(df_2020s)} papers (years {df_2020s.year.min()}-{df_2020s.year.max()})")
    print(f"  papers_2025s.csv: {len(df_2025s)} papers (years {df_2025s.year.min()}-{df_2025s.year.max()})")
    print("  PASS: Econ-Paper-Scrape data split verified")
    return True


def step2_backup_current_data():
    """Backup current data files."""
    print("\nStep 2: Backing up current data...")

    backup_dir = SEARCH_DATA_PATH + f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}/"
    os.makedirs(backup_dir, exist_ok=True)

    # Backup current papers_2020s.csv
    src = SEARCH_DATA_PATH + "papers_2020s.csv"
    if os.path.exists(src):
        shutil.copy(src, backup_dir + "papers_2020s.csv")
        print(f"  Backed up papers_2020s.csv")

    # Backup current embeddings_2020s.npy
    src = EMBEDDINGS_PATH + "embeddings_2020s.npy"
    if os.path.exists(src):
        shutil.copy(src, backup_dir + "embeddings_2020s.npy")
        print(f"  Backed up embeddings_2020s.npy")

    # Backup metadata
    src = EMBEDDINGS_PATH + "overall_metadata.json"
    if os.path.exists(src):
        shutil.copy(src, backup_dir + "overall_metadata.json")
        print(f"  Backed up overall_metadata.json")

    print(f"  Backup directory: {backup_dir}")
    return backup_dir


def step3_copy_new_data():
    """Copy the split data files from Econ-Paper-Scrape."""
    print("\nStep 3: Copying new data files...")

    # Copy papers_2020s.csv (now 2020-2024 only)
    src = SCRAPE_DATA_PATH + "papers_2020s.csv"
    dst = SEARCH_DATA_PATH + "papers_2020s.csv"
    shutil.copy(src, dst)
    print(f"  Copied papers_2020s.csv")

    # Copy papers_2025s.csv (new file)
    src = SCRAPE_DATA_PATH + "papers_2025s.csv"
    dst = SEARCH_DATA_PATH + "papers_2025s.csv"
    shutil.copy(src, dst)
    print(f"  Copied papers_2025s.csv")

    return True


def step4_regenerate_embeddings():
    """Regenerate embeddings for 2020s and 2025s."""
    print("\nStep 4: Regenerating embeddings for 2020s and 2025s...")

    # Load metadata
    metadata_path = EMBEDDINGS_PATH + "overall_metadata.json"
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    model_name = metadata.get('model_name', 'sentence-transformers/all-MiniLM-L6-v2')
    print(f"  Loading model: {model_name}")
    model = SentenceTransformer(model_name)

    # Load all papers
    print("  Loading all papers...")
    df_all = load_all_papers(data_dir="../Data")
    print(f"  Total papers after cleaning: {len(df_all)}")

    # Generate embeddings for 2020s (2020-2024)
    print("\n  Generating embeddings for 2020s (2020-2024)...")
    df_2020s = df_all[(df_all.year >= 2020) & (df_all.year < 2025)].copy()
    print(f"  Papers in 2020s: {len(df_2020s)}")

    texts_2020s = (df_2020s['title'] + ' ' + df_2020s['abstract'].fillna('')).tolist()
    embeddings_2020s = model.encode(texts_2020s, batch_size=32, show_progress_bar=True,
                                     convert_to_numpy=True, normalize_embeddings=True)

    embeddings_2020s_path = EMBEDDINGS_PATH + "embeddings_2020s.npy"
    np.save(embeddings_2020s_path, embeddings_2020s.astype(np.float16))
    file_size_2020s = os.path.getsize(embeddings_2020s_path) / (1024 * 1024)
    print(f"  Saved embeddings_2020s.npy: {len(embeddings_2020s)} embeddings, {file_size_2020s:.1f} MB")

    # Update metadata for 2020s
    metadata['files']['2020s'] = {
        'num_papers': len(df_2020s),
        'file_size_mb': file_size_2020s,
        'year_range': f'{df_2020s.year.min()}-{df_2020s.year.max()}',
        'source_csv': 'papers_2020s.csv',
        'creation_date': datetime.now().isoformat()
    }

    # Generate embeddings for 2025s (2025+)
    print("\n  Generating embeddings for 2025s (2025+)...")
    df_2025s = df_all[df_all.year >= 2025].copy()
    print(f"  Papers in 2025s: {len(df_2025s)}")

    texts_2025s = (df_2025s['title'] + ' ' + df_2025s['abstract'].fillna('')).tolist()
    embeddings_2025s = model.encode(texts_2025s, batch_size=32, show_progress_bar=True,
                                     convert_to_numpy=True, normalize_embeddings=True)

    embeddings_2025s_path = EMBEDDINGS_PATH + "embeddings_2025s.npy"
    np.save(embeddings_2025s_path, embeddings_2025s.astype(np.float16))
    file_size_2025s = os.path.getsize(embeddings_2025s_path) / (1024 * 1024)
    print(f"  Saved embeddings_2025s.npy: {len(embeddings_2025s)} embeddings, {file_size_2025s:.1f} MB")

    # Update metadata for 2025s
    metadata['files']['2025s'] = {
        'num_papers': len(df_2025s),
        'file_size_mb': file_size_2025s,
        'year_range': f'{df_2025s.year.min()}-{df_2025s.year.max()}',
        'source_csv': 'papers_2025s.csv',
        'creation_date': datetime.now().isoformat()
    }

    # Update embedding structure
    if '2025s' not in metadata['embedding_structure']:
        metadata['embedding_structure'].append('2025s')

    # Update totals
    metadata['total_papers'] = sum(f['num_papers'] for f in metadata['files'].values())
    metadata['total_size_mb'] = sum(f['file_size_mb'] for f in metadata['files'].values())
    metadata['last_update'] = datetime.now().isoformat()

    # Save metadata
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"\n  Updated metadata: total {metadata['total_papers']} papers")

    return True


def step5_validate():
    """Comprehensive validation of embedding-paper matching."""
    print("\nStep 5: Validating embedding-paper matching...")

    # Load all papers
    df_all = load_all_papers(data_dir="../Data")

    # Load all embeddings
    all_embeddings = []
    for period in ['b2000_part1', 'b2000_part2', '2000s', '2010s', '2015s', '2020s', '2025s']:
        path = EMBEDDINGS_PATH + f"embeddings_{period}.npy"
        if os.path.exists(path):
            emb = np.load(path).astype(np.float32)
            all_embeddings.append(emb)
            print(f"  Loaded {period}: {len(emb)} embeddings")

    embeddings = np.vstack(all_embeddings)

    # Check 1: Total count match
    print(f"\n  Total papers: {len(df_all)}")
    print(f"  Total embeddings: {len(embeddings)}")
    count_match = len(df_all) == len(embeddings)
    print(f"  Count match: {'PASS' if count_match else 'FAIL'}")

    if not count_match:
        print("  ERROR: Count mismatch! Aborting validation.")
        return False

    # Check 2: Per-period count match
    print("\n  Per-period validation:")
    with open(EMBEDDINGS_PATH + "overall_metadata.json", 'r') as f:
        metadata = json.load(f)

    period_checks = []
    year_ranges = {
        'b2000': (None, 2000),
        '2000s': (2000, 2010),
        '2010s': (2010, 2015),
        '2015s': (2015, 2020),
        '2020s': (2020, 2025),
        '2025s': (2025, None)
    }

    for period, (year_start, year_end) in year_ranges.items():
        if year_start is None:
            df_period = df_all[df_all.year < year_end]
        elif year_end is None:
            df_period = df_all[df_all.year >= year_start]
        else:
            df_period = df_all[(df_all.year >= year_start) & (df_all.year < year_end)]

        if period == 'b2000':
            emb_count = metadata['files'].get('b2000_part1', {}).get('num_papers', 0) + \
                       metadata['files'].get('b2000_part2', {}).get('num_papers', 0)
        else:
            emb_count = metadata['files'].get(period, {}).get('num_papers', 0)

        match = len(df_period) == emb_count
        period_checks.append(match)
        status = 'PASS' if match else 'FAIL'
        print(f"    {period}: papers={len(df_period)}, embeddings={emb_count} - {status}")

    # Check 3: Random sample embedding verification
    print("\n  Random sample verification (re-encode and check similarity):")
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

    np.random.seed(42)
    sample_indices = np.random.choice(len(df_all), 10, replace=False)

    similarities = []
    for idx in sample_indices:
        paper = df_all.iloc[idx]
        text = paper['title'] + ' ' + (paper['abstract'] if pd.notna(paper['abstract']) else '')
        new_emb = model.encode(text, normalize_embeddings=True)
        stored_emb = embeddings[idx]
        similarity = np.dot(new_emb, stored_emb)
        similarities.append(similarity)

    avg_similarity = np.mean(similarities)
    min_similarity = np.min(similarities)
    print(f"    Average similarity: {avg_similarity:.6f}")
    print(f"    Min similarity: {min_similarity:.6f}")
    similarity_pass = min_similarity > 0.999
    print(f"    Similarity check: {'PASS' if similarity_pass else 'FAIL'}")

    # Overall result
    all_passed = count_match and all(period_checks) and similarity_pass

    print("\n" + "=" * 70)
    if all_passed:
        print("VALIDATION PASSED: All checks successful!")
    else:
        print("VALIDATION FAILED: Some checks failed. Please review the output above.")
    print("=" * 70)

    return all_passed


def main():
    print("=" * 70)
    print("ONE-TIME SCRIPT: Update Econ-Paper-Search for 2025s data split")
    print("=" * 70)
    print(f"Timestamp: {datetime.now().isoformat()}")
    print()

    # Step 1: Verify Econ-Paper-Scrape has the split data
    if not step1_verify_scrape_split():
        print("\nAborting: Please run one_time_split_2025s.py in Econ-Paper-Scrape first!")
        return False

    # Step 2: Backup current data
    backup_dir = step2_backup_current_data()

    # Step 3: Copy new data files
    if not step3_copy_new_data():
        print("\nAborting: Failed to copy data files!")
        return False

    # Step 4: Regenerate embeddings
    if not step4_regenerate_embeddings():
        print("\nAborting: Failed to regenerate embeddings!")
        return False

    # Step 5: Validate
    if not step5_validate():
        print(f"\nValidation failed! Backup available at: {backup_dir}")
        return False

    print("\n" + "=" * 70)
    print("SUCCESS: All steps completed successfully!")
    print()
    print("Summary:")
    print("  - Data files copied and split verified")
    print("  - Embeddings regenerated for 2020s and 2025s")
    print("  - All validation checks passed")
    print()
    print("Next steps:")
    print("  1. Review the changes")
    print("  2. Run test_embedding_consistency.py for additional verification")
    print("  3. Commit and push changes to both repositories")
    print("=" * 70)

    return True


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
