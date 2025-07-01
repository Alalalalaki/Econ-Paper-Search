"""
Test script to verify embedding consistency after updates.

This script performs the same data loading operations as app.py to ensure
that the number and order of papers matches the embeddings exactly.
Run this after any embedding update to prevent data mismatch errors.
"""

import os
import sys
import numpy as np
import pandas as pd
import json
from datetime import datetime
from data_processing import load_all_papers

# Add parent directory to path to import from app.py
sys.path.append('..')


def load_embeddings_like_app():
    """
    Load embeddings exactly as app.py does.
    This ensures we're testing the actual loading logic.
    """
    all_embeddings = []

    for period in ['b2000_part1', 'b2000_part2', '2000s', '2010s', '2015s', '2020s']:
        path = f'../Embeddings/embeddings_{period}.npy'
        if os.path.exists(path):
            embeddings = np.load(path).astype(np.float32)
            all_embeddings.append(embeddings)
        else:
            print(f"WARNING: Missing embedding file: {path}")

    # Concatenate all embeddings in order
    if all_embeddings:
        return np.vstack(all_embeddings)
    else:
        return None


def verify_embedding_files():
    """Verify that all expected embedding files exist."""
    expected_files = [
        'embeddings_b2000_part1.npy',
        'embeddings_b2000_part2.npy',
        'embeddings_2000s.npy',
        'embeddings_2010s.npy',
        'embeddings_2015s.npy',
        'embeddings_2020s.npy',
        'overall_metadata.json'
    ]

    missing_files = []
    for file in expected_files:
        path = f'../Embeddings/{file}'
        if not os.path.exists(path):
            missing_files.append(file)

    return missing_files


def check_embedding_consistency():
    """
    Main test function to verify embedding consistency.
    Returns True if all tests pass, False otherwise.
    """
    print("="*60)
    print("EMBEDDING CONSISTENCY TEST")
    print("="*60)
    print(f"Test started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    all_tests_passed = True

    # Test 1: Check if all required files exist
    print("Test 1: Checking for required embedding files...")
    missing_files = verify_embedding_files()
    if missing_files:
        print(f"❌ FAILED: Missing files: {', '.join(missing_files)}")
        all_tests_passed = False
    else:
        print("✅ PASSED: All required embedding files exist")
    print()

    # Test 2: Load papers and embeddings
    print("Test 2: Loading data...")
    try:
        # Load papers exactly as app.py does
        df = load_all_papers(data_dir="../Data")
        print(f"   Papers loaded: {len(df):,}")

        # Load embeddings exactly as app.py does
        embeddings = load_embeddings_like_app()
        if embeddings is None:
            print("❌ FAILED: Could not load embeddings")
            return False
        print(f"   Embeddings loaded: {len(embeddings):,}")
    except Exception as e:
        print(f"❌ FAILED: Error loading data: {e}")
        return False
    print()

    # Test 3: Check if counts match
    print("Test 3: Verifying paper-embedding count match...")
    if len(df) != len(embeddings):
        print(f"❌ FAILED: Count mismatch!")
        print(f"   Papers in database: {len(df):,}")
        print(f"   Embeddings loaded: {len(embeddings):,}")
        print(f"   Difference: {abs(len(df) - len(embeddings)):,}")
        all_tests_passed = False
    else:
        print(f"✅ PASSED: Counts match ({len(df):,} papers)")
    print()

    # Test 4: Verify metadata consistency
    print("Test 4: Checking metadata consistency...")
    try:
        with open('../Embeddings/overall_metadata.json', 'r') as f:
            metadata = json.load(f)

        # Check total papers in metadata
        metadata_total = metadata.get('total_papers', 0)
        if metadata_total != len(df):
            print(f"❌ WARNING: Metadata total ({metadata_total:,}) doesn't match actual ({len(df):,})")
            all_tests_passed = False
        else:
            print(f"✅ PASSED: Metadata total matches ({metadata_total:,})")

        # Check individual file counts
        print("\n   Checking individual period counts:")

        # Count papers by period in the actual data
        actual_counts = {
            'b2000': len(df[df.year < 2000]),
            '2000s': len(df[(df.year >= 2000) & (df.year < 2010)]),
            '2010s': len(df[(df.year >= 2010) & (df.year < 2015)]),
            '2015s': len(df[(df.year >= 2015) & (df.year < 2020)]),
            '2020s': len(df[df.year >= 2020])
        }

        # For b2000, we need to check both parts
        b2000_total = actual_counts['b2000']
        expected_part1 = metadata['files'].get('b2000_part1', {}).get('num_papers', 0)
        expected_part2 = metadata['files'].get('b2000_part2', {}).get('num_papers', 0)

        if expected_part1 + expected_part2 != b2000_total:
            print(f"   ❌ b2000: Expected {b2000_total}, got {expected_part1 + expected_part2} (part1: {expected_part1}, part2: {expected_part2})")
            all_tests_passed = False
        else:
            print(f"   ✅ b2000: {b2000_total} papers (part1: {expected_part1}, part2: {expected_part2})")

        # Check other periods
        for period in ['2000s', '2010s', '2015s', '2020s']:
            expected = metadata['files'].get(period, {}).get('num_papers', 0)
            actual = actual_counts[period]
            if expected != actual:
                print(f"   ❌ {period}: Expected {actual}, got {expected}")
                all_tests_passed = False
            else:
                print(f"   ✅ {period}: {actual} papers")

    except Exception as e:
        print(f"❌ FAILED: Error checking metadata: {e}")
        all_tests_passed = False
    print()

    # Test 5: Sample embedding verification
    print("Test 5: Sample embedding verification...")
    try:
        # Check if embeddings have the expected properties
        sample_size = min(100, len(embeddings))
        sample_embeddings = embeddings[:sample_size]

        # Check dimensionality
        expected_dim = 384  # all-MiniLM-L6-v2 dimension
        if sample_embeddings.shape[1] != expected_dim:
            print(f"❌ WARNING: Unexpected embedding dimension: {sample_embeddings.shape[1]} (expected {expected_dim})")
        else:
            print(f"✅ Embedding dimension: {expected_dim}")

        # Check normalization (embeddings should be normalized for cosine similarity)
        norms = np.linalg.norm(sample_embeddings, axis=1)
        if not np.allclose(norms, 1.0, atol=1e-3):
            print(f"❌ WARNING: Embeddings may not be normalized properly")
            print(f"   Sample norms: min={norms.min():.4f}, max={norms.max():.4f}, mean={norms.mean():.4f}")
        else:
            print(f"✅ Embeddings are properly normalized")

    except Exception as e:
        print(f"❌ WARNING: Could not verify embedding properties: {e}")
    print()

    # Test 6: File modification times
    print("Test 6: Checking file freshness...")
    try:
        csv_times = {}
        embedding_times = {}

        # Get CSV modification times
        for period in ['b2000', '2000s', '2010s', '2015s', '2020s']:
            csv_path = f'../Data/papers_{period}.csv'
            if os.path.exists(csv_path):
                csv_times[period] = os.path.getmtime(csv_path)

        # Get embedding modification times
        for period in ['b2000_part1', 'b2000_part2', '2000s', '2010s', '2015s', '2020s']:
            emb_path = f'../Embeddings/embeddings_{period}.npy'
            if os.path.exists(emb_path):
                embedding_times[period] = os.path.getmtime(emb_path)

        # Check if embeddings are newer than CSVs
        stale_embeddings = []
        for csv_period, csv_time in csv_times.items():
            if csv_period == 'b2000':
                # Check both parts for b2000
                for part in ['b2000_part1', 'b2000_part2']:
                    if part in embedding_times:
                        if csv_time > embedding_times[part]:
                            stale_embeddings.append(f"{part} (CSV modified after embedding)")
            else:
                if csv_period in embedding_times:
                    if csv_time > embedding_times[csv_period]:
                        stale_embeddings.append(f"{csv_period} (CSV modified after embedding)")

        if stale_embeddings:
            print(f"❌ WARNING: Potentially stale embeddings: {', '.join(stale_embeddings)}")
            print("   Run update_embedding.py to refresh")
        else:
            print("✅ All embeddings are up to date with their source CSVs")

    except Exception as e:
        print(f"❌ WARNING: Could not check file modification times: {e}")
    print()

    # Final summary
    print("="*60)
    if all_tests_passed:
        print("✅ ALL TESTS PASSED - Embeddings are consistent!")
        print("   The app should work without data mismatch errors.")
    else:
        print("❌ SOME TESTS FAILED - Please fix issues before running the app!")
        print("   Running the app now may result in data mismatch errors.")
    print("="*60)

    return all_tests_passed


def diagnose_mismatch():
    """
    Detailed diagnosis when there's a mismatch.
    Helps identify which specific files are causing issues.
    """
    print("\nDETAILED MISMATCH DIAGNOSIS")
    print("="*60)

    # Load each CSV file individually
    csv_counts = {}
    for period in ['b2000', '2000s', '2010s', '2015s', '2020s']:
        csv_path = f'../Data/papers_{period}.csv'
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            csv_counts[period] = len(df)
            print(f"CSV {period}: {len(df):,} papers")

    print(f"\nTotal in CSVs (before deduplication): {sum(csv_counts.values()):,}")

    # Load combined data
    df_all = load_all_papers(data_dir="../Data")
    print(f"Total after deduplication: {len(df_all):,}")
    print(f"Duplicates removed: {sum(csv_counts.values()) - len(df_all):,}")

    # Check embedding counts
    print("\nEmbedding file counts:")
    embedding_counts = {}
    for period in ['b2000_part1', 'b2000_part2', '2000s', '2010s', '2015s', '2020s']:
        path = f'../Embeddings/embeddings_{period}.npy'
        if os.path.exists(path):
            embeddings = np.load(path)
            embedding_counts[period] = len(embeddings)
            print(f"Embeddings {period}: {len(embeddings):,}")

    print(f"\nTotal embeddings: {sum(embedding_counts.values()):,}")

    # Identify problem areas
    print("\nPotential issues:")
    if sum(embedding_counts.values()) > len(df_all):
        print("- More embeddings than papers: likely duplicates not removed from embeddings")
        print("- Solution: Run generate_embedding.py to regenerate all embeddings")
    elif sum(embedding_counts.values()) < len(df_all):
        print("- Fewer embeddings than papers: some papers missing embeddings")
        print("- Solution: Run update_embedding.py or generate_embedding.py")


def main():
    """Main function to run tests."""
    # Check if we're in the right directory
    if not os.path.exists('../Data') or not os.path.exists('../Embeddings'):
        print("Error: Cannot find Data or Embeddings directories.")
        print("Please run this script from the Code directory.")
        return

    # Run main consistency check
    tests_passed = check_embedding_consistency()

    # If tests failed, run detailed diagnosis
    if not tests_passed:
        diagnose_mismatch()
        print("\nRecommended action:")
        print("1. First try: python update_embedding.py")
        print("2. If that fails: python generate_embedding.py")
    else:
        print("\n✅ Your embeddings are ready to use!")


if __name__ == "__main__":
    main()
