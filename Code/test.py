"""
Proper test to verify embedding-paper alignment
"""

import numpy as np
import pandas as pd
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_processing import load_all_papers


def test_alignment():
    """Test if papers and embeddings are properly aligned"""

    print("Loading data and embeddings...\n")

    # Load data
    df = load_all_papers()
    print(f"Total papers in dataframe: {len(df)}")

    # Load embeddings in the same way as app.py
    all_embeddings = []

    if os.path.exists('../Embeddings/embeddings_b2000_part1.npy'):
        embeddings_part1 = np.load('../Embeddings/embeddings_b2000_part1.npy').astype(np.float32)
        all_embeddings.append(embeddings_part1)
        print(f"Loaded b2000_part1 embeddings: {len(embeddings_part1)}")

    if os.path.exists('../Embeddings/embeddings_b2000_part2.npy'):
        embeddings_part2 = np.load('../Embeddings/embeddings_b2000_part2.npy').astype(np.float32)
        all_embeddings.append(embeddings_part2)
        print(f"Loaded b2000_part2 embeddings: {len(embeddings_part2)}")

    for period in ['2000s', '2010s', '2015s', '2020s']:
        path = f'../Embeddings/embeddings_{period}.npy'
        if os.path.exists(path):
            embeddings = np.load(path).astype(np.float32)
            all_embeddings.append(embeddings)
            print(f"Loaded {period} embeddings: {len(embeddings)}")

    embeddings_concat = np.vstack(all_embeddings) if all_embeddings else None
    print(f"\nTotal embeddings: {len(embeddings_concat)}")

    # Check if sizes match
    if len(df) != len(embeddings_concat):
        print(f"\n⚠️  ERROR: Size mismatch!")
        print(f"   Dataframe: {len(df)} papers")
        print(f"   Embeddings: {len(embeddings_concat)} embeddings")
        return

    print("\n" + "="*60)
    print("ALIGNMENT TEST - Cross-checking embeddings:")
    print("="*60)

    # Test alignment by checking if each embedding's nearest neighbor is itself
    test_indices = [0, 1000, 5000, 10000, 20000, 50000, 100000, 150000]

    misaligned = 0
    for idx in test_indices:
        if idx < len(df):
            paper = df.iloc[idx]
            embedding = embeddings_concat[idx]

            # Find most similar papers to this embedding
            similarities = np.dot(embeddings_concat, embedding)
            top_5_indices = np.argsort(similarities)[-5:][::-1]  # Top 5 most similar

            print(f"\nIndex {idx}: {paper['title'][:60]}... ({paper['year']})")
            print(f"Top 5 most similar papers by embedding:")

            for rank, similar_idx in enumerate(top_5_indices):
                similar_paper = df.iloc[similar_idx]
                sim_score = similarities[similar_idx]
                marker = "✅" if similar_idx == idx else "  "
                print(f"  {marker} Rank {rank+1}: Index {similar_idx} (sim={sim_score:.3f})")
                print(f"     {similar_paper['title'][:60]}... ({similar_paper['year']})")

            if top_5_indices[0] != idx:
                print("  ❌ MISALIGNED! This paper's embedding doesn't match itself best!")
                misaligned += 1

    print(f"\n{'='*60}")
    print(f"ALIGNMENT SUMMARY: {misaligned} misaligned out of {len([i for i in test_indices if i < len(df)])} tested")
    print(f"{'='*60}")

    # Test the split point between b2000 parts
    print("\n" + "="*60)
    print("TESTING B2000 SPLIT BOUNDARY:")
    print("="*60)

    # Find where b2000 ends
    b2000_count = len(df[df.year < 2000])
    split_point = b2000_count // 2

    print(f"\nTotal b2000 papers: {b2000_count}")
    print(f"Split point: {split_point}")

    # Check papers around the split
    for idx in [split_point-2, split_point-1, split_point, split_point+1]:
        paper = df.iloc[idx]
        print(f"\nIndex {idx}: {paper['title'][:60]}... ({paper['year']})")

    # Additional test: Create the same text as generate_embedding.py
    print("\n" + "="*60)
    print("TESTING TEXT GENERATION CONSISTENCY:")
    print("="*60)

    for idx in [0, 1000]:
        paper = df.iloc[idx]

        # Method 1 (from generate_embedding.py)
        text1 = paper['title'] + ' ' + (paper['abstract'] if pd.notna(paper['abstract']) else '')

        # Method 2 (from original test.py)
        text2 = paper['title'] + ' ' + str(paper.get('abstract', ''))

        print(f"\nIndex {idx}:")
        print(f"Title: {paper['title'][:60]}...")
        print(f"Abstract is null: {pd.isna(paper['abstract'])}")
        print(f"Text equal: {text1 == text2}")
        if text1 != text2:
            print(f"Text1 ends with: ...{repr(text1[-20:])}")
            print(f"Text2 ends with: ...{repr(text2[-20:])}")


if __name__ == "__main__":
    test_alignment()
