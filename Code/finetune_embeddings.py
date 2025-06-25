"""
Fine-tuning with custom save/load mechanism that works around DTensor issues.
This uses the state dict method which your test confirmed works.
"""

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, InputExample, losses, models
from torch.utils.data import DataLoader
import torch
import os
import sys
import json
import random
from datetime import datetime
from tqdm import tqdm
import logging
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoModel, AutoTokenizer
import warnings
warnings.filterwarnings('ignore')

# Add parent directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_processing import load_all_papers

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
logger.info(f"Using device: {device}")

if device.type == "mps":
    os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'


def custom_save_model(model, output_dir):
    """Custom save that works around DTensor issues"""
    os.makedirs(output_dir, exist_ok=True)

    logger.info("Using custom save method...")

    # Get components
    transformer = model[0]
    pooling = model[1]

    # Save transformer state dict
    transformer_model = transformer.auto_model
    tokenizer = transformer.tokenizer

    # Save model weights
    torch.save(transformer_model.state_dict(), os.path.join(output_dir, "pytorch_model.bin"))
    logger.info("✅ Saved model weights")

    # Save tokenizer
    tokenizer.save_pretrained(output_dir)
    logger.info("✅ Saved tokenizer")

    # Save config
    transformer_model.config.save_pretrained(output_dir)
    logger.info("✅ Saved config")

    # Save pooling configuration
    pooling_config = {
        "word_embedding_dimension": pooling.word_embedding_dimension,
        "pooling_mode_cls_token": pooling.pooling_mode_cls_token,
        "pooling_mode_mean_tokens": pooling.pooling_mode_mean_tokens,
        "pooling_mode_max_tokens": pooling.pooling_mode_max_tokens,
        "pooling_mode_mean_sqrt_len_tokens": pooling.pooling_mode_mean_sqrt_len_tokens,
        "pooling_mode_weightedmean_tokens": pooling.pooling_mode_weightedmean_tokens,
        "pooling_mode_lasttoken": pooling.pooling_mode_lasttoken,
    }

    with open(os.path.join(output_dir, 'pooling_config.json'), 'w') as f:
        json.dump(pooling_config, f)
    logger.info("✅ Saved pooling config")

    # Save sentence transformer config
    st_config = {
        "model_type": "sentence-transformers",
        "max_seq_length": transformer.max_seq_length,
        "do_lower_case": transformer.do_lower_case,
    }

    with open(os.path.join(output_dir, 'sentence_transformer_config.json'), 'w') as f:
        json.dump(st_config, f)
    logger.info("✅ Saved sentence transformer config")

    return True


def custom_load_model(model_path):
    """Custom load that works around DTensor issues"""
    logger.info(f"Loading model from {model_path} using custom method...")

    # Load transformer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    transformer_model = AutoModel.from_pretrained(model_path)

    # Load configs
    with open(os.path.join(model_path, 'pooling_config.json'), 'r') as f:
        pooling_config = json.load(f)

    with open(os.path.join(model_path, 'sentence_transformer_config.json'), 'r') as f:
        st_config = json.load(f)

    # Create transformer module
    transformer = models.Transformer(
        model=transformer_model,
        tokenizer=tokenizer,
        max_seq_length=st_config.get('max_seq_length', 512),
        do_lower_case=st_config.get('do_lower_case', False)
    )

    # Create pooling module
    pooling = models.Pooling(
        transformer.get_word_embedding_dimension(),
        pooling_mode_cls_token=pooling_config.get('pooling_mode_cls_token', False),
        pooling_mode_mean_tokens=pooling_config.get('pooling_mode_mean_tokens', True),
        pooling_mode_max_tokens=pooling_config.get('pooling_mode_max_tokens', False),
        pooling_mode_mean_sqrt_len_tokens=pooling_config.get('pooling_mode_mean_sqrt_len_tokens', False),
        pooling_mode_weightedmean_tokens=pooling_config.get('pooling_mode_weightedmean_tokens', False),
        pooling_mode_lasttoken=pooling_config.get('pooling_mode_lasttoken', False),
    )

    # Create sentence transformer
    model = SentenceTransformer(modules=[transformer, pooling])

    logger.info("✅ Model loaded successfully")
    return model


def extract_vocabulary_fast(df, n_terms=3000):
    """Fast vocabulary extraction"""
    logger.info("Fast vocabulary extraction...")

    sample_size = min(30000, len(df))
    sampled = df.sample(n=sample_size, random_state=42)

    texts = (sampled['title'] + ' ' + sampled['abstract'].fillna('')).tolist()

    tfidf = TfidfVectorizer(
        max_features=n_terms,
        ngram_range=(1, 2),
        min_df=3,
        max_df=0.3,
        stop_words='english'
    )

    tfidf_matrix = tfidf.fit_transform(texts)
    feature_names = tfidf.get_feature_names_out()

    scores = tfidf_matrix.sum(axis=0).A1
    vocabulary = list(zip(feature_names, scores))
    vocabulary.sort(key=lambda x: x[1], reverse=True)

    logger.info(f"Extracted {len(vocabulary)} terms")
    logger.info(f"Top terms: {[t[0] for t in vocabulary[:20]]}")

    return vocabulary


def generate_training_pairs_fast(df, vocabulary, n_examples=80000):
    """Generate training pairs without expensive searches"""
    df = df.reset_index(drop=True)
    examples = []

    logger.info("Pre-computing text representations...")
    df['text'] = df['title'] + ' ' + df['abstract'].fillna('')

    # 1. Title-Abstract pairs (40%)
    logger.info("Generating title-abstract pairs...")
    valid_mask = df['abstract'].notna() & (df['abstract'].str.len() > 50)
    valid_indices = df[valid_mask].index.tolist()

    n_title_abs = min(n_examples * 4 // 10, len(valid_indices))
    sampled_indices = random.sample(valid_indices, n_title_abs)

    for idx in tqdm(sampled_indices, desc="Title-Abstract"):
        row = df.iloc[idx]
        examples.append(InputExample(
            texts=[row['title'], row['abstract'][:512]],
            label=1.0
        ))

    # 2. Same journal/year pairs (25%)
    logger.info("Generating same journal/year pairs...")
    journal_year_groups = df.groupby(['journal', 'year']).groups

    n_journal = n_examples // 4
    journal_pairs = []

    for group_key, indices in journal_year_groups.items():
        if len(indices) >= 2:
            indices_list = indices.tolist()
            n_pairs = min(20, len(indices_list) // 2)
            for _ in range(n_pairs):
                if len(journal_pairs) >= n_journal:
                    break
                idx1, idx2 = random.sample(indices_list, 2)
                journal_pairs.append((idx1, idx2))
        if len(journal_pairs) >= n_journal:
            break

    for idx1, idx2 in tqdm(journal_pairs, desc="Journal pairs"):
        examples.append(InputExample(
            texts=[df.iloc[idx1]['text'][:512],
                  df.iloc[idx2]['text'][:512]],
            label=0.7
        ))

    # 3. Smart Query-Document pairs (25%)
    logger.info("Generating query-document pairs (fast method)...")
    n_queries = n_examples // 4

    logger.info("  Creating paper-term associations...")
    top_terms = [term for term, _ in vocabulary[:500]]

    paper_terms = {}
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="  Indexing papers"):
        title_lower = row['title'].lower()
        found_terms = [term for term in top_terms if len(term.split()) <= 2 and term in title_lower]
        if found_terms:
            paper_terms[idx] = found_terms

    query_templates = [
        "{} analysis",
        "{} and {} study",
        "impact of {}",
        "{} empirical evidence",
        "effects of {} on {}",
        "{} in economics"
    ]

    papers_with_terms = list(paper_terms.keys())
    if papers_with_terms:
        for _ in tqdm(range(n_queries), desc="  Creating queries"):
            idx = random.choice(papers_with_terms)
            terms = paper_terms[idx]

            template = random.choice(query_templates)
            n_terms_needed = template.count('{}')
            selected_terms = random.sample(terms, min(n_terms_needed, len(terms)))

            if len(selected_terms) < n_terms_needed:
                selected_terms.extend(['economic', 'policy', 'market'][:n_terms_needed - len(selected_terms)])

            try:
                query = template.format(*selected_terms[:n_terms_needed])
                paper = df.iloc[idx]
                examples.append(InputExample(
                    texts=[query, paper['text'][:512]],
                    label=0.9
                ))
            except:
                continue

    # 4. Random negative pairs (10%)
    logger.info("Generating negative pairs...")
    n_neg = n_examples // 10

    old_papers = df[df['year'] < 2000].index.tolist()
    new_papers = df[df['year'] > 2015].index.tolist()

    if old_papers and new_papers:
        for _ in range(min(n_neg, len(old_papers), len(new_papers))):
            idx1 = random.choice(old_papers)
            idx2 = random.choice(new_papers)

            examples.append(InputExample(
                texts=[df.iloc[idx1]['text'][:512],
                      df.iloc[idx2]['text'][:512]],
                label=0.1
            ))

    remaining = n_neg - len([e for e in examples if e.label == 0.1])
    for _ in range(remaining):
        idx1, idx2 = random.sample(range(len(df)), 2)
        if df.iloc[idx1]['journal'] != df.iloc[idx2]['journal']:
            examples.append(InputExample(
                texts=[df.iloc[idx1]['text'][:512],
                      df.iloc[idx2]['text'][:512]],
                label=0.1
            ))

    random.shuffle(examples)
    logger.info(f"Generated {len(examples)} training examples")
    return examples[:n_examples]


def train_with_custom_save():
    """Train model with custom save method"""

    # Configuration
    BASE_MODEL = 'sentence-transformers/all-MiniLM-L6-v2'
    OUTPUT_DIR = '../Models/economics-fine-tuned'
    NUM_EPOCHS = 2
    BATCH_SIZE = 32
    LEARNING_RATE = 2e-5

    # Ensure Models directory exists
    os.makedirs('../Models', exist_ok=True)

    # Load model
    logger.info(f"Loading base model: {BASE_MODEL}")
    model = SentenceTransformer(BASE_MODEL)

    # Load papers
    logger.info("Loading economics papers...")
    df = load_all_papers(data_dir="../Data")
    logger.info(f"Loaded {len(df)} papers")

    # Extract vocabulary
    vocabulary = extract_vocabulary_fast(df, n_terms=3000)

    # Generate training pairs
    train_examples = generate_training_pairs_fast(df, vocabulary, n_examples=80000)

    # Create DataLoader
    train_dataloader = DataLoader(
        train_examples,
        shuffle=True,
        batch_size=BATCH_SIZE,
        num_workers=0
    )

    # Loss function
    train_loss = losses.MultipleNegativesRankingLoss(model)

    # Training parameters
    num_training_steps = len(train_dataloader) * NUM_EPOCHS
    warmup_steps = int(0.1 * num_training_steps)

    logger.info(f"""
    ========================================
    Training Configuration:
    - Model: {BASE_MODEL}
    - Device: {device}
    - Training examples: {len(train_examples):,}
    - Batch size: {BATCH_SIZE}
    - Epochs: {NUM_EPOCHS}
    - Steps: {num_training_steps}
    - Output: {OUTPUT_DIR}
    ========================================
    """)

    # Train
    start_time = datetime.now()

    # Train without auto-saving
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=NUM_EPOCHS,
        warmup_steps=warmup_steps,
        output_path=None,  # Don't auto-save
        optimizer_class=torch.optim.AdamW,
        optimizer_params={'lr': LEARNING_RATE},
        show_progress_bar=True,
        save_best_model=False,
        checkpoint_path=None,
        checkpoint_save_steps=0
    )

    training_time = datetime.now() - start_time
    logger.info(f"Training completed in {training_time}")

    # Save with custom method
    logger.info("Saving model with custom method...")
    save_success = custom_save_model(model, OUTPUT_DIR)

    if save_success:
        # Test loading
        logger.info("Testing if model loads correctly...")
        try:
            loaded_model = custom_load_model(OUTPUT_DIR)
            test_embedding = loaded_model.encode("economics test")
            logger.info(f"✅ Model loads correctly! Test embedding shape: {test_embedding.shape}")
            load_success = True
        except Exception as e:
            logger.warning(f"⚠️  Custom load failed: {e}")
            load_success = False

        # Save metadata
        metadata = {
            'base_model': BASE_MODEL,
            'training_date': datetime.now().isoformat(),
            'training_time': str(training_time),
            'num_papers': len(df),
            'training_examples': len(train_examples),
            'vocabulary_terms': len(vocabulary),
            'top_terms': [t[0] for t in vocabulary[:50]],
            'device': str(device),
            'save_method': 'custom_state_dict',
            'load_method': 'custom_load_model' if load_success else 'requires_custom_load'
        }

        with open(os.path.join(OUTPUT_DIR, 'training_metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)

        # Also save a loading script for convenience
        load_script = '''"""
Script to load the fine-tuned economics model.
This model requires custom loading due to DTensor issues.
"""

from sentence_transformers import SentenceTransformer, models
from transformers import AutoModel, AutoTokenizer
import json
import os


def load_economics_model(model_path):
    """Load the fine-tuned economics model"""

    # Load transformer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    transformer_model = AutoModel.from_pretrained(model_path)

    # Load configs
    with open(os.path.join(model_path, 'pooling_config.json'), 'r') as f:
        pooling_config = json.load(f)

    with open(os.path.join(model_path, 'sentence_transformer_config.json'), 'r') as f:
        st_config = json.load(f)

    # Create modules
    transformer = models.Transformer(
        model=transformer_model,
        tokenizer=tokenizer,
        max_seq_length=st_config.get('max_seq_length', 512),
        do_lower_case=st_config.get('do_lower_case', False)
    )

    pooling = models.Pooling(
        transformer.get_word_embedding_dimension(),
        pooling_mode_cls_token=pooling_config.get('pooling_mode_cls_token', False),
        pooling_mode_mean_tokens=pooling_config.get('pooling_mode_mean_tokens', True),
        pooling_mode_max_tokens=pooling_config.get('pooling_mode_max_tokens', False),
    )

    # Create model
    model = SentenceTransformer(modules=[transformer, pooling])
    return model


# Usage:
# model = load_economics_model('path/to/Models/economics-fine-tuned')
'''

        with open(os.path.join(OUTPUT_DIR, 'load_model.py'), 'w') as f:
            f.write(load_script)

        logger.info(f"""
        ========================================
        ✅ TRAINING AND SAVE SUCCESSFUL!
        ========================================
        Training time: {training_time}
        Model saved to: {OUTPUT_DIR}

        ⚠️  IMPORTANT: This model requires custom loading!

        To use in your app, update semantic_search.py:

        from Models.economics-fine-tuned.load_model import load_economics_model

        # In load_semantic_model():
        if os.path.exists('Models/economics-fine-tuned'):
            model = load_economics_model('Models/economics-fine-tuned')
        else:
            model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

        ========================================
        """)
    else:
        logger.error("Model save failed!")


if __name__ == "__main__":
    if sys.platform == "darwin":
        import multiprocessing
        multiprocessing.set_start_method('spawn', force=True)

    train_with_custom_save()
