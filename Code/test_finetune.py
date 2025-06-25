"""
Comprehensive test for fine-tuning process with multiple save/load methods.
Tests all possible workarounds for the DTensor issue.
Run from Code/: python test_finetune_comprehensive.py
"""

import os
import sys
import torch
import pandas as pd
from sentence_transformers import SentenceTransformer, InputExample, losses, models
from torch.utils.data import DataLoader
import json
from datetime import datetime
import shutil
from transformers import AutoModel, AutoTokenizer

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_processing import load_all_papers


def test_environment():
    """Test that the environment is set up correctly"""
    print("="*60)
    print("1. TESTING ENVIRONMENT")
    print("="*60)

    print(f"Python version: {sys.version}")
    print(f"PyTorch version: {torch.__version__}")

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")

    import sentence_transformers
    print(f"Sentence-transformers version: {sentence_transformers.__version__}")

    return True


def test_save_methods():
    """Test different save methods to find what works"""
    print("\n" + "="*60)
    print("2. TESTING SAVE METHODS")
    print("="*60)

    # Load base model
    print("Loading base model...")
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    print("✅ Model loaded")

    working_methods = []

    # Method 1: Standard save
    print("\n--- Method 1: Standard save ---")
    test_dir = '../Models/test-method1'
    try:
        model.save(test_dir)
        print("✅ Standard save successful")
        working_methods.append(("standard", test_dir))
    except Exception as e:
        print(f"❌ Standard save failed: {e}")

    # Method 2: Save with safe_serialization=False
    print("\n--- Method 2: Save with safe_serialization=False ---")
    test_dir = '../Models/test-method2'
    try:
        model.save(test_dir, safe_serialization=False)
        print("✅ Save with safe_serialization=False successful")
        working_methods.append(("safe_serialization_false", test_dir))
    except Exception as e:
        print(f"❌ Safe_serialization=False failed: {e}")

    # Method 3: Manual component save
    print("\n--- Method 3: Manual component save ---")
    test_dir = '../Models/test-method3'
    try:
        os.makedirs(test_dir, exist_ok=True)

        # Get components
        transformer = model[0]
        pooling = model[1]

        # Save transformer state dict
        transformer_model = transformer.auto_model
        tokenizer = transformer.tokenizer

        # Save model weights
        torch.save(transformer_model.state_dict(), os.path.join(test_dir, "pytorch_model.bin"))

        # Save tokenizer
        tokenizer.save_pretrained(test_dir)

        # Save config
        transformer_model.config.save_pretrained(test_dir)

        # Save pooling configuration
        pooling_config = {
            "word_embedding_dimension": pooling.word_embedding_dimension,
            "pooling_mode_cls_token": pooling.pooling_mode_cls_token,
            "pooling_mode_mean_tokens": pooling.pooling_mode_mean_tokens,
            "pooling_mode_max_tokens": pooling.pooling_mode_max_tokens,
        }

        with open(os.path.join(test_dir, 'pooling_config.json'), 'w') as f:
            json.dump(pooling_config, f)

        # Save sentence transformer config
        st_config = {
            "max_seq_length": transformer.max_seq_length,
            "do_lower_case": transformer.do_lower_case,
        }

        with open(os.path.join(test_dir, 'sentence_transformer_config.json'), 'w') as f:
            json.dump(st_config, f)

        print("✅ Manual component save successful")
        working_methods.append(("manual_components", test_dir))

    except Exception as e:
        print(f"❌ Manual component save failed: {e}")

    # Test loading for each successful method
    print("\n" + "="*60)
    print("3. TESTING LOAD METHODS")
    print("="*60)

    loadable_methods = []

    for method_name, save_dir in working_methods:
        print(f"\n--- Testing load for {method_name} ---")

        # Try standard load
        try:
            loaded_model = SentenceTransformer(save_dir)
            test_embedding = loaded_model.encode("test")
            print(f"✅ Standard load successful for {method_name}")
            loadable_methods.append((method_name, "standard_load", save_dir))
        except Exception as e:
            print(f"❌ Standard load failed: {e}")

            # Try manual load for manual_components method
            if method_name == "manual_components":
                try:
                    print("   Trying manual load...")
                    tokenizer = AutoTokenizer.from_pretrained(save_dir)
                    transformer_model = AutoModel.from_pretrained(save_dir)

                    with open(os.path.join(save_dir, 'pooling_config.json'), 'r') as f:
                        pooling_config = json.load(f)

                    with open(os.path.join(save_dir, 'sentence_transformer_config.json'), 'r') as f:
                        st_config = json.load(f)

                    transformer = models.Transformer(
                        model=transformer_model,
                        tokenizer=tokenizer,
                        max_seq_length=st_config.get('max_seq_length', 512)
                    )

                    pooling = models.Pooling(
                        transformer.get_word_embedding_dimension(),
                        pooling_mode_mean_tokens=pooling_config.get('pooling_mode_mean_tokens', True)
                    )

                    loaded_model = SentenceTransformer(modules=[transformer, pooling])
                    test_embedding = loaded_model.encode("test")
                    print(f"✅ Manual load successful for {method_name}")
                    loadable_methods.append((method_name, "manual_load", save_dir))

                except Exception as e2:
                    print(f"❌ Manual load also failed: {e2}")

    # Clean up test directories
    print("\nCleaning up test directories...")
    for _, save_dir in working_methods:
        if os.path.exists(save_dir):
            shutil.rmtree(save_dir)

    return loadable_methods


def test_full_pipeline(save_method_info):
    """Test a complete fine-tuning pipeline with working save method"""
    print("\n" + "="*60)
    print("4. TESTING FULL FINE-TUNING PIPELINE")
    print("="*60)

    if not save_method_info:
        print("❌ No working save method found. Cannot test full pipeline.")
        return False

    method_name, load_type, _ = save_method_info[0]
    print(f"Using save method: {method_name} with {load_type}")

    try:
        # 1. Load data
        print("\n1. Loading data...")
        df = load_all_papers("../Data")
        print(f"✅ Loaded {len(df)} papers")

        # 2. Create training data (small sample)
        print("\n2. Creating training data...")
        train_examples = []

        # Title-abstract pairs
        valid_papers = df[df['abstract'].notna()].head(100)
        for _, row in valid_papers.iterrows():
            train_examples.append(InputExample(
                texts=[row['title'], row['abstract'][:512]],
                label=1.0
            ))

        # Same journal pairs
        for journal in df['journal'].value_counts().head(5).index:
            journal_papers = df[df['journal'] == journal].head(10)
            if len(journal_papers) >= 2:
                for i in range(len(journal_papers) - 1):
                    train_examples.append(InputExample(
                        texts=[
                            journal_papers.iloc[i]['title'],
                            journal_papers.iloc[i+1]['title']
                        ],
                        label=0.7
                    ))

        print(f"✅ Created {len(train_examples)} training examples")

        # 3. Train model
        print("\n3. Training model...")
        model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

        train_dataloader = DataLoader(train_examples[:50], shuffle=True, batch_size=8)
        train_loss = losses.MultipleNegativesRankingLoss(model)

        model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            epochs=1,
            warmup_steps=0,
            output_path=None,
            show_progress_bar=True
        )
        print("✅ Training completed")

        # 4. Save model using working method
        print("\n4. Saving model...")
        save_dir = '../Models/test-full-pipeline'

        if method_name == "manual_components":
            # Use manual save
            os.makedirs(save_dir, exist_ok=True)
            transformer = model[0]
            pooling = model[1]

            torch.save(transformer.auto_model.state_dict(),
                      os.path.join(save_dir, "pytorch_model.bin"))
            transformer.tokenizer.save_pretrained(save_dir)
            transformer.auto_model.config.save_pretrained(save_dir)

            pooling_config = {
                "word_embedding_dimension": pooling.word_embedding_dimension,
                "pooling_mode_mean_tokens": pooling.pooling_mode_mean_tokens,
            }
            with open(os.path.join(save_dir, 'pooling_config.json'), 'w') as f:
                json.dump(pooling_config, f)

            st_config = {
                "max_seq_length": transformer.max_seq_length,
            }
            with open(os.path.join(save_dir, 'sentence_transformer_config.json'), 'w') as f:
                json.dump(st_config, f)

        elif method_name == "safe_serialization_false":
            model.save(save_dir, safe_serialization=False)
        else:
            model.save(save_dir)

        print("✅ Model saved")

        # 5. Test loading
        print("\n5. Loading saved model...")
        if load_type == "manual_load":
            # Use manual load
            tokenizer = AutoTokenizer.from_pretrained(save_dir)
            transformer_model = AutoModel.from_pretrained(save_dir)

            with open(os.path.join(save_dir, 'sentence_transformer_config.json'), 'r') as f:
                st_config = json.load(f)

            transformer = models.Transformer(
                model=transformer_model,
                tokenizer=tokenizer,
                max_seq_length=st_config.get('max_seq_length', 512)
            )

            pooling = models.Pooling(
                transformer.get_word_embedding_dimension(),
                pooling_mode_mean_tokens=True
            )

            loaded_model = SentenceTransformer(modules=[transformer, pooling])
        else:
            loaded_model = SentenceTransformer(save_dir)

        # Test the loaded model
        test_text = "monetary policy and inflation"
        embedding = loaded_model.encode(test_text)
        print(f"✅ Model loaded and working! Embedding shape: {embedding.shape}")

        # Clean up
        shutil.rmtree(save_dir)

        return True

    except Exception as e:
        print(f"❌ Full pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def generate_save_functions(method_info):
    """Generate the save/load functions to use in actual training"""
    print("\n" + "="*60)
    print("5. GENERATING SAVE/LOAD FUNCTIONS")
    print("="*60)

    if not method_info:
        print("❌ No working method found")
        return

    method_name, load_type, _ = method_info[0]

    print(f"Best working method: {method_name} with {load_type}")
    print("\nAdd this to your fine-tuning script:")
    print("-"*40)

    if method_name == "manual_components":
        print("""
def save_model(model, save_dir):
    '''Save model using manual component method'''
    os.makedirs(save_dir, exist_ok=True)

    transformer = model[0]
    pooling = model[1]

    # Save model weights
    torch.save(transformer.auto_model.state_dict(),
              os.path.join(save_dir, "pytorch_model.bin"))

    # Save tokenizer and config
    transformer.tokenizer.save_pretrained(save_dir)
    transformer.auto_model.config.save_pretrained(save_dir)

    # Save pooling config
    pooling_config = {
        "word_embedding_dimension": pooling.word_embedding_dimension,
        "pooling_mode_mean_tokens": pooling.pooling_mode_mean_tokens,
    }
    with open(os.path.join(save_dir, 'pooling_config.json'), 'w') as f:
        json.dump(pooling_config, f)

    # Save sentence transformer config
    st_config = {
        "max_seq_length": transformer.max_seq_length,
    }
    with open(os.path.join(save_dir, 'sentence_transformer_config.json'), 'w') as f:
        json.dump(st_config, f)

    print(f"Model saved to {save_dir}")
""")

    elif method_name == "safe_serialization_false":
        print("""
def save_model(model, save_dir):
    '''Save model with safe_serialization=False'''
    model.save(save_dir, safe_serialization=False)
    print(f"Model saved to {save_dir}")
""")

    else:
        print("""
def save_model(model, save_dir):
    '''Save model using standard method'''
    model.save(save_dir)
    print(f"Model saved to {save_dir}")
""")


def main():
    """Run all tests"""
    print("COMPREHENSIVE FINE-TUNING TEST")
    print("="*60)
    print(f"Time: {datetime.now()}")
    print(f"Working directory: {os.getcwd()}")
    print("="*60)

    # Test 1: Environment
    test_environment()

    # Test 2: Try different save methods
    working_methods = test_save_methods()

    # Test 3: Full pipeline with working method
    if working_methods:
        print(f"\n✅ Found {len(working_methods)} working save/load method(s)")
        pipeline_success = test_full_pipeline(working_methods)
    else:
        print("\n❌ No working save/load methods found")
        pipeline_success = False

    # Generate recommended functions
    generate_save_functions(working_methods)

    # Final summary
    print("\n" + "="*60)
    print("FINAL SUMMARY")
    print("="*60)

    if working_methods and pipeline_success:
        print("✅ Ready for fine-tuning!")
        print(f"   Working save method: {working_methods[0][0]}")
        print("\nNext step: Create your fine-tuning script with the save_model function above")
    else:
        print("❌ Issues found:")
        if not working_methods:
            print("   - No working save methods")
            print("   - Try: pip install sentence-transformers==3.0.0")
        elif not pipeline_success:
            print("   - Save methods work but full pipeline failed")


if __name__ == "__main__":
    main()
