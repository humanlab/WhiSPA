#!/usr/bin/env python3
"""
Test script for IEMOCAP dataset loader
"""

import sys
import os
import numpy as np

# Add project root to path
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from data.iemocap.dataset import IEMOCAPDataset

def test_iemocap_dataset():
    """Test IEMOCAP dataset loading and basic functionality."""
    print("Testing IEMOCAP dataset loader...")
    
    try:
        # Initialize dataset
        dataset = IEMOCAPDataset()
        print(f"\n✓ Dataset loaded: {dataset}")
        
        # Check dataset size
        print(f"\n✓ Total samples: {len(dataset)}")
        
        # Check splits
        splits = dataset.df['split'].value_counts()
        print(f"\n✓ Splits: {splits.to_dict()}")
        
        # Check outcomes
        print("\n✓ Available outcomes:")
        regression_outcomes = []
        classification_outcomes = []
        
        for outcome, outcome_type in dataset.OUTCOMES.items():
            info = dataset.get_outcome_info(outcome)
            if outcome_type == 'regression':
                regression_outcomes.append(outcome)
                print(f"  - {outcome}: {info['type']}")
            else:
                classification_outcomes.append(outcome)
                print(f"  - {outcome}: {info['type']}, {info['n_classes']} classes")
                if info['n_classes'] <= 10:
                    print(f"    Classes: {info['classes']}")
        
        print(f"\n  Total regression tasks: {len(regression_outcomes)}")
        print(f"  Total classification tasks: {len(classification_outcomes)}")
        
        # Test data access methods
        print("\n✓ Testing data access methods:")
        audio_paths = dataset.get_audio_paths()
        print(f"  - Audio paths: {len(audio_paths)} samples")
        if audio_paths:
            print(f"    Example: {audio_paths[0]}")
        
        transcriptions = dataset.get_transcriptions()
        print(f"  - Transcriptions: {len(transcriptions)} samples")
        if transcriptions:
            print(f"    Example: '{transcriptions[0][:50]}...'")
        
        sample_ids = dataset.get_sample_ids()
        print(f"  - Sample IDs: {len(sample_ids)} samples")
        if sample_ids:
            print(f"    Example: {sample_ids[0]}")
        
        # Test target extraction for regression tasks
        print("\n✓ Testing regression target extraction:")
        for outcome in regression_outcomes[:3]:  # Test first 3 regression tasks
            targets = dataset.get_targets(outcome)
            print(f"  - {outcome}: shape {targets.shape}, dtype {targets.dtype}")
            print(f"    Range: [{np.min(targets):.3f}, {np.max(targets):.3f}], mean: {np.mean(targets):.3f}")
        
        # Test target extraction for classification tasks
        print("\n✓ Testing classification target extraction:")
        for outcome in classification_outcomes:
            targets = dataset.get_targets(outcome, encoded=True)
            targets_raw = dataset.get_targets(outcome, encoded=False)
            print(f"  - {outcome}: shape {targets.shape}, unique values: {len(set(targets))}")
            print(f"    Example labels (first 5): {list(set(targets_raw))[:5]}")
        
        # Test stratification labels
        print("\n✓ Testing stratification labels:")
        strat_labels = dataset.get_stratification_labels()
        print(f"  - Shape: {strat_labels.shape}")
        print(f"  - Unique values: {len(set(strat_labels))}")
        
        # Test split-specific data access
        print("\n✓ Testing split-specific data access:")
        for split in ['dev']:  # IEMOCAP only has 'dev' split
            split_data = dataset.get_data(split)
            if len(split_data) > 0:
                print(f"  - {split}: {len(split_data)} samples")
                split_audio = dataset.get_audio_paths(split)
                print(f"    Audio paths: {len(split_audio)}")
        
        # Check data consistency
        print("\n✓ Checking data consistency:")
        df = dataset.get_data()
        print(f"  - DataFrame shape: {df.shape}")
        print(f"  - Columns: {list(df.columns)[:10]}... (showing first 10)")
        
        # Check emotion encoding
        print("\n✓ Checking emotion encoding:")
        print(f"  - Major emotion classes: {dataset.emotion_encoder.classes_}")
        print(f"  - Gender classes: {dataset.gender_encoder.classes_}")
        
        print("\n✓ All tests passed!")
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    test_iemocap_dataset()
