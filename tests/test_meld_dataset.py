#!/usr/bin/env python3
"""
Test script for MELD dataset loader
"""

import sys
import os

# Add project root to path
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from data.meld.dataset import MELDDataset

def test_meld_dataset():
    """Test MELD dataset loading and basic functionality."""
    print("Testing MELD dataset loader...")
    
    try:
        # Initialize dataset
        dataset = MELDDataset(min_speaker_samples=5)
        print(f"\n✓ Dataset loaded: {dataset}")
        
        # Check dataset size
        print(f"\n✓ Total samples: {len(dataset)}")
        
        # Check splits
        splits = dataset.df['split'].value_counts()
        print(f"\n✓ Splits: {splits.to_dict()}")
        
        # Check outcomes
        print("\n✓ Available outcomes:")
        for outcome in dataset.OUTCOMES:
            info = dataset.get_outcome_info(outcome)
            print(f"  - {outcome}: {info['type']}, {info['n_classes']} classes")
            if info['n_classes'] <= 10:
                print(f"    Classes: {info['classes']}")
        
        # Test data access methods
        print("\n✓ Testing data access methods:")
        audio_paths = dataset.get_audio_paths()
        print(f"  - Audio paths: {len(audio_paths)} samples")
        print(f"    Example: {audio_paths[0]}")
        
        transcriptions = dataset.get_transcriptions()
        print(f"  - Transcriptions: {len(transcriptions)} samples")
        print(f"    Example: {transcriptions[0]}")
        
        # Test target extraction
        print("\n✓ Testing target extraction:")
        for outcome in dataset.OUTCOMES:
            targets = dataset.get_targets(outcome, encoded=True)
            print(f"  - {outcome}: shape {targets.shape}, unique values: {len(set(targets))}")
        
        # Get split statistics
        print("\n✓ Split statistics:")
        stats = dataset.get_split_statistics()
        for split, split_stats in stats.items():
            print(f"  - {split}:")
            for key, value in split_stats.items():
                print(f"    {key}: {value}")
        
        print("\n✓ All tests passed!")
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    test_meld_dataset()
