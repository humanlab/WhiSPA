#!/usr/bin/env python3
"""
MELD Dataset Loader for Embedding Evaluation
"""

import os
import logging
from typing import Dict, List, Tuple, Optional, Any
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from collections import Counter
from dotenv import load_dotenv

load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MELDDataset:
    """MELD dataset loader for embedding evaluation."""
    
    # Define outcome types and their properties
    OUTCOMES = {
        # Classification tasks
        'speaker': 'classification',
        'emotion': 'classification',
        'sentiment': 'classification',
    }
    
    # Expected emotion and sentiment labels from the metadata
    EMOTION_LABELS = ['anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'surprise']
    SENTIMENT_LABELS = ['negative', 'neutral', 'positive']
    
    def __init__(self, min_speaker_samples: int = 2):
        """
        Initialize MELD dataset loader.
        
        Args:
            min_speaker_samples: Minimum number of samples for a speaker to have their own label.
                                Speakers with fewer samples will be grouped as "Other".
        """
        dataset_path = os.path.join(os.getenv("MELD_DIR", ""), 'meld.jsonl')
        
        if not os.path.exists(dataset_path):
            raise ValueError(f"Dataset path does not exist: {dataset_path}")
        
        logger.info(f"Loading MELD dataset from: {dataset_path}")
        self.min_speaker_samples = min_speaker_samples
        self.data = []
        self._load_dataset(dataset_path)
        self._process_dataset()
        
    def get_name(self) -> str:
        """Get the name of this dataset."""
        return "meld"
        
    def _load_dataset(self, dataset_path: str):
        """Load the MELD dataset from JSONL file."""
        with open(dataset_path, 'r') as f:
            for line in f:
                try:
                    sample = json.loads(line.strip())
                    self.data.append(sample)
                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse line: {e}")
                    continue
        
        logger.info(f"Loaded {len(self.data)} samples from MELD dataset")
        
    def _process_dataset(self):
        """Process the dataset and extract relevant features."""
        # Convert to DataFrame for easier manipulation
        self.df = pd.DataFrame(self.data)
        
        # Process speakers - group rare speakers as "Other"
        speaker_counts = self.df['speaker'].value_counts()
        rare_speakers = speaker_counts[speaker_counts < self.min_speaker_samples].index
        
        # Create processed speaker column
        self.df['speaker_processed'] = self.df['speaker'].apply(
            lambda x: 'Other' if x in rare_speakers else x
        )
        
        logger.info(f"Unique speakers (original): {self.df['speaker'].nunique()}")
        logger.info(f"Unique speakers (processed): {self.df['speaker_processed'].nunique()}")
        logger.info(f"Speakers grouped as 'Other': {len(rare_speakers)}")
        
        # Validate emotion labels
        unique_emotions = self.df['emotion'].unique()
        unexpected_emotions = set(unique_emotions) - set(self.EMOTION_LABELS)
        if unexpected_emotions:
            logger.warning(f"Unexpected emotion labels found: {unexpected_emotions}")
        
        # Validate sentiment labels
        unique_sentiments = self.df['sentiment'].unique()
        unexpected_sentiments = set(unique_sentiments) - set(self.SENTIMENT_LABELS)
        if unexpected_sentiments:
            logger.warning(f"Unexpected sentiment labels found: {unexpected_sentiments}")
        
        # Encode categorical variables
        self._encode_labels()
        
    def _encode_labels(self):
        """Encode categorical labels for classification tasks."""
        # Encode speakers
        self.speaker_encoder = LabelEncoder()
        self.df['speaker_encoded'] = self.speaker_encoder.fit_transform(self.df['speaker_processed'])
        
        # Encode emotions
        self.emotion_encoder = LabelEncoder()
        # Ensure all expected emotions are included in the encoder
        all_emotions = list(set(self.EMOTION_LABELS) | set(self.df['emotion'].unique()))
        self.emotion_encoder.fit(all_emotions)
        self.df['emotion_encoded'] = self.emotion_encoder.transform(self.df['emotion'])
        
        # Encode sentiments
        self.sentiment_encoder = LabelEncoder()
        # Ensure all expected sentiments are included in the encoder
        all_sentiments = list(set(self.SENTIMENT_LABELS) | set(self.df['sentiment'].unique()))
        self.sentiment_encoder.fit(all_sentiments)
        self.df['sentiment_encoded'] = self.sentiment_encoder.transform(self.df['sentiment'])
        
        logger.info(f"Encoded labels - Speakers: {len(self.speaker_encoder.classes_)}, "
                   f"Emotions: {len(self.emotion_encoder.classes_)}, "
                   f"Sentiments: {len(self.sentiment_encoder.classes_)}")
        
    def get_data(self, split: Optional[str] = None) -> pd.DataFrame:
        """
        Get dataset samples.
        
        Args:
            split: Optional split to filter ('train', 'dev', 'test')
            
        Returns:
            DataFrame containing samples
        """
        if split:
            return self.df[self.df['split'] == split].copy()
        return self.df.copy()
    
    def get_audio_paths(self, split: Optional[str] = None) -> List[str]:
        """Get list of audio file paths."""
        data = self.get_data(split)
        return data['audio'].tolist()
    
    def get_transcriptions(self, split: Optional[str] = None) -> List[str]:
        """Get list of transcriptions."""
        data = self.get_data(split)
        return data['transcript'].tolist()
    
    def get_targets(self, outcome: str, split: Optional[str] = None, 
                    encoded: bool = True) -> np.ndarray:
        """
        Get target values for a specific outcome.
        
        Args:
            outcome: Name of the outcome variable
            split: Optional split to filter
            encoded: Whether to return encoded labels for classification tasks
            
        Returns:
            Array of target values
        """
        if outcome not in self.OUTCOMES:
            raise ValueError(f"Unknown outcome: {outcome}")
        
        data = self.get_data(split)
        
        if encoded:
            if outcome == 'speaker':
                return data['speaker_encoded'].values
            elif outcome == 'emotion':
                return data['emotion_encoded'].values
            elif outcome == 'sentiment':
                return data['sentiment_encoded'].values
        else:
            if outcome == 'speaker':
                return data['speaker_processed'].values
            else:
                return data[outcome].values
    
    def get_sample_ids(self, split: Optional[str] = None) -> List[str]:
        """Get sample IDs."""
        data = self.get_data(split)
        # Create unique IDs from dialogue and utterance IDs
        sample_ids = [f"dia{row['dialogue_id']}_utt{row['utterance_id']}" 
                     for _, row in data.iterrows()]
        return sample_ids
    
    def get_stratification_labels(self, split: Optional[str] = None) -> np.ndarray:
        """
        Get labels for stratified splitting.
        Uses emotion as the stratification variable.
        """
        return self.get_targets('emotion', split=split, encoded=True)
    
    def get_outcome_info(self, outcome: str) -> Dict[str, Any]:
        """Get information about an outcome variable."""
        if outcome not in self.OUTCOMES:
            raise ValueError(f"Unknown outcome: {outcome}")
        
        info = {
            'name': outcome,
            'type': self.OUTCOMES[outcome],
        }
        
        if outcome == 'speaker':
            info['classes'] = self.speaker_encoder.classes_.tolist()
            info['n_classes'] = len(self.speaker_encoder.classes_)
        elif outcome == 'emotion':
            info['classes'] = self.emotion_encoder.classes_.tolist()
            info['n_classes'] = len(self.emotion_encoder.classes_)
        elif outcome == 'sentiment':
            info['classes'] = self.sentiment_encoder.classes_.tolist()
            info['n_classes'] = len(self.sentiment_encoder.classes_)
        
        return info
    
    def get_split_statistics(self) -> Dict[str, Dict[str, int]]:
        """Get statistics about the dataset splits."""
        stats = {}
        
        for split in self.df['split'].unique():
            split_data = self.df[self.df['split'] == split]
            stats[split] = {
                'n_samples': len(split_data),
                'n_speakers': split_data['speaker_processed'].nunique(),
                'n_emotions': split_data['emotion'].nunique(),
                'n_sentiments': split_data['sentiment'].nunique(),
                'n_dialogues': split_data['dialogue_id'].nunique(),
            }
        
        return stats
    
    def __len__(self) -> int:
        """Get dataset size."""
        return len(self.df)
    
    def __repr__(self) -> str:
        """String representation."""
        split_counts = self.df['split'].value_counts().to_dict()
        return (
            f"MELDDataset(n_samples={len(self)}, "
            f"splits={split_counts}, "
            f"n_speakers={self.df['speaker_processed'].nunique()})"
        )
