#!/usr/bin/env python3
"""
IEMOCAP Dataset Loader for Embedding Evaluation
"""

import os
import logging
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import pandas as pd
from datasets import load_from_disk
from sklearn.preprocessing import LabelEncoder
from dotenv import load_dotenv

load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class IEMOCAPDataset:
    """IEMOCAP dataset loader for embedding evaluation."""
    
    # Define outcome types and their properties
    OUTCOMES = {
        # Regression tasks (emotion intensities)
        'frustrated': 'regression',
        'angry': 'regression',
        'sad': 'regression',
        'disgust': 'regression',
        'excited': 'regression',
        'fear': 'regression',
        'neutral': 'regression',
        'surprise': 'regression',
        'happy': 'regression',
        'EmoAct': 'regression',  # Emotional activation
        'EmoVal': 'regression',  # Emotional valence
        'EmoDom': 'regression',  # Emotional dominance
        'speaking_rate': 'regression',
        
        # Classification tasks
        'gender': 'classification',
        'major_emotion': 'classification',
    }
    
    # Major emotion categories for classification
    MAJOR_EMOTIONS = [
        'frustrated', 'angry', 'sad', 'disgust', 'excited',
        'fear', 'neutral', 'surprise', 'happy'
    ]
    
    def __init__(self):
        """
        Initialize IEMOCAP dataset loader.
        
        Args:
            dataset_path: Path to IEMOCAP dataset. If None, uses environment variable.
        """
        dataset_path = os.getenv("IEMOCAP_DIR", None)
        
        if dataset_path is None or not os.path.exists(dataset_path):
            raise ValueError(f"Dataset path does not exist: {dataset_path}")
        
        logger.info(f"Loading IEMOCAP dataset from: {dataset_path}")
        self.dataset = load_from_disk(dataset_path)
        self.data = []
        self._process_dataset()
        
    def get_name(self) -> str:
        """Get the name of this dataset."""
        return "iemocap"
        
    def _process_dataset(self):
        """Process the dataset and extract relevant features."""
        # IEMOCAP validation dataset only has 'dev' split
        for split in ['dev']:
            if split not in self.dataset:
                logger.warning(f"Split '{split}' not found in dataset")
                continue
                
            for idx, item in enumerate(self.dataset[split]):
                # Extract audio path - in IEMOCAP it's stored as a string in 'audio' field
                audio_path = item.get('audio', item.get('file', ''))
                if not audio_path:
                    logger.warning(f"No audio path found for item at index {idx}")
                    continue
                
                # Build sample dictionary
                sample = {
                    'split': split,
                    'audio_path': audio_path,
                    'transcription': item.get('transcription', ''),
                }
                
                # Extract emotion labels (regression targets)
                for emotion in self.MAJOR_EMOTIONS:
                    value = item.get(emotion, 0.0)
                    sample[emotion] = float(value) if value is not None else 0.0
                
                # Extract emotional dimensions
                sample['EmoAct'] = float(item.get('EmoAct', 0.0) or 0.0)
                sample['EmoVal'] = float(item.get('EmoVal', 0.0) or 0.0)
                sample['EmoDom'] = float(item.get('EmoDom', 0.0) or 0.0)
                
                # Extract gender (classification target)
                sample['gender'] = item.get('gender', 'Unknown')
                
                # Determine major emotion (classification target) - use the one provided or calculate
                if 'major_emotion' in item:
                    sample['major_emotion'] = item['major_emotion']
                else:
                    emotion_scores = {
                        emotion: float(item.get(emotion, 0.0) or 0.0)
                        for emotion in self.MAJOR_EMOTIONS
                    }
                    sample['major_emotion'] = max(emotion_scores, key=emotion_scores.get) if emotion_scores else 'neutral'
                
                # Extract speaking rate if available
                sample['speaking_rate'] = float(item.get('speaking_rate', 0.0) or 0.0)
                
                # Add sample ID
                sample['sample_id'] = item.get('id', f"{split}_{idx}")
                
                self.data.append(sample)
        
        logger.info(f"Processed {len(self.data)} samples from IEMOCAP dataset")
        
        # Convert to DataFrame for easier manipulation
        self.df = pd.DataFrame(self.data)
        
        # Encode categorical variables
        self._encode_labels()
        
    def _encode_labels(self):
        """Encode categorical labels for classification tasks."""
        # Encode gender
        self.gender_encoder = LabelEncoder()
        valid_gender = self.df['gender'].isin(['Male', 'Female'])
        self.df.loc[~valid_gender, 'gender'] = 'Unknown'
        self.df['gender_encoded'] = self.gender_encoder.fit_transform(self.df['gender'])
        
        # Encode major emotion
        self.emotion_encoder = LabelEncoder()
        self.df['major_emotion_encoded'] = self.emotion_encoder.fit_transform(self.df['major_emotion'])
        
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
        return data['audio_path'].tolist()
    
    def get_transcriptions(self, split: Optional[str] = None) -> List[str]:
        """Get list of transcriptions."""
        data = self.get_data(split)
        return data['transcription'].tolist()
    
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
        
        if self.OUTCOMES[outcome] == 'classification' and encoded:
            if outcome == 'gender':
                return data['gender_encoded'].values
            elif outcome == 'major_emotion':
                return data['major_emotion_encoded'].values
        
        return data[outcome].values
    
    def get_sample_ids(self, split: Optional[str] = None) -> List[str]:
        """Get sample IDs."""
        data = self.get_data(split)
        return data['sample_id'].tolist()
    
    def get_stratification_labels(self, split: Optional[str] = None) -> np.ndarray:
        """
        Get labels for stratified splitting.
        Uses major emotion as the stratification variable.
        """
        return self.get_targets('major_emotion', split=split, encoded=True)
    
    def get_outcome_info(self, outcome: str) -> Dict[str, Any]:
        """Get information about an outcome variable."""
        if outcome not in self.OUTCOMES:
            raise ValueError(f"Unknown outcome: {outcome}")
        
        info = {
            'name': outcome,
            'type': self.OUTCOMES[outcome],
        }
        
        if self.OUTCOMES[outcome] == 'classification':
            if outcome == 'gender':
                info['classes'] = self.gender_encoder.classes_.tolist()
                info['n_classes'] = len(self.gender_encoder.classes_)
            elif outcome == 'major_emotion':
                info['classes'] = self.emotion_encoder.classes_.tolist()
                info['n_classes'] = len(self.emotion_encoder.classes_)
        
        return info
    
    def __len__(self) -> int:
        """Get dataset size."""
        return len(self.df)
    
    def __repr__(self) -> str:
        """String representation."""
        split_counts = self.df['split'].value_counts().to_dict()
        return (
            f"IEMOCAPDataset(n_samples={len(self)}, "
            f"splits={split_counts})"
        )