"""
Embedding Evaluation Toolkit for IEMOCAP Dataset

This module provides tools for evaluating audio and text embedding models
on the IEMOCAP dataset across various emotion and speech-related outcomes.
"""

from .embeddings import EmbeddingExtractor
from .evaluator import RidgeEvaluator
from .metrics import MetricsReporter

__all__ = [
    'EmbeddingExtractor',
    'RidgeEvaluator',
    'MetricsReporter',
]
