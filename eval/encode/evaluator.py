#!/usr/bin/env python3
"""
Ridge Regression Evaluator with K-Fold Cross-Validation
"""

import logging
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold, KFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, precision_recall_fscore_support, roc_auc_score,
    confusion_matrix, classification_report
)
from scipy.stats import pearsonr, spearmanr
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RidgeEvaluator:
    """Ridge regression evaluator with GPU acceleration."""
    
    def __init__(
        self,
        n_folds: int = 10,
        test_size: float = 0.2,
        alpha: float = 1.0,
        max_iter: int = 10000,
        tol: float = 1e-4,
        device: str = 'cuda',
        random_state: int = 42
    ):
        """
        Initialize Ridge evaluator.
        
        Args:
            n_folds: Number of cross-validation folds
            test_size: Test set size for train/test split
            alpha: Ridge regularization parameter
            max_iter: Maximum iterations for solver
            tol: Tolerance for convergence
            device: Device for computation
            random_state: Random seed for reproducibility
        """
        self.n_folds = n_folds
        self.test_size = test_size
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.random_state = random_state
        
        # Set random seeds for reproducibility
        np.random.seed(random_state)
        torch.manual_seed(random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(random_state)
    
    def evaluate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        task_type: str,
        stratify_labels: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Evaluate embeddings on a specific outcome.
        
        Args:
            X: Embeddings (n_samples, embedding_dim)
            y: Target values
            task_type: 'regression' or 'classification'
            stratify_labels: Labels for stratified splitting
            
        Returns:
            Dictionary containing evaluation metrics
        """
        logger.info(f"Evaluating {task_type} task with {X.shape[0]} samples")
        
        # Train/test split
        if stratify_labels is not None and task_type == 'classification':
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=self.test_size,
                stratify=stratify_labels, random_state=self.random_state
            )
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=self.test_size, random_state=self.random_state
            )
        
        # Standardize features
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
        # Run k-fold cross-validation on training set
        if task_type == 'regression':
            cv_results = self._evaluate_regression_cv(X_train, y_train)
        else:
            cv_results = self._evaluate_classification_cv(X_train, y_train)
        
        # Train final model on full training set
        logger.info("Training final model on full training set")
        if task_type == 'regression':
            model = self._train_ridge_regression(X_train, y_train)
            test_results = self._evaluate_regression(model, X_test, y_test)
        else:
            model = self._train_ridge_classification(X_train, y_train)
            test_results = self._evaluate_classification(model, X_test, y_test)
        
        # Combine results
        results = {
            'task_type': task_type,
            'n_samples': X.shape[0],
            'n_features': X.shape[1],
            'cv_results': cv_results,
            'test_results': test_results,
        }
        
        return results
    
    def _evaluate_regression_cv(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Run k-fold CV for regression."""
        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)
        
        fold_metrics = []
        
        for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X)):
            X_fold_train, X_fold_val = X[train_idx], X[val_idx]
            y_fold_train, y_fold_val = y[train_idx], y[val_idx]
            
            # Train model
            model = self._train_ridge_regression(X_fold_train, y_fold_train)
            
            # Evaluate
            metrics = self._evaluate_regression(model, X_fold_val, y_fold_val)
            fold_metrics.append(metrics)
        
        # Aggregate metrics
        aggregated = self._aggregate_metrics(fold_metrics)
        return aggregated
    
    def _evaluate_classification_cv(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Run stratified k-fold CV for classification."""
        skf = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)
        
        fold_metrics = []
        
        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            X_fold_train, X_fold_val = X[train_idx], X[val_idx]
            y_fold_train, y_fold_val = y[train_idx], y[val_idx]
            
            # Train model
            model = self._train_ridge_classification(X_fold_train, y_fold_train)
            
            # Evaluate
            metrics = self._evaluate_classification(model, X_fold_val, y_fold_val)
            fold_metrics.append(metrics)
        
        # Aggregate metrics
        aggregated = self._aggregate_metrics(fold_metrics)
        return aggregated
    
    def _train_ridge_regression(self, X: np.ndarray, y: np.ndarray) -> 'RidgeRegressionGPU':
        """Train Ridge regression model on GPU."""
        model = RidgeRegressionGPU(
            alpha=self.alpha,
            max_iter=self.max_iter,
            tol=self.tol,
            device=self.device
        )
        model.fit(X, y)
        return model
    
    def _train_ridge_classification(self, X: np.ndarray, y: np.ndarray) -> 'RidgeClassificationGPU':
        """Train Ridge classification model on GPU."""
        model = RidgeClassificationGPU(
            alpha=self.alpha,
            max_iter=self.max_iter,
            tol=self.tol,
            device=self.device
        )
        model.fit(X, y)
        return model
    
    def _evaluate_regression(self, model, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Evaluate regression model."""
        y_pred = model.predict(X)
        
        # Ensure arrays are numpy
        if isinstance(y_pred, torch.Tensor):
            y_pred = y_pred.cpu().numpy()
        
        metrics = {
            'mse': mean_squared_error(y, y_pred),
            'mae': mean_absolute_error(y, y_pred),
            'rmse': np.sqrt(mean_squared_error(y, y_pred)),
            'r2': r2_score(y, y_pred),
        }
        
        # Correlation metrics
        if len(y) > 1:
            pearson_corr, pearson_p = pearsonr(y, y_pred)
            spearman_corr, spearman_p = spearmanr(y, y_pred)
            metrics['pearson_r'] = pearson_corr
            metrics['pearson_p'] = pearson_p
            metrics['spearman_r'] = spearman_corr
            metrics['spearman_p'] = spearman_p
        
        return metrics
    
    def _evaluate_classification(self, model, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Evaluate classification model."""
        y_pred = model.predict(X)
        y_proba = model.predict_proba(X)
        
        # Ensure arrays are numpy
        if isinstance(y_pred, torch.Tensor):
            y_pred = y_pred.cpu().numpy()
        if isinstance(y_proba, torch.Tensor):
            y_proba = y_proba.cpu().numpy()
        
        # Basic metrics
        accuracy = accuracy_score(y, y_pred)
        precision, recall, f1, support = precision_recall_fscore_support(
            y, y_pred, average=None, zero_division=0
        )
        
        metrics = {
            'accuracy': accuracy,
            'precision_macro': np.mean(precision),
            'recall_macro': np.mean(recall),
            'f1_macro': np.mean(f1),
        }
        
        # Weighted metrics
        precision_w, recall_w, f1_w, _ = precision_recall_fscore_support(
            y, y_pred, average='weighted', zero_division=0
        )
        metrics['precision_weighted'] = precision_w
        metrics['recall_weighted'] = recall_w
        metrics['f1_weighted'] = f1_w
        
        # Micro metrics
        precision_m, recall_m, f1_m, _ = precision_recall_fscore_support(
            y, y_pred, average='micro', zero_division=0
        )
        metrics['precision_micro'] = precision_m
        metrics['recall_micro'] = recall_m
        metrics['f1_micro'] = f1_m
        
        # ROC-AUC
        try:
            n_classes = y_proba.shape[1]
            if n_classes == 2:
                # Binary classification
                metrics['roc_auc'] = roc_auc_score(y, y_proba[:, 1])
            else:
                # Multiclass (OVR)
                metrics['roc_auc'] = roc_auc_score(
                    y, y_proba, multi_class='ovr', average='macro'
                )
        except Exception as e:
            logger.warning(f"Could not compute ROC-AUC: {e}")
            metrics['roc_auc'] = None
        
        # Confusion matrix
        metrics['confusion_matrix'] = confusion_matrix(y, y_pred).tolist()
        
        # Compute correlations between one-vs-rest targets and predicted probabilities per class
        try:
            classes = getattr(model, 'classes', np.unique(y))
            per_class_r2 = []
            per_class_pearson_r = []
            per_class_pearson_p = []
            per_class_spearman_r = []
            per_class_spearman_p = []
            class_supports = []
            
            for idx, class_label in enumerate(classes):
                y_bin = (y == class_label).astype(float)
                class_support = y_bin.sum()
                class_supports.append(class_support)
                p = y_proba[:, idx]
                
                # Skip if y is constant for this class (no positives in split)
                if np.all(y_bin == y_bin[0]):
                    per_class_r2.append(None)
                    per_class_pearson_r.append(None)
                    per_class_pearson_p.append(None)
                    per_class_spearman_r.append(None)
                    per_class_spearman_p.append(None)
                    continue
                
                # R^2 between binary labels and predicted probability
                try:
                    r2_val = r2_score(y_bin, p)
                except Exception:
                    r2_val = None
                
                # Pearson correlation
                try:
                    pr_val, pr_p = pearsonr(y_bin, p)
                except Exception:
                    pr_val, pr_p = None, None
                
                # Spearman correlation
                try:
                    sr_val, sr_p = spearmanr(y_bin, p)
                except Exception:
                    sr_val, sr_p = None, None
                
                per_class_r2.append(r2_val)
                per_class_pearson_r.append(pr_val)
                per_class_pearson_p.append(pr_p)
                per_class_spearman_r.append(sr_val)
                per_class_spearman_p.append(sr_p)
            
            # Aggregate (macro and weighted by class support)
            def _nanmean(values):
                vals = [v for v in values if v is not None and not np.isnan(v)]
                return float(np.mean(vals)) if len(vals) > 0 else None
            
            def _nanweighted_mean(values, weights):
                pairs = [(v, w) for v, w in zip(values, weights) if v is not None and not np.isnan(v) and w > 0]
                if not pairs:
                    return None
                v_arr = np.array([v for v, _ in pairs], dtype=np.float64)
                w_arr = np.array([w for _, w in pairs], dtype=np.float64)
                w_sum = w_arr.sum()
                if w_sum <= 0:
                    return None
                return float(np.average(v_arr, weights=w_arr))
            
            metrics['r2_macro'] = _nanmean(per_class_r2)
            metrics['r2_weighted'] = _nanweighted_mean(per_class_r2, class_supports)
            metrics['pearson_r_macro'] = _nanmean(per_class_pearson_r)
            metrics['pearson_p_macro'] = _nanmean(per_class_pearson_p)
            metrics['spearman_r_macro'] = _nanmean(per_class_spearman_r)
            metrics['spearman_p_macro'] = _nanmean(per_class_spearman_p)
        except Exception as e:
            logger.warning(f"Could not compute correlation metrics for classification: {e}")
        
        return metrics
    
    def _aggregate_metrics(self, fold_metrics: List[Dict]) -> Dict[str, Any]:
        """Aggregate metrics across folds."""
        aggregated = {}
        
        # Get all metric keys
        all_keys = set()
        for metrics in fold_metrics:
            all_keys.update(metrics.keys())
        
        for key in all_keys:
            values = []
            for metrics in fold_metrics:
                if key in metrics and metrics[key] is not None:
                    if key == 'confusion_matrix':
                        # Special handling for confusion matrix
                        continue
                    values.append(metrics[key])
            
            if values:
                aggregated[f"{key}_mean"] = np.mean(values)
                aggregated[f"{key}_std"] = np.std(values)
        
        return aggregated


class RidgeRegressionGPU:
    """GPU-accelerated Ridge regression using PyTorch."""
    
    def __init__(self, alpha: float = 1.0, max_iter: int = 10000,
                 tol: float = 1e-4, device: str = 'cuda'):
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol
        self.device = device
        self.weights = None
        self.bias = None
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit Ridge regression using closed-form solution on GPU."""
        # Convert to tensors
        X_tensor = torch.tensor(X, dtype=torch.float32, device=self.device)
        y_tensor = torch.tensor(y, dtype=torch.float32, device=self.device)
        
        # Add bias term
        n_samples = X_tensor.shape[0]
        X_with_bias = torch.cat([
            torch.ones(n_samples, 1, device=self.device),
            X_tensor
        ], dim=1)
        
        # Closed-form solution: w = (X^T X + alpha*I)^{-1} X^T y
        XtX = X_with_bias.T @ X_with_bias
        
        # Add regularization (skip bias term)
        reg_matrix = self.alpha * torch.eye(XtX.shape[0], device=self.device)
        reg_matrix[0, 0] = 0  # Don't regularize bias
        
        # Solve
        try:
            weights = torch.linalg.solve(XtX + reg_matrix, X_with_bias.T @ y_tensor)
        except:
            # Use pseudo-inverse if singular
            weights = torch.linalg.pinv(XtX + reg_matrix) @ (X_with_bias.T @ y_tensor)
        
        self.bias = weights[0]
        self.weights = weights[1:]
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        X_tensor = torch.tensor(X, dtype=torch.float32, device=self.device)
        y_pred = X_tensor @ self.weights + self.bias
        return y_pred.cpu().float().numpy()


class RidgeClassificationGPU:
    """GPU-accelerated Ridge classification using one-vs-rest."""
    
    def __init__(self, alpha: float = 1.0, max_iter: int = 10000,
                 tol: float = 1e-4, device: str = 'cuda'):
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol
        self.device = device
        self.classes = None
        self.classifiers = []
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit Ridge classifier using one-vs-rest strategy."""
        self.classes = np.unique(y)
        n_classes = len(self.classes)
        
        # Train binary classifier for each class
        for class_idx, class_label in enumerate(self.classes):
            # Create binary labels
            y_binary = (y == class_label).astype(float)
            
            # Train Ridge regression for this class
            clf = RidgeRegressionGPU(
                alpha=self.alpha,
                max_iter=self.max_iter,
                tol=self.tol,
                device=self.device
            )
            clf.fit(X, y_binary)
            self.classifiers.append(clf)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        # Get predictions from all classifiers
        predictions = []
        for clf in self.classifiers:
            pred = clf.predict(X)
            predictions.append(pred)
        
        predictions = np.column_stack(predictions)
        
        # Convert to probabilities using softmax
        exp_pred = np.exp(predictions - np.max(predictions, axis=1, keepdims=True))
        probas = exp_pred / np.sum(exp_pred, axis=1, keepdims=True)
        
        return probas
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        probas = self.predict_proba(X)
        class_indices = np.argmax(probas, axis=1)
        return self.classes[class_indices]
