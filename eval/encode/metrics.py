#!/usr/bin/env python3
"""
Metrics Computation and Reporting
"""

import os
import json
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional
import numpy as np
import pandas as pd
from tabulate import tabulate

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MetricsReporter:
    """Generate and save evaluation metrics reports."""
    
    def __init__(self, results_dir: str = "eval/encode/results"):
        """
        Initialize metrics reporter.
        
        Args:
            results_dir: Directory to save results
        """
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)
        
        # Metric display names
        self.metric_names = {
            # Regression metrics
            'mse': 'MSE',
            'mae': 'MAE',
            'rmse': 'RMSE',
            'r2': 'R²',
            'pearson_r': 'Pearson r',
            'spearman_r': 'Spearman ρ',
            
            # Classification metrics
            'accuracy': 'Accuracy',
            'precision_macro': 'Precision (macro)',
            'recall_macro': 'Recall (macro)',
            'f1_macro': 'F1 (macro)',
            'precision_weighted': 'Precision (weighted)',
            'recall_weighted': 'Recall (weighted)',
            'f1_weighted': 'F1 (weighted)',
            'precision_micro': 'Precision (micro)',
            'recall_micro': 'Recall (micro)',
            'f1_micro': 'F1 (micro)',
            'roc_auc': 'ROC-AUC',
            'r2_macro': 'R² (macro)',
            'r2_weighted': 'R² (weighted)',
            'pearson_r_macro': 'Pearson r (macro)',
            'pearson_p_macro': 'Pearson p (macro)',
            'spearman_r_macro': 'Spearman ρ (macro)',
            'spearman_p_macro': 'Spearman p (macro)',
        }
    
    def generate_report(
        self,
        results: Dict[str, Dict[str, Any]],
        model_id: str,
        dataset_info: Dict[str, Any]
    ) -> str:
        """
        Generate a comprehensive evaluation report.
        
        Args:
            results: Dictionary of results for each outcome
            model_id: Model identifier
            dataset_info: Dataset information
            
        Returns:
            Formatted report string
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Build report sections
        dataset_name = dataset_info.get('dataset_name', 'IEMOCAP')
        report_lines = [
            "# Embedding Evaluation Report",
            "",
            f"**Generated:** {timestamp}",
            f"**Model:** {model_id}",
            f"**Dataset:** {dataset_name}",
            "",
            "---",
            "",
        ]
        
        # Add dataset statistics
        report_lines.extend(self._format_dataset_info(dataset_info))
        report_lines.extend(["", "---", ""])
        
        # Add results for each outcome
        report_lines.append("## Evaluation Results")
        report_lines.append("")
        
        # Separate regression and classification results
        regression_results = {}
        classification_results = {}
        
        for outcome, outcome_results in results.items():
            if outcome_results['task_type'] == 'regression':
                regression_results[outcome] = outcome_results
            else:
                classification_results[outcome] = outcome_results
        
        # Add regression results
        if regression_results:
            report_lines.append("### Regression Tasks")
            report_lines.append("")
            report_lines.extend(self._format_regression_results(regression_results))
            report_lines.append("")
        
        # Add classification results
        if classification_results:
            report_lines.append("### Classification Tasks")
            report_lines.append("")
            report_lines.extend(self._format_classification_results(classification_results))
            report_lines.append("")
        
        # Add summary statistics
        report_lines.extend(["---", ""])
        report_lines.append("## Summary Statistics")
        report_lines.append("")
        report_lines.extend(self._format_summary_statistics(results))
        
        return "\n".join(report_lines)
    
    def _format_dataset_info(self, dataset_info: Dict[str, Any]) -> List[str]:
        """Format dataset information section."""
        lines = [
            "## Dataset Information",
            "",
            f"- **Total Samples:** {dataset_info.get('n_samples', 'N/A')}",
            f"- **Train Samples:** {dataset_info.get('n_train', 'N/A')}",
            f"- **Test Samples:** {dataset_info.get('n_test', 'N/A')}",
            f"- **Embedding Dimension:** {dataset_info.get('embedding_dim', 'N/A')}",
        ]
        
        if 'splits' in dataset_info:
            lines.append("")
            lines.append("**Split Distribution:**")
            for split, count in dataset_info['splits'].items():
                lines.append(f"  - {split}: {count}")
        
        # Add MELD-specific statistics if available
        if 'split_statistics' in dataset_info:
            lines.append("")
            lines.append("**Detailed Split Statistics:**")
            for split, stats in dataset_info['split_statistics'].items():
                lines.append(f"  - {split}:")
                for stat_name, stat_value in stats.items():
                    lines.append(f"    - {stat_name}: {stat_value}")
        
        return lines
    
    def _format_regression_results(self, results: Dict[str, Dict]) -> List[str]:
        """Format regression results as a table."""
        lines = []
        
        # Prepare data for cross-validation results table
        cv_data = []
        for outcome, outcome_results in results.items():
            cv_metrics = outcome_results['cv_results']
            row = [outcome]
            
            # Add key metrics with mean ± std
            for metric in ['rmse', 'mae', 'r2', 'pearson_r', 'spearman_r']:
                mean_key = f"{metric}_mean"
                std_key = f"{metric}_std"
                
                if mean_key in cv_metrics and std_key in cv_metrics:
                    mean_val = cv_metrics[mean_key]
                    std_val = cv_metrics[std_key]
                    
                    # Format based on metric type
                    if metric in ['r2', 'pearson_r', 'spearman_r']:
                        row.append(f"{mean_val:.3f} ± {std_val:.3f}")
                    else:
                        row.append(f"{mean_val:.4f} ± {std_val:.4f}")
                else:
                    row.append("N/A")
            
            cv_data.append(row)
        
        # Create cross-validation table
        cv_headers = ['Outcome', 'RMSE', 'MAE', 'R²', 'Pearson r', 'Spearman ρ']
        cv_table = tabulate(cv_data, headers=cv_headers, tablefmt='pipe')
        
        lines.append("#### Cross-Validation Results (mean ± std across folds)")
        lines.append("")
        lines.append(cv_table)
        lines.append("")
        
        # Prepare data for test results table
        test_data = []
        for outcome, outcome_results in results.items():
            test_metrics = outcome_results['test_results']
            row = [outcome]
            
            # Add key metrics
            for metric in ['rmse', 'mae', 'r2', 'pearson_r', 'spearman_r']:
                if metric in test_metrics:
                    val = test_metrics[metric]
                    # Format based on metric type
                    if metric in ['r2', 'pearson_r', 'spearman_r']:
                        row.append(f"{val:.3f}")
                    else:
                        row.append(f"{val:.4f}")
                else:
                    row.append("N/A")
            
            test_data.append(row)
        
        # Create test results table
        test_headers = ['Outcome', 'RMSE', 'MAE', 'R²', 'Pearson r', 'Spearman ρ']
        test_table = tabulate(test_data, headers=test_headers, tablefmt='pipe')
        
        lines.append("#### Test Set Results")
        lines.append("")
        lines.append(test_table)
        
        return lines
    
    def _format_classification_results(self, results: Dict[str, Dict]) -> List[str]:
        """Format classification results as a table."""
        lines = []
        
        # Prepare data for cross-validation results table
        cv_data = []
        for outcome, outcome_results in results.items():
            cv_metrics = outcome_results['cv_results']
            row = [outcome]
            
            # Add key metrics with mean ± std
            for metric in ['accuracy', 'f1_macro', 'f1_weighted', 'roc_auc']:
                mean_key = f"{metric}_mean"
                std_key = f"{metric}_std"
                
                if mean_key in cv_metrics and std_key in cv_metrics:
                    mean_val = cv_metrics[mean_key]
                    std_val = cv_metrics[std_key]
                    row.append(f"{mean_val:.3f} ± {std_val:.3f}")
                else:
                    row.append("N/A")
            
            cv_data.append(row)
        
        # Create cross-validation table
        cv_headers = ['Outcome', 'Accuracy', 'F1 (macro)', 'F1 (weighted)', 'ROC-AUC']
        cv_table = tabulate(cv_data, headers=cv_headers, tablefmt='pipe')
        
        lines.append("#### Cross-Validation Results (mean ± std across folds)")
        lines.append("")
        lines.append(cv_table)
        lines.append("")
        
        # Prepare data for test results table
        test_data = []
        for outcome, outcome_results in results.items():
            test_metrics = outcome_results['test_results']
            row = [outcome]
            
            # Add key metrics
            for metric in ['accuracy', 'f1_macro', 'f1_weighted', 'roc_auc']:
                if metric in test_metrics and test_metrics[metric] is not None:
                    val = test_metrics[metric]
                    row.append(f"{val:.3f}")
                else:
                    row.append("N/A")
            
            test_data.append(row)
        
        # Create test results table
        test_headers = ['Outcome', 'Accuracy', 'F1 (macro)', 'F1 (weighted)', 'ROC-AUC']
        test_table = tabulate(test_data, headers=test_headers, tablefmt='pipe')
        
        lines.append("#### Test Set Results")
        lines.append("")
        lines.append(test_table)
        
        # Additional correlation metrics table (macro/weighted)
        corr_data = []
        for outcome, outcome_results in results.items():
            test_metrics = outcome_results['test_results']
            row = [outcome]
            for metric in ['r2_macro', 'r2_weighted', 'pearson_r_macro', 'spearman_r_macro']:
                if metric in test_metrics and test_metrics[metric] is not None:
                    val = test_metrics[metric]
                    # r2 may be negative; show with 3 decimals
                    row.append(f"{val:.3f}")
                else:
                    row.append("N/A")
            corr_data.append(row)
        if corr_data:
            corr_headers = ['Outcome', 'R² (macro)', 'R² (weighted)', 'Pearson r (macro)', 'Spearman ρ (macro)']
            corr_table = tabulate(corr_data, headers=corr_headers, tablefmt='pipe')
            lines.append("")
            lines.append("#### Test Set Correlation Metrics (classification)")
            lines.append("")
            lines.append(corr_table)
        
        # Add confusion matrices for each outcome
        lines.append("")
        lines.append("#### Confusion Matrices (Test Set)")
        lines.append("")
        
        for outcome, outcome_results in results.items():
            test_metrics = outcome_results['test_results']
            if 'confusion_matrix' in test_metrics:
                lines.append(f"**{outcome}:**")
                lines.append("```")
                cm = np.array(test_metrics['confusion_matrix'])
                lines.append(str(cm))
                lines.append("```")
                lines.append("")
        
        return lines
    
    def _format_summary_statistics(self, results: Dict[str, Dict]) -> List[str]:
        """Format summary statistics across all tasks."""
        lines = []
        
        # Calculate average metrics
        regression_metrics = {
            'r2': [],
            'pearson_r': [],
            'spearman_r': [],
        }
        
        classification_metrics = {
            'accuracy': [],
            'f1_macro': [],
            'roc_auc': [],
            'r2_macro': [],
            'r2_weighted': [],
            'pearson_r_macro': [],
            'spearman_r_macro': [],
        }
        
        for outcome, outcome_results in results.items():
            test_metrics = outcome_results['test_results']
            
            if outcome_results['task_type'] == 'regression':
                for metric in regression_metrics:
                    if metric in test_metrics:
                        regression_metrics[metric].append(test_metrics[metric])
            else:
                for metric in classification_metrics:
                    if metric in test_metrics and test_metrics[metric] is not None:
                        classification_metrics[metric].append(test_metrics[metric])
        
        # Format summary
        if any(regression_metrics.values()):
            lines.append("### Regression Tasks Summary")
            lines.append("")
            for metric, values in regression_metrics.items():
                if values:
                    mean_val = np.mean(values)
                    std_val = np.std(values)
                    metric_name = self.metric_names.get(metric, metric)
                    lines.append(f"- **Average {metric_name}:** {mean_val:.3f} ± {std_val:.3f}")
            lines.append("")
        
        if any(classification_metrics.values()):
            lines.append("### Classification Tasks Summary")
            lines.append("")
            for metric, values in classification_metrics.items():
                if values:
                    mean_val = np.mean(values)
                    std_val = np.std(values)
                    metric_name = self.metric_names.get(metric, metric)
                    lines.append(f"- **Average {metric_name}:** {mean_val:.3f} ± {std_val:.3f}")
        
        return lines
    
    def save_report(
        self,
        results: Dict[str, Dict[str, Any]],
        model_id: str,
        dataset_info: Dict[str, Any]
    ) -> str:
        """
        Save evaluation report to file.
        
        Args:
            results: Dictionary of results for each outcome
            model_id: Model identifier
            dataset_info: Dataset information
            
        Returns:
            Path to saved report directory
        """
        # Generate report
        report = self.generate_report(results, model_id, dataset_info)
        
        # Create timestamp directory
        timestamp = datetime.now().strftime("%y-%m-%d_%H-%M-%S-%f")[:-3]
        timestamp_dir = os.path.join(self.results_dir, timestamp)
        os.makedirs(timestamp_dir, exist_ok=True)
        
        # Save markdown summary
        summary_filepath = os.path.join(timestamp_dir, "summary.md")
        with open(summary_filepath, 'w') as f:
            f.write(report)
        
        logger.info(f"Summary saved to: {summary_filepath}")
        
        # Save raw results as JSON
        json_filepath = os.path.join(timestamp_dir, "results.json")
        
        json_data = {
            'timestamp': timestamp,
            'model_id': model_id,
            'dataset_info': dataset_info,
            'results': results,
        }
        
        with open(json_filepath, 'w') as f:
            json.dump(json_data, f, indent=2, default=str)
        
        logger.info(f"Raw results saved to: {json_filepath}")
        
        return timestamp_dir
    
    def print_summary(self, results: Dict[str, Dict[str, Any]]):
        """Print a brief summary of results to console."""
        print("\n" + "=" * 60)
        print("EVALUATION SUMMARY")
        print("=" * 60)
        
        # Regression tasks
        regression_outcomes = [k for k, v in results.items() if v['task_type'] == 'regression']
        if regression_outcomes:
            print("\nRegression Tasks:")
            for outcome in regression_outcomes:
                test_r2 = results[outcome]['test_results'].get('r2', None)
                test_pearson = results[outcome]['test_results'].get('pearson_r', None)
                
                if test_r2 is not None and test_pearson is not None:
                    print(f"  {outcome:15s} - R²: {test_r2:.3f}, Pearson r: {test_pearson:.3f}")
        
        # Classification tasks
        classification_outcomes = [k for k, v in results.items() if v['task_type'] == 'classification']
        if classification_outcomes:
            print("\nClassification Tasks:")
            for outcome in classification_outcomes:
                test_acc = results[outcome]['test_results'].get('accuracy', None)
                test_f1 = results[outcome]['test_results'].get('f1_macro', None)
                
                if test_acc is not None and test_f1 is not None:
                    print(f"  {outcome:15s} - Accuracy: {test_acc:.3f}, F1 (macro): {test_f1:.3f}")
        
        print("=" * 60 + "\n")
    
    def save_combined_report(
        self,
        all_results: Dict[str, Dict[str, Dict[str, Any]]],
        model_id: str,
        combined_dataset_info: Dict[str, Any]
    ) -> str:
        """
        Save evaluation report for multiple datasets.
        
        Args:
            all_results: Dictionary of results for each dataset
            model_id: Model identifier
            combined_dataset_info: Combined dataset information
            
        Returns:
            Path to saved report directory
        """
        # Create timestamp directory
        timestamp = datetime.now().strftime("%y-%m-%d_%H-%M-%S-%f")[:-3]
        timestamp_dir = os.path.join(self.results_dir, timestamp)
        os.makedirs(timestamp_dir, exist_ok=True)
        
        # Generate combined report
        timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Build report sections
        report_lines = [
            "# Embedding Evaluation Report - Multiple Datasets",
            "",
            f"**Generated:** {timestamp_str}",
            f"**Model:** {model_id}",
            f"**Datasets:** {combined_dataset_info.get('dataset_name', 'Multiple')}",
            "",
            "---",
            "",
        ]
        
        # Add summary for each dataset
        for dataset_name, dataset_info in combined_dataset_info.get('datasets', {}).items():
            report_lines.append(f"## {dataset_name.upper()} Dataset")
            report_lines.append("")
            report_lines.extend(self._format_dataset_info(dataset_info))
            report_lines.append("")
            
            # Add results for this dataset
            if dataset_name in all_results:
                results = all_results[dataset_name]
                
                # Separate regression and classification results
                regression_results = {}
                classification_results = {}
                
                for outcome, outcome_results in results.items():
                    if outcome_results['task_type'] == 'regression':
                        regression_results[outcome] = outcome_results
                    else:
                        classification_results[outcome] = outcome_results
                
                # Add regression results
                if regression_results:
                    report_lines.append(f"### {dataset_name.upper()} - Regression Tasks")
                    report_lines.append("")
                    report_lines.extend(self._format_regression_results(regression_results))
                    report_lines.append("")
                
                # Add classification results
                if classification_results:
                    report_lines.append(f"### {dataset_name.upper()} - Classification Tasks")
                    report_lines.append("")
                    report_lines.extend(self._format_classification_results(classification_results))
                    report_lines.append("")
            
            report_lines.extend(["---", ""])
        
        # Add combined summary statistics
        report_lines.append("## Combined Summary Statistics")
        report_lines.append("")
        
        all_regression_metrics = {
            'r2': [],
            'pearson_r': [],
            'spearman_r': [],
        }
        
        all_classification_metrics = {
            'accuracy': [],
            'f1_macro': [],
            'roc_auc': [],
        }
        
        # Collect metrics from all datasets
        for dataset_name, results in all_results.items():
            for outcome, outcome_results in results.items():
                test_metrics = outcome_results['test_results']
                
                if outcome_results['task_type'] == 'regression':
                    for metric in all_regression_metrics:
                        if metric in test_metrics:
                            all_regression_metrics[metric].append(test_metrics[metric])
                else:
                    for metric in all_classification_metrics:
                        if metric in test_metrics and test_metrics[metric] is not None:
                            all_classification_metrics[metric].append(test_metrics[metric])
        
        # Format combined summary
        if any(all_regression_metrics.values()):
            report_lines.append("### All Regression Tasks")
            report_lines.append("")
            for metric, values in all_regression_metrics.items():
                if values:
                    mean_val = np.mean(values)
                    std_val = np.std(values)
                    metric_name = self.metric_names.get(metric, metric)
                    report_lines.append(f"- **Average {metric_name}:** {mean_val:.3f} ± {std_val:.3f}")
            report_lines.append("")
        
        if any(all_classification_metrics.values()):
            report_lines.append("### All Classification Tasks")
            report_lines.append("")
            for metric, values in all_classification_metrics.items():
                if values:
                    mean_val = np.mean(values)
                    std_val = np.std(values)
                    metric_name = self.metric_names.get(metric, metric)
                    report_lines.append(f"- **Average {metric_name}:** {mean_val:.3f} ± {std_val:.3f}")
        
        # Save markdown summary
        summary_filepath = os.path.join(timestamp_dir, "summary.md")
        with open(summary_filepath, 'w') as f:
            f.write("\n".join(report_lines))
        
        logger.info(f"Summary saved to: {summary_filepath}")
        
        # Save raw results as JSON
        json_filepath = os.path.join(timestamp_dir, "results.json")
        
        json_data = {
            'timestamp': timestamp,
            'model_id': model_id,
            'dataset_info': combined_dataset_info,
            'results': all_results,
        }
        
        with open(json_filepath, 'w') as f:
            json.dump(json_data, f, indent=2, default=str)
        
        logger.info(f"Raw results saved to: {json_filepath}")
        
        return timestamp_dir
