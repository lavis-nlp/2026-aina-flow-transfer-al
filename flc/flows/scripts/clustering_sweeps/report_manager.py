import os
import csv
import json
from typing import Dict, List, Any, Set, Optional
import pandas as pd
from datetime import datetime

from .sweep_generator import SweepCombination


class ReportManager:
    """Manages CSV report for clustering sweep results"""
    
    def __init__(self, report_path: str):
        """
        Initialize report manager.
        
        Args:
            report_path: Path to CSV report file
        """
        self.report_path = report_path
        self._ensure_report_directory()
    
    def _ensure_report_directory(self) -> None:
        """Ensure the directory for the report file exists"""
        report_dir = os.path.dirname(self.report_path)
        if report_dir and not os.path.exists(report_dir):
            os.makedirs(report_dir)
    
    def get_completed_sweep_ids(self) -> Set[str]:
        """
        Get set of sweep IDs that have already been completed.
        
        Returns:
            Set of completed sweep IDs
        """
        if not os.path.exists(self.report_path):
            return set()
        
        try:
            df = pd.read_csv(self.report_path)
            if 'sweep_id' in df.columns:
                return set(df['sweep_id'].dropna().astype(str))
            else:
                return set()
        except Exception:
            # If there's any error reading the file, assume no completed sweeps
            return set()
    
    def append_result(
        self, 
        dataset_split_path: str,
        sweep_combination: SweepCombination,
        dataset_metadata: Dict[str, Any],
        clustering_summary: Dict[str, Any],
        evaluation_metrics: Dict[str, Any],
        execution_time: float,
        error_message: Optional[str] = None
    ) -> None:
        """
        Append a single result to the CSV report.
        
        Args:
            dataset_split_path: Path to dataset split file
            sweep_combination: Sweep combination that was executed
            dataset_metadata: Metadata from dataset
            clustering_summary: Summary of clustering results
            evaluation_metrics: Evaluation metrics
            execution_time: Time taken for execution
            error_message: Error message if execution failed
        """
        # Create result row
        result_row = self._create_result_row(
            dataset_split_path=dataset_split_path,
            sweep_combination=sweep_combination,
            dataset_metadata=dataset_metadata,
            clustering_summary=clustering_summary,
            evaluation_metrics=evaluation_metrics,
            execution_time=execution_time,
            error_message=error_message
        )
        
        # Check if file exists to determine if we need to write headers
        file_exists = os.path.exists(self.report_path)
        
        # Get column names (now fixed since hyperparameters is a single column)
        column_names = self._get_column_names()
        
        # Write to CSV
        with open(self.report_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=column_names)
            
            # Write header if file is new
            if not file_exists:
                writer.writeheader()
            
            # Write result row
            writer.writerow(result_row)
    
    def _create_result_row(
        self,
        dataset_split_path: str,
        sweep_combination: SweepCombination,
        dataset_metadata: Dict[str, Any],
        clustering_summary: Dict[str, Any],
        evaluation_metrics: Dict[str, Any],
        execution_time: float,
        error_message: Optional[str] = None
    ) -> Dict[str, Any]:
        """Create a single result row for the CSV report"""
        
        # Base information
        row = {
            'timestamp': datetime.now().isoformat(),
            'dataset_split_path': dataset_split_path,
            'algorithm': sweep_combination.algorithm,
            'sweep_id': sweep_combination.sweep_id,
            'execution_time': execution_time,
            'error_message': error_message or ''
        }
        
        # Dataset metadata
        row.update({
            'n_flows': dataset_metadata.get('n_flows'),
            'n_features': dataset_metadata.get('n_features'),
            'n_unique_flow_labels': dataset_metadata.get('n_unique_flow_labels'),
            'dataset_name': dataset_metadata.get('dataset_name'),
            'split_name': dataset_metadata.get('split_name'),
            'label_type': dataset_metadata.get('label_type'),
            'original_n_flows': dataset_metadata.get('original_n_flows'),
            'max_flows_used': dataset_metadata.get('max_flows_used'),
            'flows_sampled': dataset_metadata.get('flows_sampled', False)
        })
        
        # Algorithm hyperparameters (as JSON string)
        row['hyperparameters'] = json.dumps(sweep_combination.hyperparameters, sort_keys=True)
        
        # Clustering summary
        row.update({
            'n_clusters': clustering_summary.get('n_clusters'),
            'n_noise_samples': clustering_summary.get('n_noise_samples'),
            'noise_ratio': clustering_summary.get('noise_ratio')
        })
        
        # Evaluation metrics
        metric_columns = [
            'silhouette_score', 'calinski_harabasz_score', 'davies_bouldin_score',
            'adjusted_rand_score', 'adjusted_mutual_info_score', 'homogeneity_score',
            'completeness_score', 'v_measure_score', 'avg_cluster_label_entropy',
            'avg_cluster_label_diversity', 'avg_label_coverage_per_cluster',
            'cluster_size_std', 'cluster_size_cv', 'largest_cluster_ratio'
        ]
        
        for metric in metric_columns:
            row[metric] = evaluation_metrics.get(metric)
        
        # Handle evaluation errors
        if 'evaluation_error' in evaluation_metrics:
            row['evaluation_error'] = evaluation_metrics['evaluation_error']
        else:
            row['evaluation_error'] = ''
        
        return row
    
    def _get_column_names(self) -> List[str]:
        """Get standardized column names for the CSV report"""
        base_columns = [
            'timestamp', 'dataset_split_path', 'algorithm', 'hyperparameters', 'sweep_id',
            'n_flows', 'n_features', 'n_unique_flow_labels',
            'dataset_name', 'split_name', 'label_type',
            'original_n_flows', 'max_flows_used', 'flows_sampled',
            'n_clusters', 'n_noise_samples', 'noise_ratio'
        ]
        
        evaluation_columns = [
            'silhouette_score', 'calinski_harabasz_score', 'davies_bouldin_score',
            'adjusted_rand_score', 'adjusted_mutual_info_score', 'homogeneity_score',
            'completeness_score', 'v_measure_score', 'avg_cluster_label_entropy',
            'avg_cluster_label_diversity', 'avg_label_coverage_per_cluster',
            'cluster_size_std', 'cluster_size_cv', 'largest_cluster_ratio'
        ]
        
        final_columns = [
            'execution_time', 'error_message', 'evaluation_error'
        ]
        
        return base_columns + evaluation_columns + final_columns
    
    def get_report_summary(self) -> Dict[str, Any]:
        """
        Get summary statistics from the current report.
        
        Returns:
            Dictionary with report summary statistics
        """
        if not os.path.exists(self.report_path):
            return {
                'total_experiments': 0,
                'successful_experiments': 0,
                'failed_experiments': 0,
                'algorithms_tested': [],
                'datasets_tested': []
            }
        
        try:
            df = pd.read_csv(self.report_path)
            
            total_experiments = len(df)
            failed_experiments = len(df[df['error_message'].notna() & (df['error_message'] != '')])
            successful_experiments = total_experiments - failed_experiments
            
            algorithms_tested = df['algorithm'].unique().tolist()
            datasets_tested = df['dataset_split_path'].unique().tolist()
            
            return {
                'total_experiments': total_experiments,
                'successful_experiments': successful_experiments,
                'failed_experiments': failed_experiments,
                'algorithms_tested': algorithms_tested,
                'datasets_tested': datasets_tested
            }
        
        except Exception as e:
            return {
                'error': f"Failed to read report: {str(e)}",
                'total_experiments': 0,
                'successful_experiments': 0,
                'failed_experiments': 0,
                'algorithms_tested': [],
                'datasets_tested': []
            }