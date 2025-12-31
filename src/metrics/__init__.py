"""
Comprehensive evaluation metrics for treatment recommendation systems.

This module provides clinical-specific evaluation metrics including
accuracy, precision, recall, F1-score, AUC, calibration, and explainability metrics.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score, auc, average_precision_score, confusion_matrix,
    f1_score, precision_recall_curve, precision_score, recall_score,
    roc_auc_score, roc_curve
)
from sklearn.metrics.pairwise import cosine_similarity


class TreatmentRecommendationMetrics:
    """Comprehensive metrics for treatment recommendation evaluation."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize metrics calculator.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.top_k_values = config.get("top_k_recommendations", [1, 3, 5])
    
    def calculate_classification_metrics(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray, 
        y_prob: np.ndarray
    ) -> Dict[str, float]:
        """Calculate standard classification metrics.
        
        Args:
            y_true: True binary labels
            y_pred: Predicted binary labels
            y_prob: Predicted probabilities
            
        Returns:
            Dictionary of classification metrics
        """
        metrics = {}
        
        # Basic classification metrics
        metrics["accuracy"] = accuracy_score(y_true, y_pred)
        metrics["precision"] = precision_score(y_true, y_pred, zero_division=0)
        metrics["recall"] = recall_score(y_true, y_pred, zero_division=0)
        metrics["f1_score"] = f1_score(y_true, y_pred, zero_division=0)
        
        # AUC metrics
        try:
            metrics["auc_roc"] = roc_auc_score(y_true, y_prob)
        except ValueError:
            metrics["auc_roc"] = 0.0
        
        try:
            metrics["auc_pr"] = average_precision_score(y_true, y_prob)
        except ValueError:
            metrics["auc_pr"] = 0.0
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            metrics["specificity"] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            metrics["sensitivity"] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            metrics["npv"] = tn / (tn + fn) if (tn + fn) > 0 else 0.0
            metrics["ppv"] = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        
        return metrics
    
    def calculate_top_k_metrics(
        self, 
        recommendations: List[List[Dict[str, Any]]], 
        ground_truth: List[List[str]]
    ) -> Dict[str, float]:
        """Calculate top-k recommendation metrics.
        
        Args:
            recommendations: List of recommendation lists for each patient
            ground_truth: List of ground truth treatment lists for each patient
            
        Returns:
            Dictionary of top-k metrics
        """
        metrics = {}
        
        for k in self.top_k_values:
            hits = 0
            total_patients = len(recommendations)
            
            for i, (recs, gt) in enumerate(zip(recommendations, ground_truth)):
                # Get top-k recommended treatment names
                top_k_treatments = [rec["treatment"]["name"] for rec in recs[:k]]
                
                # Check if any ground truth treatment is in top-k
                if any(gt_treatment in top_k_treatments for gt_treatment in gt):
                    hits += 1
            
            metrics[f"hit_rate_at_{k}"] = hits / total_patients if total_patients > 0 else 0.0
        
        return metrics
    
    def calculate_calibration_metrics(
        self, 
        y_true: np.ndarray, 
        y_prob: np.ndarray,
        n_bins: int = 10
    ) -> Dict[str, float]:
        """Calculate calibration metrics.
        
        Args:
            y_true: True binary labels
            y_prob: Predicted probabilities
            n_bins: Number of bins for calibration curve
            
        Returns:
            Dictionary of calibration metrics
        """
        metrics = {}
        
        try:
            # Calibration curve
            fraction_of_positives, mean_predicted_value = calibration_curve(
                y_true, y_prob, n_bins=n_bins
            )
            
            # Expected Calibration Error (ECE)
            bin_boundaries = np.linspace(0, 1, n_bins + 1)
            bin_lowers = bin_boundaries[:-1]
            bin_uppers = bin_boundaries[1:]
            
            ece = 0
            for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
                in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)
                prop_in_bin = in_bin.mean()
                
                if prop_in_bin > 0:
                    accuracy_in_bin = y_true[in_bin].mean()
                    avg_confidence_in_bin = y_prob[in_bin].mean()
                    ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
            
            metrics["ece"] = ece
            
            # Maximum Calibration Error (MCE)
            mce = 0
            for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
                in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)
                prop_in_bin = in_bin.mean()
                
                if prop_in_bin > 0:
                    accuracy_in_bin = y_true[in_bin].mean()
                    avg_confidence_in_bin = y_prob[in_bin].mean()
                    mce = max(mce, np.abs(avg_confidence_in_bin - accuracy_in_bin))
            
            metrics["mce"] = mce
            
        except Exception as e:
            self.logger.warning(f"Failed to calculate calibration metrics: {e}")
            metrics["ece"] = 0.0
            metrics["mce"] = 0.0
        
        return metrics
    
    def calculate_diversity_metrics(
        self, 
        recommendations: List[List[Dict[str, Any]]]
    ) -> Dict[str, float]:
        """Calculate recommendation diversity metrics.
        
        Args:
            recommendations: List of recommendation lists for each patient
            
        Returns:
            Dictionary of diversity metrics
        """
        metrics = {}
        
        # Intra-list diversity (average pairwise similarity within recommendations)
        intra_diversities = []
        
        for recs in recommendations:
            if len(recs) < 2:
                continue
            
            # Extract treatment descriptions
            treatments = [rec["treatment"]["description"] for rec in recs]
            
            # Calculate pairwise similarities (simplified using word overlap)
            similarities = []
            for i in range(len(treatments)):
                for j in range(i + 1, len(treatments)):
                    # Simple word-based similarity
                    words_i = set(treatments[i].lower().split())
                    words_j = set(treatments[j].lower().split())
                    similarity = len(words_i.intersection(words_j)) / len(words_i.union(words_j))
                    similarities.append(similarity)
            
            if similarities:
                intra_diversities.append(1 - np.mean(similarities))  # Diversity = 1 - similarity
        
        metrics["intra_list_diversity"] = np.mean(intra_diversities) if intra_diversities else 0.0
        
        # Inter-list diversity (coverage of different treatments)
        all_treatments = set()
        for recs in recommendations:
            for rec in recs:
                all_treatments.add(rec["treatment"]["name"])
        
        metrics["treatment_coverage"] = len(all_treatments)
        
        return metrics
    
    def calculate_clinical_metrics(
        self, 
        recommendations: List[List[Dict[str, Any]]], 
        patient_conditions: List[str]
    ) -> Dict[str, float]:
        """Calculate clinical-specific metrics.
        
        Args:
            recommendations: List of recommendation lists for each patient
            patient_conditions: List of patient conditions
            
        Returns:
            Dictionary of clinical metrics
        """
        metrics = {}
        
        # Condition-treatment alignment
        aligned_recommendations = 0
        total_recommendations = 0
        
        for recs, condition in zip(recommendations, patient_conditions):
            for rec in recs:
                total_recommendations += 1
                treatment = rec["treatment"]
                
                # Check if treatment is indicated for the condition
                if condition.lower() in treatment.get("description", "").lower():
                    aligned_recommendations += 1
        
        metrics["condition_treatment_alignment"] = (
            aligned_recommendations / total_recommendations 
            if total_recommendations > 0 else 0.0
        )
        
        # Average confidence score
        all_confidences = []
        for recs in recommendations:
            for rec in recs:
                all_confidences.append(rec.get("confidence", 0.0))
        
        metrics["average_confidence"] = np.mean(all_confidences) if all_confidences else 0.0
        
        return metrics
    
    def calculate_comprehensive_metrics(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray, 
        y_prob: np.ndarray,
        recommendations: List[List[Dict[str, Any]]],
        ground_truth: List[List[str]],
        patient_conditions: List[str]
    ) -> Dict[str, float]:
        """Calculate comprehensive evaluation metrics.
        
        Args:
            y_true: True binary labels
            y_pred: Predicted binary labels
            y_prob: Predicted probabilities
            recommendations: List of recommendation lists
            ground_truth: List of ground truth treatments
            patient_conditions: List of patient conditions
            
        Returns:
            Dictionary of all metrics
        """
        all_metrics = {}
        
        # Classification metrics
        classification_metrics = self.calculate_classification_metrics(y_true, y_pred, y_prob)
        all_metrics.update(classification_metrics)
        
        # Top-k metrics
        top_k_metrics = self.calculate_top_k_metrics(recommendations, ground_truth)
        all_metrics.update(top_k_metrics)
        
        # Calibration metrics
        calibration_metrics = self.calculate_calibration_metrics(y_true, y_prob)
        all_metrics.update(calibration_metrics)
        
        # Diversity metrics
        diversity_metrics = self.calculate_diversity_metrics(recommendations)
        all_metrics.update(diversity_metrics)
        
        # Clinical metrics
        clinical_metrics = self.calculate_clinical_metrics(recommendations, patient_conditions)
        all_metrics.update(clinical_metrics)
        
        return all_metrics


class ModelEvaluator:
    """Model evaluator for treatment recommendation systems."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize model evaluator.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.metrics_calculator = TreatmentRecommendationMetrics(config)
        self.logger = logging.getLogger(__name__)
    
    def evaluate_model(
        self, 
        model: Any, 
        test_data: List[Dict[str, Any]],
        treatment_database: List[Dict[str, str]]
    ) -> Dict[str, float]:
        """Evaluate model on test data.
        
        Args:
            model: Trained model
            test_data: Test dataset
            treatment_database: Available treatments
            
        Returns:
            Dictionary of evaluation metrics
        """
        self.logger.info("Starting model evaluation")
        
        # Prepare data
        y_true = []
        y_pred = []
        y_prob = []
        recommendations = []
        ground_truth = []
        patient_conditions = []
        
        for sample in test_data:
            patient_text = sample["patient_profile"]
            true_treatment = sample["treatment_name"]
            condition = sample.get("condition", "")
            
            # Get recommendations
            recs = model.recommend_treatments(patient_text, treatment_database, top_k=5)
            recommendations.append(recs)
            ground_truth.append([true_treatment])
            patient_conditions.append(condition)
            
            # Check if true treatment is in top-1 recommendation
            top_rec = recs[0] if recs else None
            if top_rec and top_rec["treatment"]["name"] == true_treatment:
                y_true.append(1)
                y_pred.append(1)
                y_prob.append(top_rec["confidence"])
            else:
                y_true.append(0)
                y_pred.append(0)
                y_prob.append(1 - (top_rec["confidence"] if top_rec else 0.0))
        
        # Calculate metrics
        metrics = self.metrics_calculator.calculate_comprehensive_metrics(
            np.array(y_true),
            np.array(y_pred),
            np.array(y_prob),
            recommendations,
            ground_truth,
            patient_conditions
        )
        
        self.logger.info("Model evaluation completed")
        return metrics
    
    def create_evaluation_report(
        self, 
        metrics: Dict[str, float], 
        model_name: str
    ) -> str:
        """Create a formatted evaluation report.
        
        Args:
            metrics: Evaluation metrics
            model_name: Name of the evaluated model
            
        Returns:
            Formatted evaluation report
        """
        report = f"\n{'='*60}\n"
        report += f"EVALUATION REPORT: {model_name}\n"
        report += f"{'='*60}\n\n"
        
        # Classification metrics
        report += "CLASSIFICATION METRICS:\n"
        report += f"  Accuracy:           {metrics.get('accuracy', 0.0):.4f}\n"
        report += f"  Precision:          {metrics.get('precision', 0.0):.4f}\n"
        report += f"  Recall:             {metrics.get('recall', 0.0):.4f}\n"
        report += f"  F1-Score:           {metrics.get('f1_score', 0.0):.4f}\n"
        report += f"  AUC-ROC:            {metrics.get('auc_roc', 0.0):.4f}\n"
        report += f"  AUC-PR:             {metrics.get('auc_pr', 0.0):.4f}\n"
        report += f"  Sensitivity:        {metrics.get('sensitivity', 0.0):.4f}\n"
        report += f"  Specificity:        {metrics.get('specificity', 0.0):.4f}\n\n"
        
        # Top-k metrics
        report += "TOP-K RECOMMENDATION METRICS:\n"
        for k in self.config.get("top_k_recommendations", [1, 3, 5]):
            hit_rate = metrics.get(f"hit_rate_at_{k}", 0.0)
            report += f"  Hit Rate @ {k}:        {hit_rate:.4f}\n"
        report += "\n"
        
        # Calibration metrics
        report += "CALIBRATION METRICS:\n"
        report += f"  Expected Calibration Error: {metrics.get('ece', 0.0):.4f}\n"
        report += f"  Maximum Calibration Error:   {metrics.get('mce', 0.0):.4f}\n\n"
        
        # Clinical metrics
        report += "CLINICAL METRICS:\n"
        report += f"  Condition-Treatment Alignment: {metrics.get('condition_treatment_alignment', 0.0):.4f}\n"
        report += f"  Average Confidence:             {metrics.get('average_confidence', 0.0):.4f}\n"
        report += f"  Treatment Coverage:             {metrics.get('treatment_coverage', 0.0):.0f}\n"
        report += f"  Intra-list Diversity:          {metrics.get('intra_list_diversity', 0.0):.4f}\n\n"
        
        report += f"{'='*60}\n"
        
        return report
