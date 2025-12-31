"""
Personalized Treatment Recommendation System - Modern Implementation

This is a research demonstration system that uses advanced NLP techniques
to recommend treatments based on patient profiles. It includes multiple
models (TF-IDF, ClinicalBERT, Dual Encoder) with comprehensive evaluation
and safety features.

DISCLAIMER: This is for research and educational purposes only.
NOT intended for clinical use or medical diagnosis.
"""

import os
import sys
from typing import Dict, List, Any

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from omegaconf import OmegaConf
from data import TreatmentDataProcessor
from models import TreatmentRecommenderFactory
from metrics import ModelEvaluator
from utils import set_seed, SafetyChecker


def main():
    """Main demonstration function."""
    print("="*80)
    print("PERSONALIZED TREATMENT RECOMMENDATION SYSTEM")
    print("="*80)
    print()
    
    # Safety disclaimer
    print("⚠️  IMPORTANT DISCLAIMER:")
    print("   This is a research demonstration system only.")
    print("   NOT intended for clinical use or medical diagnosis.")
    print("   All recommendations should be reviewed by qualified healthcare professionals.")
    print()
    
    # Load configuration
    config_path = os.path.join(os.path.dirname(__file__), 'configs', 'config.yaml')
    config = OmegaConf.load(config_path)
    
    # Set seed for reproducibility
    set_seed(42)
    
    # Initialize components
    data_processor = TreatmentDataProcessor(config.data)
    safety_checker = SafetyChecker(config)
    
    # Generate synthetic data
    print("Generating synthetic clinical data...")
    data = data_processor.generate_synthetic_data(100)
    
    # Create treatment database
    treatment_database = []
    for sample in data:
        treatment = {
            "name": sample["treatment_name"],
            "description": sample["treatment_description"]
        }
        if treatment not in treatment_database:
            treatment_database.append(treatment)
    
    print(f"Created treatment database with {len(treatment_database)} treatments")
    print()
    
    # Test different models
    models_to_test = ["tfidf", "clinicalbert", "dual_encoder"]
    
    for model_type in models_to_test:
        print(f"Testing {model_type.upper()} model:")
        print("-" * 40)
        
        try:
            # Create model
            model = TreatmentRecommenderFactory.create_model(model_type, config.model)
            
            # For TF-IDF, fit the model
            if model_type == "tfidf":
                texts = [sample["patient_profile"] + " " + sample["treatment_description"] 
                        for sample in data]
                model.fit(texts)
            
            # Test with sample patient
            sample_patient = data[0]["patient_profile"]
            print(f"Patient Profile: {sample_patient[:100]}...")
            print()
            
            # Get recommendations
            recommendations = model.recommend_treatments(
                sample_patient, treatment_database, top_k=3
            )
            
            # Display results
            print("Top 3 Recommendations:")
            for i, rec in enumerate(recommendations, 1):
                treatment = rec["treatment"]
                confidence = rec["confidence"]
                print(f"  {i}. {treatment['name']}")
                print(f"     Confidence: {confidence:.3f}")
                print(f"     Description: {treatment['description'][:80]}...")
                print()
            
            print()
            
        except Exception as e:
            print(f"Error with {model_type}: {str(e)}")
            print()
    
    # Evaluation
    print("Running comprehensive evaluation...")
    evaluator = ModelEvaluator(config.evaluation)
    
    # Evaluate TF-IDF model (most reliable)
    try:
        tfidf_model = TreatmentRecommenderFactory.create_model("tfidf", config.model)
        texts = [sample["patient_profile"] + " " + sample["treatment_description"] 
                for sample in data]
        tfidf_model.fit(texts)
        
        metrics = evaluator.evaluate_model(tfidf_model, data[:20], treatment_database)
        
        print("Evaluation Results:")
        print("-" * 20)
        print(f"Accuracy:     {metrics.get('accuracy', 0.0):.3f}")
        print(f"Precision:    {metrics.get('precision', 0.0):.3f}")
        print(f"Recall:       {metrics.get('recall', 0.0):.3f}")
        print(f"F1-Score:     {metrics.get('f1_score', 0.0):.3f}")
        print(f"AUC-ROC:      {metrics.get('auc_roc', 0.0):.3f}")
        print(f"Hit Rate @ 1: {metrics.get('hit_rate_at_1', 0.0):.3f}")
        print(f"Hit Rate @ 3: {metrics.get('hit_rate_at_3', 0.0):.3f}")
        print(f"Hit Rate @ 5: {metrics.get('hit_rate_at_5', 0.0):.3f}")
        
    except Exception as e:
        print(f"Evaluation error: {str(e)}")
    
    print()
    print("="*80)
    print("DEMONSTRATION COMPLETED")
    print("="*80)
    print()
    print("To run the interactive demo:")
    print("  streamlit run demo/app.py")
    print()
    print("To train models:")
    print("  python scripts/train.py")
    print()


if __name__ == "__main__":
    main()