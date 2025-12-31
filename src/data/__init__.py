"""
Data processing utilities for treatment recommendation system.

This module handles data loading, preprocessing, and synthetic data generation
for the personalized treatment recommendation system.
"""

import json
import logging
import random
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer


class TreatmentDataProcessor:
    """Data processor for treatment recommendation system."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize data processor.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.tokenizer = None
        self.vectorizer = None
        
    def setup_tokenizer(self, model_name: str) -> None:
        """Set up tokenizer for the specified model.
        
        Args:
            model_name: Name of the model to use
        """
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.logger.info(f"Tokenizer loaded for {model_name}")
        except Exception as e:
            self.logger.error(f"Failed to load tokenizer for {model_name}: {e}")
            raise
    
    def setup_tfidf_vectorizer(self) -> None:
        """Set up TF-IDF vectorizer for baseline model."""
        self.vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        self.logger.info("TF-IDF vectorizer initialized")
    
    def generate_synthetic_data(self, num_samples: int = 1000) -> List[Dict[str, Any]]:
        """Generate synthetic clinical data for demonstration.
        
        Args:
            num_samples: Number of samples to generate
            
        Returns:
            List of synthetic patient-treatment pairs
        """
        self.logger.info(f"Generating {num_samples} synthetic samples")
        
        # Define synthetic patient profiles
        patient_templates = [
            "Patient presents with {condition} and {comorbidity}. Age: {age}, BMI: {bmi}. "
            "Current medications: {medications}. Allergies: {allergies}. "
            "Treatment goals: {goals}.",
            
            "{age}-year-old {gender} with {condition}. Medical history includes {history}. "
            "Vital signs: BP {bp}, HR {hr}. Laboratory values: {labs}. "
            "Recommended treatment approach: {approach}.",
            
            "Clinical presentation: {presentation}. Patient reports {symptoms}. "
            "Physical examination reveals {findings}. Diagnostic workup shows {results}. "
            "Treatment plan: {plan}."
        ]
        
        # Define conditions and treatments
        conditions = [
            "type 2 diabetes", "hypertension", "hyperlipidemia", "obesity",
            "depression", "anxiety", "chronic pain", "asthma", "COPD",
            "heart failure", "atrial fibrillation", "osteoporosis"
        ]
        
        treatments = [
            {
                "name": "Metformin + Lifestyle Modification",
                "description": "Metformin 500mg twice daily with comprehensive lifestyle counseling including diet modification and exercise program. Recommended for newly diagnosed type 2 diabetes patients with BMI >25.",
                "indications": ["type 2 diabetes", "obesity"],
                "contraindications": ["renal impairment", "liver disease"]
            },
            {
                "name": "ACE Inhibitor Therapy",
                "description": "Lisinopril 10mg daily for blood pressure control. Monitor renal function and potassium levels. First-line therapy for hypertension with diabetes or heart failure.",
                "indications": ["hypertension", "heart failure", "type 2 diabetes"],
                "contraindications": ["pregnancy", "angioedema history"]
            },
            {
                "name": "Statin Therapy",
                "description": "Atorvastatin 20mg daily for cholesterol management. Regular monitoring of liver enzymes and muscle symptoms. Indicated for cardiovascular risk reduction.",
                "indications": ["hyperlipidemia", "heart failure", "type 2 diabetes"],
                "contraindications": ["active liver disease", "pregnancy"]
            },
            {
                "name": "SSRI Antidepressant",
                "description": "Sertraline 50mg daily for depression management. Gradual dose titration with monitoring for side effects. Effective for mood disorders with good safety profile.",
                "indications": ["depression", "anxiety"],
                "contraindications": ["MAOI use", "seizure disorder"]
            },
            {
                "name": "Physical Therapy Program",
                "description": "Structured physical therapy program focusing on pain management and functional improvement. Includes exercises, manual therapy, and education.",
                "indications": ["chronic pain", "osteoporosis"],
                "contraindications": ["acute fracture", "severe osteoporosis"]
            }
        ]
        
        # Generate synthetic data
        synthetic_data = []
        random.seed(42)  # For reproducibility
        
        for i in range(num_samples):
            # Randomly select template and fill in variables
            template = random.choice(patient_templates)
            
            # Generate random clinical variables
            age = random.randint(18, 85)
            gender = random.choice(["male", "female"])
            bmi = round(random.uniform(18.5, 45.0), 1)
            bp_systolic = random.randint(110, 180)
            bp_diastolic = random.randint(70, 110)
            hr = random.randint(60, 100)
            
            condition = random.choice(conditions)
            comorbidity = random.choice([c for c in conditions if c != condition])
            
            # Fill template
            patient_profile = template.format(
                condition=condition,
                comorbidity=comorbidity,
                age=age,
                gender=gender,
                bmi=bmi,
                medications=random.choice(["metformin", "lisinopril", "atorvastatin", "none"]),
                allergies=random.choice(["penicillin", "sulfa", "none known"]),
                goals=random.choice(["blood sugar control", "weight loss", "pain management", "mood improvement"]),
                history=random.choice(["diabetes", "hypertension", "depression", "none significant"]),
                bp=f"{bp_systolic}/{bp_diastolic}",
                hr=hr,
                labs=random.choice(["normal", "elevated glucose", "high cholesterol", "normal"]),
                approach=random.choice(["conservative", "aggressive", "stepwise"]),
                presentation=random.choice(["acute", "chronic", "progressive"]),
                symptoms=random.choice(["fatigue", "pain", "shortness of breath", "depression"]),
                findings=random.choice(["normal", "abnormal heart sounds", "joint swelling", "normal"]),
                results=random.choice(["normal", "abnormal", "inconclusive"]),
                plan=random.choice(["medication", "therapy", "surgery", "observation"])
            )
            
            # Select appropriate treatment based on condition
            suitable_treatments = [t for t in treatments if condition in t["indications"]]
            if not suitable_treatments:
                suitable_treatments = treatments  # Fallback
            
            treatment = random.choice(suitable_treatments)
            
            # Create sample
            sample = {
                "patient_id": f"patient_{i:04d}",
                "patient_profile": patient_profile,
                "treatment_name": treatment["name"],
                "treatment_description": treatment["description"],
                "condition": condition,
                "age": age,
                "gender": gender,
                "bmi": bmi,
                "label": 1,  # Positive match
                "confidence": random.uniform(0.7, 0.95)
            }
            
            synthetic_data.append(sample)
        
        self.logger.info(f"Generated {len(synthetic_data)} synthetic samples")
        return synthetic_data
    
    def load_data(self, data_path: Optional[str] = None) -> List[Dict[str, Any]]:
        """Load data from file or generate synthetic data.
        
        Args:
            data_path: Path to data file (optional)
            
        Returns:
            List of patient-treatment pairs
        """
        if data_path and os.path.exists(data_path):
            self.logger.info(f"Loading data from {data_path}")
            with open(data_path, 'r') as f:
                data = json.load(f)
        else:
            self.logger.info("No data file found, generating synthetic data")
            data = self.generate_synthetic_data(self.config.get("max_samples", 1000))
        
        return data
    
    def preprocess_text(self, text: str, max_length: int = 512) -> Dict[str, Any]:
        """Preprocess text for model input.
        
        Args:
            text: Input text
            max_length: Maximum sequence length
            
        Returns:
            Preprocessed text data
        """
        if self.tokenizer is None:
            raise ValueError("Tokenizer not initialized")
        
        # Tokenize and encode
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze()
        }
    
    def create_tfidf_features(self, texts: List[str]) -> np.ndarray:
        """Create TF-IDF features for baseline model.
        
        Args:
            texts: List of input texts
            
        Returns:
            TF-IDF feature matrix
        """
        if self.vectorizer is None:
            self.setup_tfidf_vectorizer()
        
        return self.vectorizer.fit_transform(texts).toarray()
    
    def split_data(self, data: List[Dict[str, Any]]) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """Split data into train/validation/test sets.
        
        Args:
            data: List of data samples
            
        Returns:
            Tuple of (train, validation, test) data splits
        """
        train_split = self.config.get("train_split", 0.7)
        val_split = self.config.get("val_split", 0.15)
        
        # Split data
        train_data, temp_data = train_test_split(
            data, 
            test_size=1-train_split, 
            random_state=42
        )
        
        val_data, test_data = train_test_split(
            temp_data,
            test_size=val_split/(val_split + (1-train_split-val_split)),
            random_state=42
        )
        
        self.logger.info(f"Data split - Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")
        
        return train_data, val_data, test_data
    
    def save_data(self, data: List[Dict[str, Any]], file_path: str) -> None:
        """Save data to JSON file.
        
        Args:
            data: Data to save
            file_path: Path to save file
        """
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        self.logger.info(f"Data saved to {file_path}")


class DeidentificationProcessor:
    """Processor for de-identifying clinical text."""
    
    def __init__(self):
        """Initialize de-identification processor."""
        self.logger = logging.getLogger(__name__)
        
        # Common patterns for PHI
        self.patterns = {
            'phone': r'\b\d{3}-\d{3}-\d{4}\b',
            'ssn': r'\b\d{3}-\d{2}-\d{4}\b',
            'date': r'\b\d{1,2}/\d{1,2}/\d{4}\b',
            'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        }
    
    def deidentify_text(self, text: str) -> str:
        """De-identify text by replacing PHI patterns.
        
        Args:
            text: Input text
            
        Returns:
            De-identified text
        """
        import re
        
        deidentified_text = text
        
        # Replace patterns with placeholders
        deidentified_text = re.sub(self.patterns['phone'], '[PHONE]', deidentified_text)
        deidentified_text = re.sub(self.patterns['ssn'], '[SSN]', deidentified_text)
        deidentified_text = re.sub(self.patterns['date'], '[DATE]', deidentified_text)
        deidentified_text = re.sub(self.patterns['email'], '[EMAIL]', deidentified_text)
        
        return deidentified_text
    
    def check_phi_content(self, text: str) -> bool:
        """Check if text contains potential PHI.
        
        Args:
            text: Input text
            
        Returns:
            True if PHI detected, False otherwise
        """
        import re
        
        for pattern in self.patterns.values():
            if re.search(pattern, text):
                return True
        
        return False
