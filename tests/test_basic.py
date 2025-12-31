"""
Basic tests for the Personalized Treatment Recommendation System.

This module provides unit tests to ensure the system works correctly
and maintains quality standards.
"""

import os
import sys
import unittest
from unittest.mock import Mock, patch

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from data import TreatmentDataProcessor, DeidentificationProcessor
from models import TreatmentRecommenderFactory
from utils import set_seed, SafetyChecker


class TestTreatmentDataProcessor(unittest.TestCase):
    """Test cases for TreatmentDataProcessor."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = {
            'max_samples': 10,
            'train_split': 0.7,
            'val_split': 0.15,
            'test_split': 0.15
        }
        self.processor = TreatmentDataProcessor(self.config)
    
    def test_generate_synthetic_data(self):
        """Test synthetic data generation."""
        data = self.processor.generate_synthetic_data(5)
        
        self.assertEqual(len(data), 5)
        self.assertIsInstance(data, list)
        
        # Check data structure
        sample = data[0]
        required_keys = ['patient_id', 'patient_profile', 'treatment_name', 'treatment_description']
        for key in required_keys:
            self.assertIn(key, sample)
    
    def test_split_data(self):
        """Test data splitting."""
        data = self.processor.generate_synthetic_data(10)
        train_data, val_data, test_data = self.processor.split_data(data)
        
        # Check splits sum to total
        total_split = len(train_data) + len(val_data) + len(test_data)
        self.assertEqual(total_split, len(data))
        
        # Check splits are not empty
        self.assertGreater(len(train_data), 0)
        self.assertGreater(len(val_data), 0)
        self.assertGreater(len(test_data), 0)


class TestDeidentificationProcessor(unittest.TestCase):
    """Test cases for DeidentificationProcessor."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.processor = DeidentificationProcessor()
    
    def test_deidentify_text(self):
        """Test text de-identification."""
        text_with_phi = "Patient John Doe, phone 555-123-4567, SSN 123-45-6789, email john@example.com"
        deidentified = self.processor.deidentify_text(text_with_phi)
        
        # Check PHI is replaced
        self.assertIn('[PHONE]', deidentified)
        self.assertIn('[SSN]', deidentified)
        self.assertIn('[EMAIL]', deidentified)
        
        # Check original PHI is not present
        self.assertNotIn('555-123-4567', deidentified)
        self.assertNotIn('123-45-6789', deidentified)
        self.assertNotIn('john@example.com', deidentified)
    
    def test_check_phi_content(self):
        """Test PHI detection."""
        text_with_phi = "Patient phone 555-123-4567"
        text_without_phi = "Patient has diabetes"
        
        self.assertTrue(self.processor.check_phi_content(text_with_phi))
        self.assertFalse(self.processor.check_phi_content(text_without_phi))


class TestTreatmentRecommenderFactory(unittest.TestCase):
    """Test cases for TreatmentRecommenderFactory."""
    
    def test_get_available_models(self):
        """Test getting available models."""
        models = TreatmentRecommenderFactory.get_available_models()
        
        self.assertIsInstance(models, list)
        self.assertIn('tfidf', models)
        self.assertIn('clinicalbert', models)
        self.assertIn('dual_encoder', models)
    
    def test_create_tfidf_model(self):
        """Test creating TF-IDF model."""
        config = {'max_features': 100}
        
        model = TreatmentRecommenderFactory.create_model('tfidf', config)
        
        self.assertIsNotNone(model)
        self.assertEqual(model.__class__.__name__, 'TFIDFRecommender')


class TestSafetyChecker(unittest.TestCase):
    """Test cases for SafetyChecker."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = {'safety': {'deidentify': True}}
        self.checker = SafetyChecker(self.config)
    
    def test_check_input_safety(self):
        """Test input safety checking."""
        # Valid input
        valid_text = "Patient has diabetes"
        self.assertTrue(self.checker.check_input_safety(valid_text))
        
        # Empty input
        self.assertFalse(self.checker.check_input_safety(""))
        
        # Too long input
        long_text = "x" * 15000
        self.assertFalse(self.checker.check_input_safety(long_text))
        
        # Non-string input
        self.assertFalse(self.checker.check_input_safety(123))


class TestUtils(unittest.TestCase):
    """Test cases for utility functions."""
    
    def test_set_seed(self):
        """Test seed setting."""
        # This is a basic test - in practice, you'd test reproducibility
        set_seed(42)
        # Seed setting should not raise an exception
        self.assertTrue(True)


class TestIntegration(unittest.TestCase):
    """Integration tests."""
    
    def test_end_to_end_tfidf(self):
        """Test end-to-end TF-IDF workflow."""
        # Generate data
        config = {'max_samples': 5}
        processor = TreatmentDataProcessor(config)
        data = processor.generate_synthetic_data(5)
        
        # Create model
        model_config = {'max_features': 100}
        model = TreatmentRecommenderFactory.create_model('tfidf', model_config)
        
        # Fit model
        texts = [sample["patient_profile"] + " " + sample["treatment_description"] 
                for sample in data]
        model.fit(texts)
        
        # Test recommendation
        patient_text = data[0]["patient_profile"]
        treatment_database = [{"name": "Test Treatment", "description": "Test description"}]
        
        recommendations = model.recommend_treatments(patient_text, treatment_database, top_k=1)
        
        self.assertIsInstance(recommendations, list)
        self.assertGreater(len(recommendations), 0)


if __name__ == '__main__':
    # Run tests
    unittest.main()
