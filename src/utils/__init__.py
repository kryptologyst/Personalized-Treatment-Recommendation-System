"""
Core utilities for the Personalized Treatment Recommendation System.

This module provides essential utilities including device management,
seeding, logging, and safety features for healthcare AI applications.
"""

import logging
import os
import random
import warnings
from typing import Any, Dict, Optional, Union

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf


def setup_logging(config: DictConfig) -> logging.Logger:
    """Set up structured logging for the application.
    
    Args:
        config: Configuration object containing logging settings
        
    Returns:
        Configured logger instance
    """
    # Create logs directory if it doesn't exist
    os.makedirs(config.paths.logs_dir, exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, config.logging.level),
        format=config.logging.format,
        handlers=[
            logging.FileHandler(config.logging.file),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info("Logging initialized successfully")
    return logger


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # For deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Suppress warnings for cleaner output
    warnings.filterwarnings("ignore", category=UserWarning)


def get_device(config: DictConfig) -> torch.device:
    """Get the appropriate device for computation.
    
    Args:
        config: Configuration object
        
    Returns:
        PyTorch device object
    """
    device_config = config.training.device
    
    if device_config == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(device_config)
    
    return device


def load_config(config_path: str) -> DictConfig:
    """Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        OmegaConf configuration object
    """
    return OmegaConf.load(config_path)


def save_config(config: DictConfig, save_path: str) -> None:
    """Save configuration to YAML file.
    
    Args:
        config: Configuration object to save
        save_path: Path where to save the configuration
    """
    OmegaConf.save(config, save_path)


def create_directories(config: DictConfig) -> None:
    """Create necessary directories for the project.
    
    Args:
        config: Configuration object containing path settings
    """
    directories = [
        config.paths.data_dir,
        config.paths.model_dir,
        config.paths.logs_dir,
        config.paths.assets_dir,
        config.training.checkpoint_dir
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)


def validate_config(config: DictConfig) -> None:
    """Validate configuration parameters.
    
    Args:
        config: Configuration object to validate
        
    Raises:
        ValueError: If configuration is invalid
    """
    # Validate data splits
    total_split = config.data.train_split + config.data.val_split + config.data.test_split
    if not np.isclose(total_split, 1.0, atol=1e-6):
        raise ValueError(f"Data splits must sum to 1.0, got {total_split}")
    
    # Validate model parameters
    if config.model.learning_rate <= 0:
        raise ValueError("Learning rate must be positive")
    
    if config.model.batch_size <= 0:
        raise ValueError("Batch size must be positive")
    
    # Validate evaluation parameters
    if config.evaluation.confidence_threshold < 0 or config.evaluation.confidence_threshold > 1:
        raise ValueError("Confidence threshold must be between 0 and 1")


class SafetyChecker:
    """Safety checker for healthcare AI applications."""
    
    def __init__(self, config: DictConfig):
        """Initialize safety checker.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def check_input_safety(self, text: str) -> bool:
        """Check if input text is safe for processing.
        
        Args:
            text: Input text to check
            
        Returns:
            True if input is safe, False otherwise
        """
        if not isinstance(text, str):
            self.logger.warning("Input is not a string")
            return False
        
        if len(text.strip()) == 0:
            self.logger.warning("Input text is empty")
            return False
        
        if len(text) > 10000:  # Reasonable limit for clinical text
            self.logger.warning("Input text is too long")
            return False
        
        return True
    
    def log_disclaimer(self) -> None:
        """Log the required disclaimer."""
        disclaimer = (
            "DISCLAIMER: This is a research demonstration system. "
            "It is NOT intended for clinical use or medical diagnosis. "
            "All recommendations should be reviewed by qualified healthcare professionals."
        )
        self.logger.warning(disclaimer)


def format_model_size(model: torch.nn.Module) -> str:
    """Format model size in human-readable format.
    
    Args:
        model: PyTorch model
        
    Returns:
        Formatted string with model size information
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return f"Total: {total_params:,} | Trainable: {trainable_params:,}"


def count_parameters(model: torch.nn.Module) -> Dict[str, int]:
    """Count model parameters.
    
    Args:
        model: PyTorch model
        
    Returns:
        Dictionary with parameter counts
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        "total": total_params,
        "trainable": trainable_params,
        "non_trainable": total_params - trainable_params
    }
