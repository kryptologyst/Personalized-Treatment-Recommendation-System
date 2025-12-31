"""
Training utilities and loss functions for treatment recommendation.

This module provides training utilities, loss functions, and optimization
strategies specifically designed for treatment recommendation tasks.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


class TreatmentDataset(Dataset):
    """Dataset class for treatment recommendation."""
    
    def __init__(self, data: List[Dict[str, Any]], tokenizer: Any, max_length: int = 512):
        """Initialize dataset.
        
        Args:
            data: List of data samples
            tokenizer: Text tokenizer
            max_length: Maximum sequence length
        """
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.logger = logging.getLogger(__name__)
    
    def __len__(self) -> int:
        """Get dataset length."""
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get item by index.
        
        Args:
            idx: Item index
            
        Returns:
            Data sample
        """
        sample = self.data[idx]
        
        # Tokenize patient profile
        patient_inputs = self.tokenizer(
            sample["patient_profile"],
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # Tokenize treatment description
        treatment_inputs = self.tokenizer(
            sample["treatment_description"],
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'patient_input_ids': patient_inputs['input_ids'].squeeze(),
            'patient_attention_mask': patient_inputs['attention_mask'].squeeze(),
            'treatment_input_ids': treatment_inputs['input_ids'].squeeze(),
            'treatment_attention_mask': treatment_inputs['attention_mask'].squeeze(),
            'label': torch.tensor(sample.get('label', 1), dtype=torch.float),
            'patient_id': sample.get('patient_id', ''),
            'treatment_name': sample.get('treatment_name', '')
        }


class ContrastiveLoss(nn.Module):
    """Contrastive loss for treatment recommendation."""
    
    def __init__(self, temperature: float = 0.05):
        """Initialize contrastive loss.
        
        Args:
            temperature: Temperature parameter for scaling
        """
        super().__init__()
        self.temperature = temperature
    
    def forward(self, patient_embeddings: torch.Tensor, treatment_embeddings: torch.Tensor) -> torch.Tensor:
        """Compute contrastive loss.
        
        Args:
            patient_embeddings: Patient embeddings
            treatment_embeddings: Treatment embeddings
            
        Returns:
            Contrastive loss
        """
        # Normalize embeddings
        patient_embeddings = F.normalize(patient_embeddings, p=2, dim=1)
        treatment_embeddings = F.normalize(treatment_embeddings, p=2, dim=1)
        
        # Compute similarity matrix
        similarity_matrix = torch.matmul(patient_embeddings, treatment_embeddings.T) / self.temperature
        
        # Create labels (diagonal should be positive pairs)
        batch_size = patient_embeddings.size(0)
        labels = torch.arange(batch_size).to(patient_embeddings.device)
        
        # Compute loss
        loss = F.cross_entropy(similarity_matrix, labels)
        
        return loss


class TripletLoss(nn.Module):
    """Triplet loss for treatment recommendation."""
    
    def __init__(self, margin: float = 1.0):
        """Initialize triplet loss.
        
        Args:
            margin: Margin for triplet loss
        """
        super().__init__()
        self.margin = margin
    
    def forward(
        self, 
        anchor: torch.Tensor, 
        positive: torch.Tensor, 
        negative: torch.Tensor
    ) -> torch.Tensor:
        """Compute triplet loss.
        
        Args:
            anchor: Anchor embeddings (patient)
            positive: Positive embeddings (correct treatment)
            negative: Negative embeddings (incorrect treatment)
            
        Returns:
            Triplet loss
        """
        # Compute distances
        pos_distance = F.pairwise_distance(anchor, positive, p=2)
        neg_distance = F.pairwise_distance(anchor, negative, p=2)
        
        # Compute loss
        loss = F.relu(pos_distance - neg_distance + self.margin)
        
        return loss.mean()


class FocalLoss(nn.Module):
    """Focal loss for handling class imbalance."""
    
    def __init__(self, alpha: float = 1.0, gamma: float = 2.0):
        """Initialize focal loss.
        
        Args:
            alpha: Weighting factor
            gamma: Focusing parameter
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute focal loss.
        
        Args:
            inputs: Model predictions
            targets: Ground truth labels
            
        Returns:
            Focal loss
        """
        # Compute cross entropy
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        
        # Compute focal loss
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        return focal_loss.mean()


class TreatmentTrainer:
    """Advanced trainer for treatment recommendation models."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize trainer.
        
        Args:
            config: Training configuration
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Loss functions
        self.contrastive_loss = ContrastiveLoss(config.get('temperature', 0.05))
        self.triplet_loss = TripletLoss(config.get('margin', 1.0))
        self.focal_loss = FocalLoss(config.get('alpha', 1.0), config.get('gamma', 2.0))
    
    def create_data_loader(
        self, 
        data: List[Dict[str, Any]], 
        tokenizer: Any, 
        batch_size: int,
        shuffle: bool = True
    ) -> DataLoader:
        """Create data loader.
        
        Args:
            data: Training data
            tokenizer: Text tokenizer
            batch_size: Batch size
            shuffle: Whether to shuffle data
            
        Returns:
            Data loader
        """
        dataset = TreatmentDataset(data, tokenizer, self.config.get('max_length', 512))
        
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=self.config.get('num_workers', 0),
            pin_memory=self.config.get('pin_memory', False)
        )
    
    def train_epoch(
        self, 
        model: nn.Module, 
        data_loader: DataLoader, 
        optimizer: torch.optim.Optimizer,
        device: torch.device
    ) -> float:
        """Train for one epoch.
        
        Args:
            model: Model to train
            data_loader: Training data loader
            optimizer: Optimizer
            device: Device to use
            
        Returns:
            Average training loss
        """
        model.train()
        total_loss = 0.0
        num_batches = len(data_loader)
        
        for batch_idx, batch in enumerate(data_loader):
            # Move batch to device
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            try:
                # Get model outputs
                outputs = model(
                    patient_input_ids=batch['patient_input_ids'],
                    patient_attention_mask=batch['patient_attention_mask'],
                    treatment_input_ids=batch['treatment_input_ids'],
                    treatment_attention_mask=batch['treatment_attention_mask']
                )
                
                # Compute loss
                if hasattr(model, 'patient_embeddings') and hasattr(model, 'treatment_embeddings'):
                    # Use contrastive loss for dual encoder models
                    loss = self.contrastive_loss(outputs['patient_embeddings'], outputs['treatment_embeddings'])
                else:
                    # Use standard loss for other models
                    loss = F.cross_entropy(outputs['logits'], batch['label'].long())
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping
                if self.config.get('max_grad_norm', 0) > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.config['max_grad_norm'])
                
                # Update parameters
                optimizer.step()
                
                total_loss += loss.item()
                
            except Exception as e:
                self.logger.error(f"Training error in batch {batch_idx}: {str(e)}")
                continue
        
        return total_loss / num_batches if num_batches > 0 else 0.0
    
    def validate_epoch(
        self, 
        model: nn.Module, 
        data_loader: DataLoader, 
        device: torch.device
    ) -> Tuple[float, Dict[str, float]]:
        """Validate for one epoch.
        
        Args:
            model: Model to validate
            data_loader: Validation data loader
            device: Device to use
            
        Returns:
            Tuple of (average loss, metrics)
        """
        model.eval()
        total_loss = 0.0
        num_batches = len(data_loader)
        
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch in data_loader:
                # Move batch to device
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                try:
                    # Forward pass
                    outputs = model(
                        patient_input_ids=batch['patient_input_ids'],
                        patient_attention_mask=batch['patient_attention_mask'],
                        treatment_input_ids=batch['treatment_input_ids'],
                        treatment_attention_mask=batch['treatment_attention_mask']
                    )
                    
                    # Compute loss
                    if hasattr(model, 'patient_embeddings') and hasattr(model, 'treatment_embeddings'):
                        loss = self.contrastive_loss(outputs['patient_embeddings'], outputs['treatment_embeddings'])
                    else:
                        loss = F.cross_entropy(outputs['logits'], batch['label'].long())
                    
                    total_loss += loss.item()
                    
                    # Collect predictions for metrics
                    if 'logits' in outputs:
                        predictions = torch.argmax(outputs['logits'], dim=1)
                        all_predictions.extend(predictions.cpu().numpy())
                        all_labels.extend(batch['label'].cpu().numpy())
                
                except Exception as e:
                    self.logger.error(f"Validation error: {str(e)}")
                    continue
        
        # Compute metrics
        metrics = {}
        if all_predictions and all_labels:
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
            
            metrics['accuracy'] = accuracy_score(all_labels, all_predictions)
            metrics['precision'] = precision_score(all_labels, all_predictions, average='weighted', zero_division=0)
            metrics['recall'] = recall_score(all_labels, all_predictions, average='weighted', zero_division=0)
            metrics['f1_score'] = f1_score(all_labels, all_predictions, average='weighted', zero_division=0)
        
        return total_loss / num_batches if num_batches > 0 else 0.0, metrics
