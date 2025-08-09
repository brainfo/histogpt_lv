"""
Utility functions for loading pretrained HistoGPT weights
"""

import os
import torch
import torch.nn as nn
from transformers import BioGptForCausalLM
from models.histogpt import HistoGPTForCausalLM
from models.aggregator import CancerClassifier, OriginalAggregator
import logging

logger = logging.getLogger(__name__)


def load_pretrained_histogpt_weights(model: CancerClassifier, weights_path: str, biogpt_path: str = "../microsoft_biogpt-large") -> bool:
    """
    Load pretrained HistoGPT weights into the model (OFFLINE ONLY)
    
    Args:
        model: Target CancerClassifier model
        weights_path: Path to pretrained HistoGPT weights
        biogpt_path: Path to local BioGPT model directory
        
    Returns:
        bool: True if successful, False otherwise
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Force offline mode
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    
    try:
        # Verify files exist before proceeding
        if not os.path.exists(weights_path):
            logger.error(f"HistoGPT weights not found: {weights_path}")
            return False
            
        if not os.path.exists(biogpt_path):
            logger.error(f"BioGPT model directory not found: {biogpt_path}")
            logger.error("Please ensure BioGPT model is downloaded locally")
            return False
        
        logger.info("Loading HistoGPT model (offline mode)...")
        
        # Create temporary aggregator for loading (matches original structure)
        temp_aggregator = OriginalAggregator(
            d_input=1024,
            d_model=1536,
            num_cls=2,
        )
        
        # Load BioGPT from local directory (offline)
        logger.info(f"Loading BioGPT from local path: {biogpt_path}")
        
        # Load pretrained weights directly
        logger.info(f"Loading HistoGPT weights from: {weights_path}")
        state_dict = torch.load(weights_path, map_location=device)
        
        # Extract aggregator weights from the state dict
        aggregator_state = {}
        for key, value in state_dict.items():
            if key.startswith('histogpt.aggregator.'):
                # Remove the histogpt.aggregator. prefix
                new_key = key[len('histogpt.aggregator.'):]
                aggregator_state[new_key] = value
        
        logger.info("HistoGPT weights loaded successfully")
        
        # Map old aggregator keys to new Aggregator keys
        # The new model has: model, norm, position, head
        # The old model has: model, norm, position, head (same structure)
        
        # Load weights into our model - simple and clean
        model_state = model.state_dict()
        transferred_keys = []
        
        for key in aggregator_state:
            # Skip head/classifier weights - only load backbone features
            if key.startswith('head.') or key.startswith('classifier.'):
                continue
            if key in model_state:
                if model_state[key].shape == aggregator_state[key].shape:
                    model_state[key] = aggregator_state[key]
                    transferred_keys.append(key)
        
        model.load_state_dict(model_state, strict=False)
        logger.info(f"Transferred {len(transferred_keys)} weight tensors from pretrained aggregator")
        
        # Log which weights were transferred
        logger.debug(f"Transferred keys: {transferred_keys}")
        
        return True
        
    except FileNotFoundError as e:
        logger.error(f"Required file not found: {e}")
        logger.error("Ensure all files are downloaded locally for offline training")
        return False
        
    except Exception as e:
        logger.error(f"Failed to load pretrained weights: {e}")
        logger.error("Training from scratch...")
        return False


def freeze_pretrained_layers(model: CancerClassifier, freeze_backbone: bool = True) -> None:
    """
    Freeze pretrained layers for fine-tuning
    
    Args:
        model: CancerClassifier model
        freeze_backbone: Whether to freeze the backbone (everything except head)
    """
    if freeze_backbone:
        # Freeze everything except the classification head
        for name, param in model.named_parameters():
            if not name.startswith('head.'):
                param.requires_grad = False
                
        logger.info("Frozen all layers except classification head")
    else:
        # Unfreeze all layers
        for param in model.parameters():
            param.requires_grad = True
            
        logger.info("All layers unfrozen")


def get_trainable_parameters(model: CancerClassifier) -> dict:
    """
    Get information about trainable parameters
    
    Args:
        model: CancerClassifier model
        
    Returns:
        dict: Information about trainable parameters
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    
    return {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'frozen_parameters': frozen_params,
        'trainable_percentage': (trainable_params / total_params) * 100
    }