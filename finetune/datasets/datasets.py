import os
import h5py
import torch
import numpy as np
import json
# import pickle  # No longer needed
import logging
import hashlib
from torch.utils.data import Dataset
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union

logger = logging.getLogger(__name__)

class MILDataset(Dataset):
    """
    Multiple instance learning dataset with filename-based label extraction

    Parameters
    ----------
    feat_path: path to the features
    max_patches_per_slide: maximum number of patches per slide to prevent memory issues

    Outputs
    ----------
    feats: feature vectors
    coords: coordinate vectors
    label: disease encodings (extracted from filename)
    """
    def __init__(self, feat_path: str, transform=None, max_patches_per_slide: int = 1000, use_cache: bool = True, seed: Optional[int] = None):
        """
        Initialize with feat_path, transform, and patch limiting
        Labels are extracted from filenames
        """
        self.transform = transform
        self.feat_path = Path(feat_path)
        self.max_patches_per_slide = max_patches_per_slide
        self.seed = seed
        # Cache deterministic subsample indices per file to keep them stable across epochs
        self._subsample_indices_cache: Dict[str, np.ndarray] = {}
        
        # Disease patterns for label extraction
        self.disease_patterns = {
            'sbbc': 'basal cell carcinoma',
            'ibcc': 'basal cell carcinoma', 
            'pek': 'squamous cell carcinoma'
        }
        
        # Create label mapping for binary classification
        self.label_to_idx = {
            'basal cell carcinoma': 0,
            'squamous cell carcinoma': 1,
            'unknown_pathology': -1  # Will be filtered out
        }
        
        # Try to load from cache first
        if self._load_from_cache(use_cache):
            logger.info(f"Loaded dataset from cache: {len(self.file_list)} files")
        else:
            # Cache miss - scan files and create cache
            self._scan_files()
            if use_cache:
                self._save_to_cache()
            logger.info(f"Scanned and cached dataset: {len(self.file_list)} files")
        
        logger.info(f"Class distribution: {np.bincount(self.label_list)}")
    
    def _load_from_cache(self, use_cache: bool) -> bool:
        """Try to load dataset from cache"""
        if not use_cache:
            return False
            
        cache_file = self.feat_path / ".dataset_cache.json"
        
        # Check if cache exists and is newer than directory
        if (cache_file.exists() and 
            cache_file.stat().st_mtime > self.feat_path.stat().st_mtime):
            
            try:
                with open(cache_file) as f:
                    cache = json.load(f)
                    self.file_list = cache['file_list']
                    self.label_list = cache['label_list']
                    return True
            except Exception as e:
                logger.warning(f"Cache corrupted, rescanning: {e}")
                return False
        
        return False
    
    def _save_to_cache(self):
        """Save dataset to cache"""
        cache_file = self.feat_path / ".dataset_cache.json"
        try:
            cache = {
                'file_list': self.file_list,
                'label_list': self.label_list
            }
            with open(cache_file, 'w') as f:
                json.dump(cache, f)
        except Exception as e:
            logger.warning(f"Failed to save cache: {e}")
    
    def _scan_files(self):
        """Scan files and extract labels"""
        self.file_list = []
        self.label_list = []
        
        try:
            file_list = [f for f in os.listdir(self.feat_path) if f.endswith('.h5')]
            
            # Extract labels from filenames and filter valid ones
            for filename in file_list:
                diagnosis = self._extract_diagnosis_from_filename(filename)
                if diagnosis != 'unknown_pathology':
                    self.file_list.append(filename)
                    self.label_list.append(self.label_to_idx[diagnosis])
                else:
                    logger.warning(f"Unknown diagnosis for file: {filename}")
            
        except Exception as e:
            logger.error(f"Failed to initialize dataset: {e}")
            raise

    def _extract_diagnosis_from_filename(self, filename: str) -> str:
        """
        Extract diagnosis from filename using same logic as slide_level_dataset
        """
        filename = filename.lower()
        
        # Direct pattern matching
        for file_code, diagnosis in self.disease_patterns.items():
            if file_code.lower() in filename:
                return diagnosis
        
        # Split by underscore and dash, then check each part
        parts = filename.replace('-', '_').split('_')
        for part in parts:
            for file_code, diagnosis in self.disease_patterns.items():
                if file_code.lower() in part:
                    return diagnosis
        
        return "unknown_pathology"
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """Return ``(feats, coords, label)`` with optional patch limiting."""
        filename = self.file_list[idx]
        path = self.feat_path / filename

        try:
            with h5py.File(path, 'r') as file:
                try:
                    feats = file['features'][()]
                    coords = file['coordinates'][()]
                    
                    # Apply patch limiting if needed
                    if len(feats) > self.max_patches_per_slide:
                        # Deterministic subsampling if seed provided; otherwise use global RNG
                        if self.seed is not None:
                            # Build a stable per-file seed by mixing dataset seed with filename hash
                            file_hash = int(hashlib.sha256(filename.encode('utf-8')).hexdigest()[:16], 16) % (2**32)
                            combined_seed = (int(self.seed) + file_hash) % (2**32 - 1)
                            # Cache per-file indices to keep them constant across epochs
                            if filename not in self._subsample_indices_cache:
                                rng = np.random.default_rng(combined_seed)
                                chosen = rng.choice(len(feats), self.max_patches_per_slide, replace=False)
                                self._subsample_indices_cache[filename] = np.sort(chosen)
                            indices = self._subsample_indices_cache[filename]
                        else:
                            indices = np.random.choice(
                                len(feats),
                                self.max_patches_per_slide,
                                replace=False
                            )
                            indices = np.sort(indices)
                        feats = feats[indices]
                        coords = coords[indices]
                    
                    # Convert to tensors
                    feats = torch.tensor(feats, dtype=torch.float32)
                    coords = torch.tensor(coords, dtype=torch.float32)

                except Exception as e:
                    logger.warning(f"Failed to load data from {path}: {e}")
                    feats = torch.tensor([])
                    coords = torch.tensor([])

        except Exception as e:
            logger.error(f"Failed to open file {path}: {e}")
            feats = torch.tensor([])
            coords = torch.tensor([])

        label = self.label_list[idx]

        return feats, coords, label

    def __len__(self) -> int:
        """
        Return length of dataset
        """
        return len(self.file_list)


def mil_collate_fn(batch: List[Tuple[torch.Tensor, torch.Tensor, int]]):
    """Custom collate function for ``MILDataset`` including coordinates."""
    feats, coords, labels = zip(*batch)
    print(f"[COLLATE] Raw batch size: {len(batch)}")
    print(f"[COLLATE] Raw labels: {labels}")
    
    # Convert to tensor while preserving batch dimension
    labels = torch.tensor(list(labels), dtype=torch.long)
    print(f"[COLLATE] Labels tensor shape: {labels.shape}, dtype: {labels.dtype}")

    # unwrap for common ``batch_size=1`` case
    if len(feats) == 1:
        print(f"[COLLATE] Single batch - unwrapping")
        print(f"[COLLATE] Before unwrap - feats: {type(feats[0])}, coords: {type(coords[0])}, labels: {labels.shape}")
        feats = feats[0]
        coords = coords[0]
        # Keep labels with batch dimension: don't unwrap labels to maintain [1] shape
        print(f"[COLLATE] After unwrap - feats: {feats.shape if hasattr(feats, 'shape') else type(feats)}")
        print(f"[COLLATE] After unwrap - coords: {coords.shape if hasattr(coords, 'shape') else type(coords)}")
        print(f"[COLLATE] After unwrap - labels: {labels.shape if hasattr(labels, 'shape') else labels} (type: {type(labels)})")

    return feats, coords, labels