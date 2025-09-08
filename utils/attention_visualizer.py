"""
Gradient Attention Map Visualizer for HistoGPT
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
from typing import Tuple, Optional, Dict, List, Union
from pathlib import Path
import cv2
from PIL import Image
import h5py


class AttentionExtractor:
    """
    Extracts attention weights from CancerClassifier model during forward pass.
    """
    
    def __init__(self, model: nn.Module):
        """
        Initialize attention extractor.
        
        Args:
            model: CancerClassifier model instance
        """
        self.model = model
        self.attention_weights = None
        self.gate_scores = None
        self.features = None
        self.hooks = []
        self._register_hooks()
    
    def _register_hooks(self):
        """Register forward hooks to capture attention weights."""
        
        def attention_hook(module, input, output):
            # Capture attention weights from the forward pass
            # This hook captures the attention_weights from line 97 in aggregator.py
            if hasattr(self, '_temp_attention_weights'):
                self.attention_weights = self._temp_attention_weights.detach().cpu()
        
        def gate_hook(module, input, output):
            # Capture gate scores
            if hasattr(self, '_temp_gate_scores'):
                self.gate_scores = self._temp_gate_scores.detach().cpu()
        
        # We'll modify the forward pass to capture intermediate values
        # Register hooks on the classifier layer to know when forward pass is complete
        handle = self.model.classifier.register_forward_hook(attention_hook)
        self.hooks.append(handle)
    
    def extract_attention(self, x: torch.Tensor, pos: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Extract attention weights by running a modified forward pass.
        
        Args:
            x: Input features tensor
            pos: Position tensor
            
        Returns:
            Dictionary containing attention weights, gate scores, and features
        """
        self.model.eval()
        
        with torch.no_grad():
            # Run the forward pass up to attention computation
            x_processed = [self.model.position(x[i], pos[i]) for i in range(len(x))]
            x_processed = torch.stack(x_processed, dim=0)
            features = self.model.model(x_processed)
            features = self.model.norm(features)
            
            # Compute attention weights (matching aggregator.py lines 94-97)
            attention_scores = torch.tanh(self.model.attention_V(features))
            attention_scores = self.model.attention_w(attention_scores)
            attention_weights = torch.softmax(attention_scores, dim=1)
            
            # Compute gate scores (matching aggregator.py lines 99-100)
            gate_scores = torch.sigmoid(self.model.attention_U(features))
            
            # Store results (move to CPU for visualization)
            results = {
                'attention_weights': attention_weights.squeeze(-1).cpu(),  # Remove last dimension (B, 640)
                'gate_scores': gate_scores.cpu(),  # (B, 640, d_model)
                'features': features.cpu(),  # (B, 640, d_model)
                'combined_attention': (attention_weights.squeeze(-1) * gate_scores.mean(dim=-1)).cpu()  # (B, 640)
            }
            
        return results
    
    def cleanup(self):
        """Remove registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []


class GradientAttentionVisualizer:
    """
    Creates beautiful gradient attention map visualizations for histopathology images.
    """
    
    def __init__(self, patch_size: int = 512, overlap: float = 0.0):
        """
        Initialize the visualizer.
        
        Args:
            patch_size: Size of patches used for feature extraction
            overlap: Overlap ratio between patches (0.0 to 1.0)
        """
        self.patch_size = patch_size
        self.overlap = overlap
        
        # Create custom colormap for attention visualization
        self.attention_cmap = self._create_attention_colormap()
    
    def _create_attention_colormap(self) -> LinearSegmentedColormap:
        """Create a custom colormap for attention visualization."""
        colors = ['#000080', '#0000FF', '#00FFFF', '#00FF00', '#FFFF00', '#FF8000', '#FF0000']
        n_bins = 256
        cmap = LinearSegmentedColormap.from_list('attention', colors, N=n_bins)
        return cmap
    
    def create_attention_heatmap(
        self,
        attention_weights: np.ndarray,
        coordinates: np.ndarray,
        slide_dimensions: Tuple[int, int],
        downsample_factor: int = 16
    ) -> np.ndarray:
        """
        Create a spatial attention heatmap from attention weights and coordinates.
        
        Args:
            attention_weights: Attention weights array (N_patches,)
            coordinates: Patch coordinates array (N_patches, 2) with (x, y) positions
            slide_dimensions: (width, height) of the original slide
            downsample_factor: Factor to downsample the heatmap for efficiency
            
        Returns:
            Attention heatmap as numpy array
        """
        # Calculate heatmap dimensions
        heatmap_w = slide_dimensions[0] // downsample_factor
        heatmap_h = slide_dimensions[1] // downsample_factor
        
        # Initialize heatmap
        heatmap = np.zeros((heatmap_h, heatmap_w), dtype=np.float32)
        counts = np.zeros((heatmap_h, heatmap_w), dtype=np.float32)
        
        # Map attention weights to spatial locations
        for i, (x, y) in enumerate(coordinates):
            # Convert to heatmap coordinates
            hx = int(x // downsample_factor)
            hy = int(y // downsample_factor)
            
            # Calculate patch size in heatmap coordinates
            patch_w = max(1, self.patch_size // downsample_factor)
            patch_h = max(1, self.patch_size // downsample_factor)
            
            # Ensure coordinates are within bounds
            hx_end = min(heatmap_w, hx + patch_w)
            hy_end = min(heatmap_h, hy + patch_h)
            hx = max(0, hx)
            hy = max(0, hy)
            
            if hx < hx_end and hy < hy_end:
                # Add attention weight to the region
                heatmap[hy:hy_end, hx:hx_end] += attention_weights[i]
                counts[hy:hy_end, hx:hx_end] += 1
        
        # Average overlapping regions
        mask = counts > 0
        heatmap[mask] /= counts[mask]
        
        # Normalize to [0, 1]
        if heatmap.max() > heatmap.min():
            heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
        
        return heatmap
    
    def overlay_attention_on_image(
        self,
        background_image: np.ndarray,
        attention_heatmap: np.ndarray,
        alpha: float = 0.6,
        colormap: str = 'jet'
    ) -> np.ndarray:
        """
        Overlay attention heatmap on background image.
        
        Args:
            background_image: Background image (H, W, 3) in RGB format
            attention_heatmap: Attention heatmap (H, W)
            alpha: Transparency of the heatmap overlay
            colormap: Matplotlib colormap name
            
        Returns:
            Combined image with attention overlay
        """
        # Ensure background image is RGB and normalized
        if background_image.dtype != np.uint8:
            background_image = (background_image * 255).astype(np.uint8)
        
        # Resize heatmap to match background image
        if attention_heatmap.shape != background_image.shape[:2]:
            attention_heatmap = cv2.resize(
                attention_heatmap, 
                (background_image.shape[1], background_image.shape[0]),
                interpolation=cv2.INTER_LINEAR
            )
        
        # Apply colormap to heatmap
        cmap = plt.get_cmap(colormap)
        colored_heatmap = cmap(attention_heatmap)[:, :, :3]  # Remove alpha channel
        colored_heatmap = (colored_heatmap * 255).astype(np.uint8)
        
        # Create overlay
        overlay = cv2.addWeighted(
            background_image, 
            1 - alpha, 
            colored_heatmap, 
            alpha, 
            0
        )
        
        return overlay
    
    def plot_attention_comparison(
        self,
        original_image: np.ndarray,
        attention_overlay: np.ndarray,
        attention_weights: np.ndarray,
        title: str = "Gradient Attention Analysis",
        save_path: Optional[Path] = None,
        dpi: int = 300
    ) -> plt.Figure:
        """
        Create a comparison plot showing original image and attention overlay.
        
        Args:
            original_image: Original histopathology image
            attention_overlay: Image with attention heatmap overlay
            attention_weights: Raw attention weights for statistics
            title: Plot title
            save_path: Optional path to save the figure
            dpi: Resolution for saved figure
            
        Returns:
            Matplotlib figure object
        """
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        
        # Plot original image
        axes[0].imshow(original_image)
        axes[0].set_title("Input whole slide image\n(serial section)", fontsize=12, fontweight='bold')
        axes[0].axis('off')
        
        # Plot attention overlay
        axes[1].imshow(attention_overlay)
        axes[1].set_title("Gradient-attention for\nbasal cell carcinoma", fontsize=12, fontweight='bold')
        axes[1].axis('off')
        
        # Add statistics
        stats_text = f"Attention Statistics:\n"
        stats_text += f"Mean: {attention_weights.mean():.3f}\n"
        stats_text += f"Max: {attention_weights.max():.3f}\n"
        stats_text += f"Min: {attention_weights.min():.3f}\n"
        stats_text += f"Std: {attention_weights.std():.3f}"
        
        fig.text(0.02, 0.02, stats_text, fontsize=8, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        
        plt.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        
        return fig
    
    def create_patch_attention_grid(
        self,
        patches: List[np.ndarray],
        attention_weights: np.ndarray,
        coordinates: np.ndarray,
        grid_size: Tuple[int, int] = (8, 8),
        save_path: Optional[Path] = None
    ) -> plt.Figure:
        """
        Create a grid visualization of top-attended patches.
        
        Args:
            patches: List of patch images
            attention_weights: Attention weights for each patch
            coordinates: Coordinates for each patch
            grid_size: (rows, cols) for the grid
            save_path: Optional path to save the figure
            
        Returns:
            Matplotlib figure object
        """
        # Sort patches by attention weight (descending)
        sorted_indices = np.argsort(attention_weights)[::-1]
        
        n_patches = min(len(patches), grid_size[0] * grid_size[1])
        
        fig, axes = plt.subplots(grid_size[0], grid_size[1], figsize=(12, 12))
        axes = axes.flatten()
        
        for i in range(n_patches):
            idx = sorted_indices[i]
            patch = patches[idx]
            weight = attention_weights[idx]
            coord = coordinates[idx]
            
            axes[i].imshow(patch)
            axes[i].set_title(f'Attn: {weight:.3f}\nPos: ({coord[0]}, {coord[1]})', 
                            fontsize=8)
            axes[i].axis('off')
            
            # Add colored border based on attention weight
            color_intensity = weight / attention_weights.max()
            border_color = plt.cm.Reds(color_intensity)
            for spine in axes[i].spines.values():
                spine.set_edgecolor(border_color)
                spine.set_linewidth(3)
        
        # Hide unused subplots
        for i in range(n_patches, len(axes)):
            axes[i].axis('off')
        
        plt.suptitle("Top Attended Patches", fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig


def load_slide_data(h5_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load features and coordinates from HDF5 file.
    
    Args:
        h5_path: Path to HDF5 file containing features and coordinates
        
    Returns:
        Tuple of (features, coordinates)
    """
    with h5py.File(h5_path, 'r') as f:
        features = f['features'][:]
        coordinates = f['coordinates'][:]
    
    return features, coordinates


def create_thumbnail_from_patches(
    coordinates: np.ndarray,
    patches: List[np.ndarray],
    slide_dimensions: Tuple[int, int],
    patch_size: int = 512,
    thumbnail_size: Tuple[int, int] = (1024, 1024)
) -> np.ndarray:
    """
    Create a thumbnail image from patches and coordinates.
    
    Args:
        coordinates: Patch coordinates (N, 2)
        patches: List of patch images
        slide_dimensions: Original slide dimensions (width, height)
        patch_size: Size of each patch
        thumbnail_size: Target thumbnail size
        
    Returns:
        Thumbnail image as numpy array
    """
    # Calculate scale factors
    scale_x = thumbnail_size[0] / slide_dimensions[0]
    scale_y = thumbnail_size[1] / slide_dimensions[1]
    
    # Create thumbnail canvas
    thumbnail = np.ones((*thumbnail_size[::-1], 3), dtype=np.uint8) * 255
    
    # Calculate scaled patch size
    scaled_patch_size = int(patch_size * min(scale_x, scale_y))
    
    for i, (x, y) in enumerate(coordinates):
        if i < len(patches):
            # Scale coordinates
            thumb_x = int(x * scale_x)
            thumb_y = int(y * scale_y)
            
            # Resize patch
            patch = cv2.resize(patches[i], (scaled_patch_size, scaled_patch_size))
            
            # Place patch on thumbnail
            x_end = min(thumbnail_size[0], thumb_x + scaled_patch_size)
            y_end = min(thumbnail_size[1], thumb_y + scaled_patch_size)
            
            if thumb_x < x_end and thumb_y < y_end:
                thumbnail[thumb_y:y_end, thumb_x:x_end] = patch[:y_end-thumb_y, :x_end-thumb_x]
    
    return thumbnail


# Example usage and utility functions
def example_usage():
    """
    Example of how to use the attention visualization tools.
    """
    print("Example: Gradient Attention Visualization")
    print("=" * 50)
    
    print("""
    # 1. Load your trained model
    from models.aggregator import CancerClassifier
    model = CancerClassifier(d_input=1536, d_model=768, num_cls=2)
    model.load_state_dict(torch.load('path/to/your/model.pth'))
    
    # 2. Load slide data
    features, coordinates = load_slide_data('path/to/slide.h5')
    
    # 3. Extract attention weights
    extractor = AttentionExtractor(model)
    x = torch.tensor(features).unsqueeze(0)  # Add batch dimension
    pos = torch.zeros((1, features.shape[0], 2))  # Dummy positions
    
    attention_data = extractor.extract_attention(x, pos)
    attention_weights = attention_data['attention_weights'][0].numpy()  # Float16 is numpy-compatible
    
    # 4. Create visualizations
    visualizer = GradientAttentionVisualizer(patch_size=512)
    
    # Create spatial heatmap
    slide_dims = (20000, 15000)  # Example slide dimensions
    heatmap = visualizer.create_attention_heatmap(
        attention_weights, coordinates, slide_dims
    )
    
    # Create thumbnail and overlay
    patches = []  # Load your patch images here
    thumbnail = create_thumbnail_from_patches(coordinates, patches, slide_dims)
    
    attention_overlay = visualizer.overlay_attention_on_image(
        thumbnail, heatmap, alpha=0.6, colormap='jet'
    )
    
    # Plot comparison
    fig = visualizer.plot_attention_comparison(
        thumbnail, attention_overlay, attention_weights,
        title="Gradient Attention for Basal Cell Carcinoma",
        save_path=Path("attention_visualization.png")
    )
    
    # Create patch grid
    patch_fig = visualizer.create_patch_attention_grid(
        patches, attention_weights, coordinates,
        save_path=Path("top_patches.png")
    )
    
    # Cleanup
    extractor.cleanup()
    """)


if __name__ == "__main__":
    example_usage()
