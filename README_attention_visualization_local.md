# Gradient Attention Visualization for HistoGPT

This module provides tools to create beautiful gradient attention maps for histopathology images, similar to those shown in research papers. The visualization shows which parts of the whole slide image the model is focusing on when making predictions.

## Features

- **Attention Weight Extraction**: Extract attention weights from the CancerClassifier model
- **Spatial Mapping**: Map attention weights back to spatial coordinates on the original slide
- **Beautiful Visualizations**: Create publication-ready gradient attention heatmaps
- **Multiple Visualization Types**: 
  - Side-by-side comparisons (original vs attention overlay)
  - Pure heatmaps with colorbars
  - Top-attended patch grids
  - Statistical summaries

## Quick Start

### 1. Run the Demo

```bash
# Create a simple demonstration with dummy data
python examples/quick_attention_demo.py
```

This will create visualizations in `./demo_outputs/` showing:
- Attention comparison plots
- Pure heatmaps
- Statistical distributions

### 2. Use with Your Real Data

```bash
python examples/visualize_attention.py \
    --model_path /path/to/your/trained_model.pth \
    --h5_path /path/to/slide_features.h5 \
    --slide_image /path/to/slide_thumbnail.png \
    --output_dir ./attention_outputs \
    --slide_name "sample_slide"
```

## File Structure

```
utils/
├── attention_visualizer.py     # Core visualization classes
examples/
├── visualize_attention.py      # Full-featured script for real data
├── quick_attention_demo.py     # Simple demo with dummy data
```

## Core Classes

### `AttentionExtractor`
Extracts attention weights from the CancerClassifier model during forward pass.

```python
from utils.attention_visualizer import AttentionExtractor

extractor = AttentionExtractor(model)
attention_data = extractor.extract_attention(features, positions)
attention_weights = attention_data['attention_weights'][0].numpy()
```

### `GradientAttentionVisualizer`
Creates beautiful gradient attention visualizations.

```python
from utils.attention_visualizer import GradientAttentionVisualizer

visualizer = GradientAttentionVisualizer(patch_size=512)

# Create spatial heatmap
heatmap = visualizer.create_attention_heatmap(
    attention_weights, coordinates, slide_dimensions
)

# Create overlay on background image
overlay = visualizer.overlay_attention_on_image(
    background_image, heatmap, alpha=0.6, colormap='jet'
)

# Create comparison plot
fig = visualizer.plot_attention_comparison(
    original_image, overlay, attention_weights,
    title="Gradient Attention for Basal Cell Carcinoma"
)
```

## Input Data Requirements

### Model
- Trained `CancerClassifier` model (from `models.aggregator`)
- Model checkpoint file (`.pth`)

### Features
- H5 file containing:
  - `features`: Patch features array (N_patches, feature_dim)
  - `coords`: Patch coordinates array (N_patches, 2) with (x, y) positions

### Optional: Background Image
- Slide thumbnail image (PNG/JPG)
- If not provided, a synthetic background will be created

## Output Files

The visualization tools generate several types of outputs:

1. **Comparison Plots** (`*_attention_comparison.png`)
   - Side-by-side original and attention overlay
   - Similar to research paper figures

2. **Pure Heatmaps** (`*_heatmap_only.png`)
   - Standalone attention heatmaps with colorbars
   - Good for presentations

3. **Top Patches** (`*_top_patches.png`)
   - Grid showing most-attended patches
   - Useful for understanding model focus

4. **Statistics** (`*_attention_stats.txt`)
   - Numerical summaries of attention weights
   - Top-attended patch coordinates

## Customization

### Color Schemes
```python
# Different colormaps for different visualization styles
overlay = visualizer.overlay_attention_on_image(
    background, heatmap, 
    alpha=0.6, 
    colormap='jet'      # Classic rainbow
    # colormap='plasma'   # Purple-pink-yellow
    # colormap='viridis'  # Blue-green-yellow
    # colormap='hot'      # Black-red-yellow-white
)
```

### Attention Types
```python
# Standard attention weights (what the model looks at)
attention_weights = attention_data['attention_weights']

# Combined attention with gates (how much the model trusts what it sees)
combined_attention = attention_data['combined_attention']

# Raw gate scores (trust/confidence values)
gate_scores = attention_data['gate_scores']
```

### Spatial Resolution
```python
# Higher resolution (slower, more detailed)
heatmap = visualizer.create_attention_heatmap(
    attention_weights, coordinates, slide_dims, 
    downsample_factor=8  # Higher resolution
)

# Lower resolution (faster, less detailed)
heatmap = visualizer.create_attention_heatmap(
    attention_weights, coordinates, slide_dims, 
    downsample_factor=32  # Lower resolution
)
```

## Understanding the Visualizations

### Attention Weights
- **High values (red/yellow)**: Areas the model focuses on most
- **Low values (blue/purple)**: Areas the model largely ignores
- Values are normalized across all patches in the slide

### Combined Attention (with Gates)
- Combines attention weights with gate scores
- Shows both "what the model looks at" AND "how much it trusts what it sees"
- Often provides more interpretable results

### Spatial Patterns
- **Clustered attention**: Model focuses on specific regions
- **Scattered attention**: Model considers many different areas
- **Edge attention**: Model might be focusing on tissue boundaries

## Troubleshooting

### Common Issues

1. **"ModuleNotFoundError: No module named 'utils'"**
   ```bash
   # Make sure you're running from the project root
   cd /path/to/histogpt_lv
   python examples/visualize_attention.py ...
   ```

2. **"Model loading failed"**
   - Check that your model checkpoint matches the expected architecture
   - Verify `d_input`, `d_model`, and `num_classes` parameters

3. **"H5 file format not recognized"**
   - Ensure your H5 file contains 'features' and 'coords' datasets
   - Check that coordinates are in (x, y) format

4. **"Attention weights all zero"**
   - Model might not be properly loaded
   - Try running the demo first to verify the code works

### Performance Tips

1. **Large slides**: Use higher `downsample_factor` for initial exploration
2. **Memory issues**: Process slides in smaller chunks
3. **Speed**: Use `use_flash_attn=False` for compatibility if flash attention causes issues

## Integration with Your Workflow

### With Existing Training Code
```python
# After training your model
from utils.attention_visualizer import AttentionExtractor, GradientAttentionVisualizer

# Extract attention for validation slides
extractor = AttentionExtractor(model)
visualizer = GradientAttentionVisualizer()

for slide_data in validation_loader:
    attention_data = extractor.extract_attention(slide_data['features'], slide_data['pos'])
    # Create visualizations...
```

### With Inference Pipeline
```python
# During inference, save attention weights
def inference_with_attention(model, slide_path):
    # Load slide data
    features, coords = load_slide_data(slide_path)
    
    # Get prediction AND attention
    extractor = AttentionExtractor(model)
    attention_data = extractor.extract_attention(features, coords)
    
    # Make prediction
    logits = model(features, coords)
    prediction = torch.softmax(logits, dim=-1)
    
    # Create visualization
    visualizer = GradientAttentionVisualizer()
    # ... create plots
    
    return prediction, attention_data
```

## Citation

If you use this visualization code in your research, please cite:

```bibtex
@software{histogpt_attention_viz,
  title={Gradient Attention Visualization for HistoGPT},
  author={Manuel Tran},
  organization={Helmholtz Munich},
  year={2024}
}
```
