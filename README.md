## HistoGPT-LV — for classifying BCC vs cSCC

This repo fine-tunes [HistoGPT](https://github.com/marrlab/HistoGPT). you can run offline on `.h5` feature files.

### Ackowledgment

This repo is for Anne Petzold's project. The model was from [HistoGPT](https://github.com/marrlab/HistoGPT) under Apache License. The dataset was from Anne Petzold at Markus Heppt's lab at Universitätsklinikum Erlangen.

### Setup

```bash
pip install -r requirements.txt
# Required (Perceiver backbone, not on PyPI):
pip install git+https://github.com/lucidrains/flash-perceiver-pytorch.git
```

Tested with Python 3.10 PyTorch 2.2.2 + CUDA 11.8

### What’s here
- `finetune/train_mil_classifier.py`: Cross-validated fine-tuning using PyTorch Lightning
- `finetune/utils/export_inference_model.py`: Export a single `.pth` inference model from a Lightning checkpoint
- `run_best_model.py`: Run inference on one `.h5` file with the exported model
- `models/aggregator.py`: MIL classifier (position embed + FlashPerceiver + gated attention + classifier)
- `models/lightning.py`: Lightning module with metrics (AUROC, F1, balanced accuracy)
- `finetune/datasets/datasets.py`: Loads `.h5` with `features` (N×1024) and `coordinates` (N×3), labels from filename

### 1) How fine-tuning works
- **Data**: Precomputed patch features per slide are stored as `.h5` with datasets `features` and `coordinates`. Labels are inferred from filename tokens:
  - `sbbc`, `ibcc` → basal cell carcinoma (class 0)
  - `pek` → squamous cell carcinoma (class 1)
- **Model** (`models/aggregator.py`): `CancerClassifier(d_input=1024, d_model=1536, num_cls=2)`
  - NaViT position embedding → FlashPerceiver backbone → RMSNorm → gated attention (learns what/how much) → linear classifier
- **Pretrained init**: `finetune/utils/pretrained_loader.py` loads HistoGPT aggregator weights offline, transferring backbone weights (head stays random).
- **Training loop**: `finetune/train_mil_classifier.py`
  - Stratified K-fold CV with `MILDataset` and custom collate
  - BF16 mixed precision, gradient clipping, AdamW + warmup + cosine decay
  - Metrics via TorchMetrics; checkpoint monitors `val_auroc`; early stopping on AUROC
  - After each fold: reload best checkpoint and validate; after CV: compute summary and copy the best fold’s checkpoint dir to `results/best_model`

Run training (offline; requires local files):

```bash
python finetune/train_mil_classifier.py \
  --feat_path /path/to/h5_dir \
  --histogpt_weights /path/to/histogpt_pretrained.pth \
  --biogpt_path /path/to/local/microsoft_biogpt-large \
  --k_folds 5 --max_epochs 50 --max_lr 5e-5 \
  --batch_size 1 --max_patches_per_slide 1000 \
  --output_dir ./results
```

Notes:
- Offline check verifies `histogpt_weights` and BioGPT files: `config.json`, `pytorch_model.bin`, `tokenizer_config.json`.
- Dataloaders feed lists of tensors expected by the aggregator; MIL operates with batch size 1 (one slide per step).

### Export the best checkpoint to an inference-ready `.pth`
Pick the best `.ckpt` inside `./checkpoints/fold_X/` (or the one copied to `./results/best_model/`) and export:

```bash
python -m finetune.utils.export_inference_model \
  --checkpoint ./checkpoints/fold_1/fold_1_best_epoch=E_val_auroc=A.ckpt \
  --histogpt_weights /path/to/histogpt_pretrained.pth \
  --biogpt_path /path/to/local/microsoft_biogpt-large \
  --output histogpt-bcc_cscc.pth
```

The exported file contains the wrapped weights plus `architecture_config` so it can be loaded without Lightning.

### 2) How `run_best_model` works
- Loads the exported `.pth`, reconstructs `CancerClassifier` from `architecture_config`, wraps it in an `InferenceModel` that disables training features and sets eval.
- Loads one `.h5` file, optionally limits patches (for memory), converts features to bfloat16, and runs a forward pass under autocast.
- Produces logits → softmax probabilities → predicted class and human-readable diagnosis, with confidence; optionally saves a JSON or text record.

Note: FlashAttention is not required for inference. The loader sets `use_flash_attn=False`.

Run inference on one slide:

```bash
python run_best_model.py \
  --model histogpt-bcc_cscc.pth \
  --h5_file /path/to/slide_features.h5 \
  --max_patches 1000 \
  --device auto \
  --output_file inference_result.json
```

### Optional: Extracting features from WSIs
If you need to build `.h5` features yourself, see `finetune/helpers/patching.py` (tiling, filtering, and ViT feature extraction). The training/inference here assume `.h5` layout with `features` and `coordinates`.

### Tips
- Run scripts from the repo root so relative imports resolve (e.g., `models.*`).
 - GPU with bfloat16 support is recommended; inference will fall back to CPU if no cuda.

### Troubleshooting
- If you see `ModuleNotFoundError: fused_layer_norm_cuda` on GPU:
  - Run on CPU (`--device cpu`) or
  - Install a fused layer norm CUDA extension (e.g., NVIDIA Apex with fused layer norm). If you prefer, open an issue and we can provide a non-fused fallback.

### Charts (CV results)
You can generate charts from a saved CV JSON (e.g., `finetune/results/cv_results_YYYYMMDD_HHMMSS.json`):

```bash
pip install matplotlib
python -m finetune.utils.plot_cv_results \
  --json /abs/path/to/cv_results_YYYYMMDD_HHMMSS.json \
  --outdir ./plots --format png --dpi 150
```

