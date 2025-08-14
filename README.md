## HistoGPT-LV — for classifying BCC vs cSCC

This repo fine-tunes [HistoGPT](https://github.com/marrlab/HistoGPT). you can run offline on `.h5` feature files with [the fine‑tuned model](https://drive.google.com/file/d/136LGjBabMBJ0Q3mMuEtv_ZIPQiisDSBU/view?usp=drive_link).

### Ackowledgment

This repo is for Anne Petzold's project. The model was from [HistoGPT](https://github.com/marrlab/HistoGPT) under Apache License. The dataset was from Anne Petzold at Markus Heppt's lab at Universitätsklinikum Erlangen.

### Setup

```bash
pip install -r requirements.txt
# Required (Perceiver backbone, not on PyPI):
pip install git+https://github.com/lucidrains/flash-perceiver-pytorch.git
```

Tested with Python 3.10 PyTorch 2.2.2 + CUDA 11.8

### `run_best_model`
- Loads the [`.pth`](https://drive.google.com/file/d/136LGjBabMBJ0Q3mMuEtv_ZIPQiisDSBU/view?usp=drive_link), reconstructs `CancerClassifier` from `architecture_config`, wraps it in an `InferenceModel` that disables training features and sets eval.
- Loads one `.h5` file, optionally limits patches (for memory), converts features to bfloat16, and runs a forward pass under autocast.
- Produces logits → softmax probabilities → predicted class and human-readable diagnosis, with confidence.

Run inference on one slide:

```bash
python run_best_model.py \
  --model histogpt-bcc_cscc.pth \
  --h5_file /path/to/slide_features.h5 \
  --max_patches 1000 \
  --device auto \
  --output_file inference_result.json
```

### Notes on fine-tuning
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

#### Export the best checkpoint to an inference-ready `.pth`
Pick the best `.ckpt` inside `./checkpoints/fold_X/` (or the one copied to `./results/best_model/`) and export:

```bash
python -m finetune.utils.export_inference_model \
  --checkpoint ./checkpoints/fold_1/fold_1_best_epoch=E_val_auroc=A.ckpt \
  --histogpt_weights /path/to/histogpt_pretrained.pth \
  --biogpt_path /path/to/local/microsoft_biogpt-large \
  --output histogpt-bcc_cscc.pth
```