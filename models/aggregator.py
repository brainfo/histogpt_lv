import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchmetrics.classification import Accuracy
from .perceiver import FlashPerceiver
from .normalization import FusedRMSNorm


class OriginalAggregator(nn.Module):
    """
    Original aggregator structure - intact for loading pretrained weights
    """
    def __init__(self, d_input, d_model, num_cls, use_flash_attn=True):
        super().__init__()
        from .embedder import NaViTEmbedding
        self.position = NaViTEmbedding(
            d_model=d_input,
            max_len=1024,
            patch_size=512,
        )
        self.model = FlashPerceiver(
            d_input=d_input,
            d_model=d_model,
            n_heads=16,
            n_layers=6,
            n_latents=640,
            attn_drop=0.0,
            concat_latents=True,
            use_flash_attn=use_flash_attn,
        )
        self.norm = FusedRMSNorm(
            normalized_shape=d_model,
            eps=1e-05,
            elementwise_affine=True,
        )
        self.head = nn.Linear(d_model, num_cls)

    def forward(self, x, pos):
        x = [self.position(x[i], pos[i]) for i in range(len(x))]
        x = torch.stack(x, dim=0)
        x = self.model(x)
        x = self.norm(x)
        x = x.unsqueeze(1) if x.ndim == 2 else x
        return self.head(x)


class CancerClassifier(nn.Module):
    """
    Cancer classifier - complete model for training
    """
    def __init__(self, d_input, d_model, num_cls=2, use_flash_attn=True):
        super().__init__()
        from .embedder import NaViTEmbedding
        # Position embedding
        self.position = NaViTEmbedding(
            d_model=d_input,
            max_len=1024,
            patch_size=512,
        )
        # FlashPerceiver model
        self.model = FlashPerceiver(
            d_input=d_input,
            d_model=d_model,
            n_heads=16,
            n_layers=6,
            n_latents=640,
            attn_drop=0.0,
            concat_latents=True,
            use_flash_attn=use_flash_attn,
        )
        # Normalization
        self.norm = FusedRMSNorm(
            normalized_shape=d_model,
            eps=1e-05,
            elementwise_affine=True,
        )
        # Gated attention mechanism
        self.attention_V = nn.Linear(d_model, d_model)
        self.attention_U = nn.Linear(d_model, d_model)
        self.attention_w = nn.Linear(d_model, 1)
        
        # Classification head
        self.classifier = nn.Linear(d_model, num_cls)

    def forward(self, x, pos):
        x = [self.position(x[i], pos[i]) for i in range(len(x))]
        x = torch.stack(x, dim=0)
        x = self.model(x)
        x = self.norm(x)
        
        # Gated attention aggregation
        # x shape: (B, 640, d_model)
        
        # Attention branch: learns what to look at
        attention_scores = torch.tanh(self.attention_V(x))  # (B, 640, d_model)
        attention_scores = self.attention_w(attention_scores)  # (B, 640, 1)
        attention_weights = torch.softmax(attention_scores, dim=1)  # (B, 640, 1)
        
        # Gate branch: learns how much to trust
        gate_scores = torch.sigmoid(self.attention_U(x))  # (B, 640, d_model)
        
        # Apply gated attention
        gated_features = gate_scores * x  # (B, 640, d_model)
        aggregated = torch.sum(attention_weights * gated_features, dim=1)  # (B, d_model)
        
        logits = self.classifier(aggregated)  # (B, num_cls)
        return logits
