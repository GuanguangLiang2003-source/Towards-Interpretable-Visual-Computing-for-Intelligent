import torch
import torch.nn as nn
import torchvision.models as models

class CNNViTHybrid(nn.Module):
    """
    Proposed CNN-ViT Hybrid model for Aesthetic quality evaluation.
    Combines ResNet50 for local features and a Transformer for global context.
    """
    def __init__(self, num_classes=1):
        super(CNNViTHybrid, self).__init__()
        # CNN Branch
        resnet = models.resnet50(pretrained=True)
        self.cnn_features = nn.Sequential(*list(resnet.children())[:-2])
        
        # Transformer Branch
        self.vit_adapter = nn.Conv2d(2048, 768, kernel_size=1)
        encoder_layer = nn.TransformerEncoderLayer(d_model=768, nhead=8)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=4)
        
        # Output Head
        self.fc = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Local features
        feat = self.cnn_features(x)  # (B, 2048, 7, 7)
        feat = self.vit_adapter(feat) # (B, 768, 7, 7)
        
        # Flatten for transformer (B, SeqLen, D)
        B, C, H, W = feat.shape
        feat = feat.view(B, C, -1).permute(2, 0, 1) # (H*W, B, 768)
        
        # Transformer encoding
        encoded = self.transformer_encoder(feat) # (49, B, 768)
        
        # Global average pooling
        out = encoded.mean(dim=0)
        return self.fc(out)
