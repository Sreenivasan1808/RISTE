from .resnet import ResNetBackbone
from .transformers import TransformerEncoder, TransformerDecoder
from torch import nn
import torch

# Complete I2C Model with ResNet-50 Backbone and Transformer Encoder-Decoder
class I2CModel(nn.Module):
    def __init__(self, num_heads=8, embed_dim=512, ff_dim=1024, num_encoder_layers=3, num_decoder_layers=1, num_classes=256):
        super(I2CModel, self).__init__()
        self.cnn_backbone = ResNetBackbone()  # ResNet-50 Backbone
        
        
        # Positional Encoding (can be customized)
        self.positional_encoding = nn.Parameter(torch.randn(1, embed_dim))
        
        # Transformer Encoder stack (I2C)
        self.encoder_layers = nn.ModuleList([
            TransformerEncoder(embed_dim, num_heads, ff_dim) for _ in range(num_encoder_layers)
        ])
        
        # Transformer Decoder stack (C2W)
        self.decoder_layers = nn.ModuleList([
            TransformerDecoder(embed_dim, num_heads, ff_dim) for _ in range(num_decoder_layers)
        ])
        
        # Output layers for character class and position
        self.char_class_output = nn.Linear(embed_dim, num_classes)
        self.char_position_output = nn.Linear(embed_dim, 2)  # x, y coordinates

        # Xavier initialization for I2C and C2W weights
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
            elif isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)

    def forward(self, img, tgt):
        # CNN Backbone to extract features from image (ResNet-50)
        img_features = self.cnn_backbone(img)
        
        # Add positional encoding
        img_features = img_features + self.positional_encoding
        
        # Transformer Encoder (I2C)
        for encoder in self.encoder_layers:
            img_features = encoder(img_features.unsqueeze(0)).squeeze(0)
        
        # Transformer Decoder (C2W)
        decoder_output = tgt
        for decoder in self.decoder_layers:
            decoder_output = decoder(decoder_output.unsqueeze(0), img_features.unsqueeze(0)).squeeze(0)
        
        # Output heads for character class and position
        char_class_pred = self.char_class_output(decoder_output)
        char_position_pred = self.char_position_output(decoder_output)
        
        return char_class_pred, char_position_pred

