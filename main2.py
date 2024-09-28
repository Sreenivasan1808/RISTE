import os
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torchvision.models import ResNet50_Weights
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as transforms
import xml.etree.ElementTree as ET
from load_dataset import CustomImageDataset

# ResNet-50 Backbone (pretrained on ImageNet)
class ResNetBackbone(nn.Module):
    def __init__(self):
        super(ResNetBackbone, self).__init__()
        resnet = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])  # Remove the last FC and AvgPool layers
        self.fc = nn.Linear(2048, 512)  # ResNet-50 outputs 2048-dim feature maps

    def forward(self, x):
        x = self.backbone(x)
        x = torch.flatten(x, 1)  # Flatten the feature maps
        x = self.fc(x)
        return x

# Transformer Encoder Layer
class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super(TransformerEncoder, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, embed_dim)
        )
        self.layer_norm1 = nn.LayerNorm(embed_dim)
        self.layer_norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        # Self-attention layer
        attn_output, _ = self.attention(x, x, x)
        x = self.layer_norm1(x + self.dropout(attn_output))
        
        # Feed-forward network
        ffn_output = self.ffn(x)
        x = self.layer_norm2(x + self.dropout(ffn_output))
        return x

# Transformer Decoder Layer
class TransformerDecoder(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super(TransformerDecoder, self).__init__()
        self.attention1 = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.attention2 = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, embed_dim)
        )
        self.layer_norm1 = nn.LayerNorm(embed_dim)
        self.layer_norm2 = nn.LayerNorm(embed_dim)
        self.layer_norm3 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, tgt, memory):
        # Self-attention over decoder inputs
        tgt2, _ = self.attention1(tgt, tgt, tgt)
        tgt = self.layer_norm1(tgt + self.dropout(tgt2))
        
        # Cross-attention with encoder output
        tgt2, _ = self.attention2(tgt, memory, memory)
        tgt = self.layer_norm2(tgt + self.dropout(tgt2))
        
        # Feed-forward network
        tgt2 = self.ffn(tgt)
        tgt = self.layer_norm3(tgt + self.dropout(tgt2))
        return tgt

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




# Image transformations (Resize, Normalize, etc.)
transform = transforms.Compose([
    transforms.Resize((256, 256)),  # Resize images to 256x256
    transforms.ToTensor(),  # Convert image to PyTorch tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize based on ImageNet stats
])


# Define loss functions
char_class_loss_fn = nn.CrossEntropyLoss()
char_position_loss_fn = nn.MSELoss()


if __name__ == '__main__':
    import torch.multiprocessing as mp
    mp.freeze_support()

    # Your dataset, dataloader, model, optimizer, etc. initialization code here
    dataset = CustomImageDataset(image_dir='./train_images/', label_dir='./train_labels/', transform=transform)
    
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    
    train_loader = DataLoader(train_dataset, batch_size=48, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=48, shuffle=False, num_workers=0)

    model = I2CModel()

    # Optimizer (AdamW with different learning rates for backbone and transformer)
    params = [
        {"params": model.cnn_backbone.parameters(), "lr": 1e-5},  # ResNet backbone
        {"params": model.encoder_layers.parameters(), "lr": 1e-4},  # Transformer encoder
        {"params": model.decoder_layers.parameters(), "lr": 1e-4},  # Transformer decoder
    ]

    optimizer = optim.AdamW(params, weight_decay=0.01)
    # Define loss functions and start the training loop
    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            char_class_pred, char_position_pred = model(images, labels)
            char_class_loss = char_class_loss_fn(char_class_pred.view(-1, 256), true_char_classes.view(-1))
            char_position_loss = char_position_loss_fn(char_position_pred, true_char_positions)
            total_loss = char_class_loss + char_position_loss
            total_loss.backward()
            optimizer.step()
            running_loss += total_loss.item()
        
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader)}")

