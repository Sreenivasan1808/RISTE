import os
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as transforms
import xml.etree.ElementTree as ET

# ResNet-50 Backbone (pretrained on ImageNet)
class ResNetBackbone(nn.Module):
    def __init__(self):
        super(ResNetBackbone, self).__init__()
        resnet = models.resnet50(pretrained=True)
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

# Custom Dataset class to load images and labels from XML files
class CustomImageDataset(Dataset):
    def __init__(self, image_dir, label_dir, transform=None):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.image_files = os.listdir(image_dir)
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # Load the image
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        image = Image.open(img_path).convert("RGB")
        
        # Apply transformations to the image if specified
        if self.transform:
            image = self.transform(image)

        # Load the corresponding label from XML
        label_filename = self.image_files[idx].replace('.jpg', '.xml')  # Assuming .xml labels
        label_path = os.path.join(self.label_dir, label_filename)
        
        # Parse the XML file
        tree = ET.parse(label_path)
        root = tree.getroot()

        # Example: Let's say we extract a label named 'class' from the XML
        # This part may need adjustments based on your XML structure
        label = int(root.find('class').text)

        return image, torch.tensor(label)

# Image transformations (Resize, Normalize, etc.)
transform = transforms.Compose([
    transforms.Resize((256, 256)),  # Resize images to 256x256
    transforms.ToTensor(),  # Convert image to PyTorch tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize based on ImageNet stats
])

# Create the complete dataset
dataset = CustomImageDataset(image_dir='./train_images/', label_dir='./train_labels/', transform=transform)

# Split dataset into 80% training and 20% testing
train_size = int(0.8 * len(dataset))  # 80% for training
test_size = len(dataset) - train_size  # 20% for testing
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# Create DataLoaders for training and testing
train_loader = DataLoader(train_dataset, batch_size=48, shuffle=True, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=48, shuffle=False, num_workers=4)

# Define loss functions
char_class_loss_fn = nn.CrossEntropyLoss()
char_position_loss_fn = nn.MSELoss()

# Model instantiation
model = I2CModel()

# Optimizer (AdamW with different learning rates for backbone and transformer)
params = [
    {"params": model.cnn_backbone.parameters(), "lr": 1e-5},  # ResNet backbone
    {"params": model.encoder_layers.parameters(), "lr": 1e-4},  # Transformer encoder
    {"params": model.decoder_layers.parameters(), "lr": 1e-4},  # Transformer decoder
]

optimizer = optim.AdamW(params, weight_decay=0.01)


# Example: Iterate through the train_loader and feed the data to the model
for batch_idx, (images, labels) in enumerate(train_loader):
    # Feed the images and labels to the model
    # images -> shape [batch_size, 3, 256, 256]
    # labels -> shape [batch_size]
    optimizer.zero_grad()  # Zero the gradients
    char_class_pred, char_position_pred = model(images, labels)  # Forward pass with the model


    # Print shapes to confirm
    print(f"Batch {batch_idx}:")
    print(f"Image batch shape: {images.shape}")
    print(f"Label batch shape: {labels.shape}")

# Example: Testing loader iteration (if needed)
for batch_idx, (images, labels) in enumerate(test_loader):
    # Test the model here
    char_class_pred, char_position_pred = model(images, labels)













# Example ground truth labels for the character classes and positions
true_char_classes = torch.randint(0, 256, (48, 10))  # Example ground truth (batch size 48, sequence length 10)
true_char_positions = torch.randn(48, 10, 2)  # Example ground truth positions (batch size 48, sequence length 10, 2 coordinates)

# Define loss functions
char_class_loss_fn = nn.CrossEntropyLoss()
char_position_loss_fn = nn.MSELoss()

# Training loop (simplified for one batch)
for epoch in range(num_epochs):  # Assuming `num_epochs` is defined
    optimizer.zero_grad()  # Zero the gradients

    # Forward pass
    char_class_pred, char_position_pred = model(img_input, tgt_input)

    # Compute loss
    # For character class prediction
    char_class_loss = char_class_loss_fn(char_class_pred.view(-1, num_classes), true_char_classes.view(-1))

    # For character position prediction
    char_position_loss = char_position_loss_fn(char_position_pred, true_char_positions)

    # Total loss
    total_loss = char_class_loss + char_position_loss

    # Backward pass (compute gradients)
    total_loss.backward()

    # Update model parameters
    optimizer.step()

    # Print the loss for monitoring
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss.item()}")
