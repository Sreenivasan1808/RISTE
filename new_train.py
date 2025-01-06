from models.I2C2W import I2C2W
from data.dataloader import SCUTLoader
import torchvision.transforms as transforms
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import torch.optim as optim
from utils import encode_labels_with_positions
from test import test_model
import torch

# Image transformations (Resize, Normalize, etc.)
transform = transforms.Compose([
    transforms.Resize((128, 128)),  # Resize images to 128x128
    transforms.ToTensor(),  # Convert image to PyTorch tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize based on ImageNet stats
])

# Define loss functions
char_class_loss_fn = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padded indices
pos_loss_fn = nn.CrossEntropyLoss(ignore_index=-1)  # Ignore padded positions
max_length = 170

# Dataset training images path
DATA_IMG_PATH = "./data/SCUT-CTW1500/cropped_train_images2/"
DATA_LABELS_PATH = "./data/SCUT-CTW1500/processed_labels2.csv"

if __name__ == '__main__':
    import torch.multiprocessing as mp
    mp.freeze_support()

    # Your dataset, dataloader, model, optimizer, etc. initialization code here
    dataset = SCUTLoader(image_dir=DATA_IMG_PATH, label_dir=DATA_LABELS_PATH, transform=transform)

    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)

    model = I2C2W(d_model=256, num_heads=8, num_encoder_layers=3, num_decoder_layers=1, d_ff=2048, height=128, width=128, max_length=max_length, dropout=0.1)

    # Optimizer (AdamW with different learning rates for backbone and transformer)
    params = [
        {"params": model.i2c.cnn_backbone.parameters(), "lr": 1e-5},  # ResNet backbone
        {"params": model.i2c.encoder_layers.parameters(), "lr": 1e-4},  # Transformer encoder
        {"params": model.i2c.decoder_layers.parameters(), "lr": 1e-4},  # Transformer decoder
        {"params": model.c2w.decoder_layers.parameters(), "lr": 1e-4},  # C2W decoder
    ]

    # Move model to device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = optim.AdamW(params, weight_decay=0.01)
    # Training Loop
    num_epochs = 10

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        total_char_correct = 0
        total_pos_correct = 0
        total_samples = 0

        for batch_idx, (images, labels) in enumerate(train_loader):
            images = images.to(device)

            # Encode labels with positions
            encoded_labels = encode_labels_with_positions(labels, max_length=max_length)
            encoded_labels['char_classes'] = encoded_labels['char_classes'].to(device)
            encoded_labels['char_positions'] = encoded_labels['char_positions'].to(device)

            # Forward pass
            optimizer.zero_grad()
            char_logits, pos_logits, word_logits = model(images)

            # Compute total loss
            total_loss = model.compute_loss(char_logits, pos_logits, word_logits, encoded_labels, encoded_labels['char_positions'])

            # Backpropagation and optimization
            total_loss.backward()
            optimizer.step()

            running_loss += total_loss.item()

            # Calculate accuracy
            char_preds = torch.argmax(char_logits, dim=-1)  # [batch_size, max_length]
            pos_preds = torch.argmax(pos_logits, dim=-1)    # [batch_size, max_length]

            # Mask out padded positions
            char_mask = (encoded_labels['char_classes'] != 0)  # Ignore padded characters
            pos_mask = (encoded_labels['char_positions'] != -1)  # Ignore padded positions

            # Count correct predictions
            char_correct = (char_preds[char_mask] == encoded_labels['char_classes'][char_mask]).sum().item()
            pos_correct = (pos_preds[pos_mask] == encoded_labels['char_positions'][pos_mask]).sum().item()

            total_char_correct += char_correct
            total_pos_correct += pos_correct
            total_samples += char_mask.sum().item()

        # Calculate average loss and accuracy for the epoch
        avg_loss = running_loss / len(train_loader)
        char_accuracy = total_char_correct / total_samples
        pos_accuracy = total_pos_correct / total_samples

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss}, Character Accuracy: {char_accuracy * 100:.2f}%, Position Accuracy: {pos_accuracy * 100:.2f}%")

    torch.save(model, "./results/i2c2w_saved.pt")
    test_model(test_dataloader=test_loader)
