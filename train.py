from models.I2C import I2CModel
from models.resnet import ResNetBackbone
from data.dataloader import SCUTLoader
import torchvision.transforms as transforms
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import torch.optim as optim

# Image transformastions (Resize, Normalize, etc.)
transform = transforms.Compose([
    transforms.Resize((256, 256)),  # Resize images to 256x256
    transforms.ToTensor(),  # Convert image to PyTorch tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize based on ImageNet stats
])


# Define loss functions
char_class_loss_fn = nn.CrossEntropyLoss()
char_position_loss_fn = nn.MSELoss()

#Dataset training images path
DATA_IMG_PATH = "data/SCUT-CTW1500/train_images/"
DATA_LABELS_PATH = "data/SCUT-CTW1500/train_labels/"

if __name__ == '__main__':
    import torch.multiprocessing as mp
    mp.freeze_support()

    # Your dataset, dataloader, model, optimizer, etc. initialization code here
    dataset = SCUTLoader(image_dir=DATA_IMG_PATH, label_dir=DATA_LABELS_PATH, transform=transform)
    
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    
    train_loader = DataLoader(train_dataset, batch_size=48, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=48, shuffle=False, num_workers=0)

    model = ResNetBackbone()

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

