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
import gensim.downloader as api
from gensim.models import Word2Vec, KeyedVectors
import numpy as np

class CustomImageDataset(Dataset):
    def __init__(self, image_dir, label_dir, transform=None):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.image_files = os.listdir(image_dir)
        self.transform = transform

        # Load the pre-trained Word2Vec model (e.g., Google News vectors)
        print("Loading Word2Vec model...")
        model_path = './pretrained_models/word2vec-google-news-300.model'
        self.word2vec_model = KeyedVectors.load(model_path)
        #self.word2vec_model = api.load('word2vec-google-news-300')  # 300-dimensional vectors
        print("Word2Vec model loaded successfully.")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # Load the image
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        image = Image.open(img_path).convert("RGB")

        # Apply transformations to the image if specified
        

        # Load the corresponding label from XML
        label_filename = self.image_files[idx].replace('.jpg', '.xml')
        label_path = os.path.join(self.label_dir, label_filename)
        tree = ET.parse(label_path)
        root = tree.getroot()

        # Iterate over the boxes in the XML
        for box in root.findall(".//box"):
            # Get the box coordinates
            left = int(box.attrib['left'])
            top = int(box.attrib['top'])
            width = int(box.attrib['width'])
            height = int(box.attrib['height'])

            # Crop the image using the coordinates
            cropped_image = image.crop((left, top, left + width, top + height))
            # cropped_image.show() 
            if self.transform:
                cropped_image = self.transform(cropped_image)

            # Extract the label from XML and convert to Word2Vec embedding
            label_str = box.find('label').text

            # Convert the label into a Word2Vec vector (handle out-of-vocabulary cases)
            if label_str in self.word2vec_model:
                label_embedding = self.word2vec_model[label_str]
            else:
                # Handle unknown labels by returning zeros
                label_embedding = np.zeros(self.word2vec_model.vector_size)

            # Convert the embedding to a tensor
            label_embedding_tensor = torch.tensor(label_embedding, dtype=torch.float)

            # Save the cropped image (for visualization/debugging purposes)
            # cropped_image.save(f"cropped_{label_str}.jpg")
            # print(f"Cropped image saved for label: {label_str}")

            # Example of showing the last cropped image (optional)
            

        # Return the cropped image and label embedding tensor
        return cropped_image, label_embedding_tensor

if __name__ == '__main__':
    import torch.multiprocessing as mp
    mp.freeze_support()
    # Image transformations (Resize, Normalize, etc.)
    transform = transforms.Compose([
        transforms.Resize((256, 256)),  # Resize images to 256x256
        transforms.ToTensor(),  # Convert image to PyTorch tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize based on ImageNet stats
    ])

    # Your dataset, dataloader, model, optimizer, etc. initialization code here
    dataset = CustomImageDataset(image_dir='./train_images/', label_dir='./train_labels/', transform=transform)

    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=48, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=48, shuffle=False, num_workers=0)

    for batch_idx, (images, labels) in enumerate(train_loader):
        for img, lab in zip(images, labels):   
            print(lab)

