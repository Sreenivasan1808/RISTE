import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import xml.etree.ElementTree as ET
import numpy as np
from gensim.models import KeyedVectors

class SCUTLoader(Dataset):
    def __init__(self, image_dir, label_dir, transform=None):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.image_files = os.listdir(image_dir)
        self.transform = transform

        # Load the pre-trained Word2Vec model (e.g., Google News vectors)
        print("Loading Word2Vec model...")
        model_path = 'models/word_embeddings/word2vec-google-news-300.model'
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