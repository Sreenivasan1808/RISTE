import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
import xml.etree.ElementTree as ET
import numpy as np

class SCUTLoader(Dataset):
    def __init__(self, image_dir, label_dir, transform=None):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.image_files = os.listdir(image_dir)
        self.transform = transform
        self.labels_df = pd.read_csv(self.label_dir)
        print(self.labels_df.head(5))


    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # Load the image
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        image = Image.open(img_path).convert("RGB")
        label = self.labels_df[self.labels_df["img_file_name"] == self.image_files[idx]]
        label = label["label"]
        label = label.iloc[0]


       
        if self.transform:
            image = self.transform(image)
                
        return image, label