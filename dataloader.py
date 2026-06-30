import os
import json
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
from torchvision.datasets import OxfordIIITPet
from PIL import Image
import torch

class ImageNet(Dataset):
    def __init__(self, img_dir, 
                 annotations_file,
                 transform: None
                 ):
        
        self.img_dir = img_dir
        self.transform = transform
        self.load_annotations_clean(annotations_file)
        
        
    def load_annotations_clean(self, annotations_file):
        self.samples = []
        self.all_class_names = []

        
        with open(annotations_file, 'r') as f:
            annotations = json.load(f) # {<folder_name>: [<class_id>, <class_name>]}
            for folder_name, (class_id, class_name) in annotations.items():                
                self.all_class_names.append(class_name.replace('_', ' '))
                folder_path = os.path.join(self.img_dir, folder_name)
                try:
                    for file_name in os.listdir(folder_path):
                        img_path = os.path.abspath(os.path.join(folder_path, file_name))
                        self.samples.append((img_path, class_id, class_name, file_name, folder_name))
                except:
                    continue
        
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, class_id, class_name, image_name, folder_name = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        try:
            class_id = int(class_id)
        except (TypeError, ValueError):
            pass
        return image, class_id, class_name.replace('_', ' '), img_path, folder_name


       
                
                

if __name__ == "__main__":
    data = ImageNet(
        img_dir='/datastore/elo/quanphm/dataset/ImageNet1K/val',
        annotations_file='./imgnet1k_label.json',
    )




        
    
        
    
        
        
        