from vlm_models import CLIP_Base
from torch.utils.data import DataLoader
from torchvision import transforms
import os


PILLOW_TRANSFORM = {
    'ViT-B/32': transforms.Compose(
        [
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
            transforms.CenterCrop((224, 224)),
            transforms.ToTensor(),

        ]
    ),    

    'ViT-B/16': transforms.Compose(
        [
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
            transforms.CenterCrop((224, 224)),
            transforms.ToTensor(),

        ]
    ),  
    'ViT-L/14': transforms.Compose(
        [
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
            transforms.CenterCrop((224, 224)),
            transforms.ToTensor(),
        ]
    ),   
     
    'ViT-L/14@336px': transforms.Compose(
        [
            transforms.Resize(336, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
            transforms.CenterCrop((336, 336)),
            transforms.ToTensor(),
        ]
    ),          
}



TENSOR_TRANSFORM = transforms.Normalize(
    mean=(0.48145466, 0.4578275, 0.40821073),
    std=(0.26862954, 0.26130258, 0.27577711))

MODEL_INFO = {
    "clip": ["ViT-B/32", 'ViT-B/16', 'ViT-L/14', "ViT-L/14@336px"],
}

def get_models(model_name: str, pretrained: str, device: str = "gpu"):
    if model_name not in MODEL_INFO:
        raise ValueError(f"Model {model_name} is not supported. Available models: {list(MODEL_INFO.keys())}")
    if pretrained not in MODEL_INFO[model_name]:
        raise ValueError(f"Pretrained model {pretrained} is not available for {model_name}. Available pretrained models: {MODEL_INFO[model_name]}")     
    
    if model_name == "clip":
        return CLIP_Base(model_name=pretrained, device=device)
    
    
def get_dataloader(dataset_name, batch_size, transform):
    if dataset_name == "imagenet":
        from dataloader import ImageNet

        # Allow overriding the ImageNet val directory from environment.
        imagenet_val_dir = "/datastore/elo/quanphm/dataset/ImageNet1K/val"
        annotations_file = "./imgnet1k_label.json"

        dataset = ImageNet(
            img_dir=imagenet_val_dir,
            annotations_file=annotations_file,
            transform=transform
        )
        
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        # print(dataset.all_class_names[:5])
        # print(dataset.all_descriptions[:5])
        return dataloader, dataset.all_class_names, None
    
    elif dataset_name == "oxford_pet":
        from dataloader import OxfordPet
        dataset = OxfordPet( 
                            description_file="/data2/elo/khoatn/VLM_Classification/dataset_annotation/oxford_pet_description.json",
                            split="test",
                            transform=transform
                            )
        
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        # print(transform)
        # raise
        
        return dataloader, dataset.dataset.classes, dataset.all_descriptions
    
    elif dataset_name == "cub":
        from dataloader import CUB
        dataset = CUB(
            img_dir="/data/elo/data/CUB/cub2002011/CUB_200_2011/images",
            label_path="/data/elo/data/CUB/cub2002011/CUB_200_2011/image_class_labels.txt",
            img_index_path="/data/elo/data/CUB/cub2002011/CUB_200_2011/images.txt",
            split_path="/data/elo/data/CUB/cub2002011/CUB_200_2011/train_test_split.txt",
            transform=transform
        )   
                
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
        
        return dataloader, dataset.all_classes, None
            
            
            
            
        
        
def get_prompt_template(class_name):
    SYSTEM_PROMPT = """
    You are a helpful assistant that generates descriptive characteristics for various object categories.
    Each description must highlight features that make the category visually distinguishable in images.

    Requirements:
    - Return a Python list of strings.
    - Each list item describes ONE object category.
    - Emphasize visual cues: shape, parts, color/pattern, texture/material, typical context/background, and relative scale.
    - Keep each item short; avoid brand names and subjective words. 
    - Do not include any extra commentary—only the list.

    Output format example, if I want descriptions for "bicycle":
    [
    "two equal-sized wheels with thin tires",
    "a triangular frame, handlebars, and visible chain/gears",
    "often seen on roads or bike paths"
    ]    
    """  
    
    return f"{SYSTEM_PROMPT}\n\nCategories:\n{class_name}"  
