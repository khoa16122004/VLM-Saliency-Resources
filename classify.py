import torch
from util import get_models, get_dataloader
from tqdm import tqdm 
import os
import json
from torchvision import transforms

from util import PILLOW_TRANSFORM, TENSOR_TRANSFORM


@torch.no_grad()
def classification_simple(model, test_loader, all_class_names):
    texts = [f"{name}" for name in all_class_names]
    text_feats = model.encode_text(texts)                         # [M, D]
    total, correct = 0, 0
    correct_by_class = {name: [] for name in all_class_names}

    idx_counter = 0
    for (image, class_id, class_name, img_path, _) in tqdm(test_loader):
        image = image.cuda()
        class_id = torch.as_tensor(class_id, device=image.device, dtype=torch.long)
        img_feats = model.encode_image(image)                     # [B, D]
        logits = img_feats @ text_feats.T                         # [B, M]
        pred = logits.argmax(dim=-1)

        mask = (pred == class_id)                                 # [B]

        if isinstance(class_name, str):
            class_name = [class_name]
        if isinstance(img_path, str):
            img_path = [img_path]

        for i in mask.nonzero(as_tuple=True)[0].tolist():
            gt_class = class_name[i].replace('_', ' ')
            if gt_class not in correct_by_class:
                correct_by_class[gt_class] = []
            correct_by_class[gt_class].append(os.path.abspath(img_path[i]))

        correct += mask.sum().item()
        total += image.size(0)
        idx_counter += image.size(0)
        # print(class_id)
        # print(pred)
        # print(total)
        # print(mask)
        # print(mask.sum())
        # input()
    acc = correct / total
    return correct, acc, correct_by_class
    
@torch.no_grad()
def classification_sampling(model, test_loader, all_descriptions):
    
    # caption preprocess
    for key, desc in all_descriptions.items():
        deses = []
        for des_ in desc:
            text = f"{key}. {des_}"
            deses.append(text)
        all_descriptions[key] = deses
    all_texts = all_descriptions.values()
    
    # mean vector
    class_protos = []
    for desc_list in all_texts:
        dfeats = model.encode_text(desc_list)                     # [K, D]
        proto = dfeats.mean(dim=0)  
        class_protos.append(proto)
    text_class_features = torch.stack(class_protos, dim=0)        # [M, D]
    
    total, correct = 0, 0
    correct_indices = [] 

    idx_counter = 0
    for (image, class_id, _, _, _) in tqdm(test_loader):
        image = image.cuda()
        class_id = class_id.cuda()

        img_feats = model.encode_image(image)     # [B, D]

        logits = img_feats @ text_class_features.T                # [B, M]
        pred = logits.argmax(dim=-1)
        
        mask = (pred == class_id)                                 # [B]
        if mask.any():
            batch_idxs = mask.nonzero(as_tuple=True)[0].tolist()
            correct_indices.extend([idx_counter + i for i in batch_idxs])
        
        correct += (pred == class_id).sum().item()
        total += image.size(0)

    acc = correct / total
    return correct, acc, correct_indices


    

def main(args):
    transform = transforms.Compose([
        PILLOW_TRANSFORM[args.pretrained],
        TENSOR_TRANSFORM
    ])
    
    model = get_models(args.model_name, args.pretrained, device="cuda")
    dataloader, all_class_names, all_descriptions = get_dataloader(args.dataset_name, args.batch_size, transform)
    if args.classification_method == "simple":
        correct, acc, correct_by_class = classification_simple(model, dataloader, all_class_names)
    elif args.classification_method == "sampling":
        correct, acc, correct_indices = classification_sampling(model, dataloader, all_descriptions)
        correct_by_class = {name: [] for name in all_class_names}
    else:
        raise ValueError(f"Unknown classification method: {args.classification_method}")
    
    
    print("Acc(Top1): ", acc)
    print("number of correct: ", correct)
    
    os.makedirs(args.right_dir, exist_ok=True)
    pretrained_ = args.pretrained.replace("/", "")
    output_path = os.path.join(
        args.right_dir,
        f"{args.model_name}_{pretrained_}_{args.dataset_name}_{args.classification_method}.json"
    )
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(correct_by_class, f, ensure_ascii=False, indent=2)
            
    print("The right sample was written in the path: ", output_path)
    
    
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="VLM Classification")
    parser.add_argument("--model_name", type=str, default="clip", help="Model name")
    parser.add_argument("--pretrained", type=str, default="ViT-B/32", help="Pretrained model name")
    parser.add_argument("--dataset_name", type=str, default="imagenet", help="Dataset name", choices=["imagenet", "oxford_pet", "cub"])
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for dataloader")
    parser.add_argument("--classification_method", type=str, choices=["simple", "sampling"], default="simple",
                        help="Classification method to use")
    parser.add_argument("--run_file", type=str, default=None, help="Run file for sampling specific samples")
    parser.add_argument("--right_dir", type=str, default='right', help="Run file for sampling specific samples")

    args = parser.parse_args()
    main(args)
