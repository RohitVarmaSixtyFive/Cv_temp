import os
import requests
from PIL import Image
import torch
import clip
from io import BytesIO
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.models import ResNet50_Weights
from torchvision import datasets, transforms
import random
import matplotlib.pyplot as plt    
import numpy as np
from matplotlib import image as mpimg
import torch.nn.functional as F
import time
from tabulate import tabulate
import subprocess
import pandas as pd


"""
I downloaded the synset_words.txt from https://github.com/torch/tutorials/blob/master/7_imagenet_classification/synset_words.txt

[PROMPT] : Write a helper function to read this text file and print both the synset and its group's name
"""

def load_synset_words(synset_word_location):
    """
    Just a helper function to read the synset ID and the names 
    """
    synset_to_class = {}

    with open(synset_word_location, "r") as file:
        for line in file:
            if not line.strip():
                continue
            
            parts = line.strip().split(" ", 1)
            
            if len(parts) == 2:
                synset_to_class[parts[0]] = parts[1]
            else:
                print(f"Skipping invalid line: {line.strip()}")
    
    return synset_to_class


synset_word_location = "synset_words.txt"
synset_words = load_synset_words(synset_word_location)


count = 0 

for synset_id, name in synset_words.items():
    print(f"Synset ID: {synset_id}, Name: {name}")
    count = count + 1
    
    if(count>5):
        break

print(f"Number of labels : {len(synset_words)}")

def load_resnet_50_weight():
    """
    This function loads the pretrained weights for ResNet-50 
    # Ref : https://pytorch.org/vision/main/models/generated/torchvision.models.resnet50.html
    """
    
    resnet50_imagenet = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    resnet50_imagenet.eval()

    return resnet50_imagenet

resnet50_imagenet = load_resnet_50_weight()


def load_clip_model(model):
    """
    This function loads the CLIP and any of its available model 
    # Ref => https://github.com/openai/CLIP?tab=readme-ov-file
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model, preprocess = clip.load(model, device)

    return model, preprocess

clip_RN50 , preprocess = load_clip_model(model="RN50")

labels = list(synset_words.keys())

print(labels[:5])

def classify_with_clip(model, preprocess, device, image_path, class_labels):
    """
    This function classifies an image with CLIP and returns the top-5 predictions.

    The code is exactly from https://github.com/openai/CLIP?tab=readme-ov-file
    """
    
    image = Image.open(image_path).convert('RGB')
    image_input = preprocess(image).unsqueeze(0).to(device)
    
    text_inputs = torch.cat([clip.tokenize(f"a photo of a {label}") for label in class_labels]).to(device)
    
    with torch.no_grad():
        image_features = model.encode_image(image_input)
        text_features = model.encode_text(text_inputs)

    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)

    similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1).squeeze(0)
    
    values, indices = similarity.topk(5)
    
    predictions = [(class_labels[idx.item()], values[i].item()) for i, idx in enumerate(indices)]
    
    return predictions

"""
I downloaded the mini imagenet dataset from https://www.kaggle.com/datasets/ifigotin/imagenetmini-1000/discussion/284032
"""

device = "cuda" if torch.cuda.is_available() else "cpu"

train_dir = "../../data/external/3/imagenet-mini/train"

imagenet_data = datasets.ImageFolder(root=train_dir, transform=preprocess)

synset_ids = imagenet_data.classes  # these are like n017839 .. and not human readable -> so convert to human labels 


human_readable_labels = []

for synset_id in synset_ids:
    if synset_id in synset_words:
        human_readable_labels.append(synset_words[synset_id].split(",")[0])

print(human_readable_labels[:5])

print(f"\nNumber of labels = {len(human_readable_labels)}")

NUMBER_IMAGES = 5 

for idx in range(NUMBER_IMAGES):

    synset_id = synset_ids[idx]  
    synset_dir = os.path.join(train_dir, synset_id)  
    
    image_files = [f for f in os.listdir(synset_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    random_image_file = random.choice(image_files)
    image_path = os.path.join(synset_dir, random_image_file)

    true_label = human_readable_labels[idx]
    
    predictions = classify_with_clip(clip_RN50, preprocess, device, image_path, human_readable_labels)

    print(f"\nTrue label: {true_label} (synset ID: {synset_id})")

    img = mpimg.imread(image_path)
    plt.figure(figsize=(6,6))
    plt.imshow(img)
    plt.axis('off') 
    plt.title(f"True: {true_label}\nTop-5 Predictions")
    plt.show()

    for label, score in predictions:
        print(f"Label: {label}, Score: {score:.4f}")

"""
This is a standard pre-processing done for imagenet dataset and ref = https://stackoverflow.com/questions/67185623/image-net-preprocessing-using-torch-transforms
"""

resnet_preprocess = transforms.Compose([

    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),

])

def classify_with_resnet50(model, image_path, imagenet_labels , top_k=5):
    """
    This function classifies an image with ResNet50 and returns the top-k predictions.
    """

    image = Image.open(image_path).convert('RGB')
    input_tensor = resnet_preprocess(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(input_tensor)
    
    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    
    values, indices = torch.topk(probabilities, top_k)
    
    predictions = [(imagenet_labels[idx.item()], values[i].item()) for i, idx in enumerate(indices)]

    return predictions

resnet50_imagenet = resnet50_imagenet.to(device)

imagenet_labels = human_readable_labels


NUMBER_IMAGES = 5

for idx in range(NUMBER_IMAGES):

    synset_id = synset_ids[idx]
    synset_dir = os.path.join(train_dir, synset_id)

    image_files = [f for f in os.listdir(synset_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    random_image_file = random.choice(image_files)
    image_path = os.path.join(synset_dir, random_image_file)
    
    true_label = human_readable_labels[idx]
    
    resnet_predictions = classify_with_resnet50(resnet50_imagenet, image_path, imagenet_labels)

    print(f"\nClassifying with ResNet50")
    print(f"True label: {true_label} (synset ID: {synset_id})")
    
    img = mpimg.imread(image_path)
    plt.figure(figsize=(6,6))
    plt.imshow(img)
    plt.axis('off')  
    plt.title(f"True: {true_label}\nResNet50 Top-5 Predictions")
    plt.show()
    
    for label, score in resnet_predictions:
        print(f"Label: {label}, Score: {score:.4f}")
        
def get_predictions(resnet50_model, clip_model, preprocess , image_path, imagenet_labels, device):
    """
    Function to get predictions from both ResNet-50 and CLIP models.
    """

    resnet_prediction = classify_with_resnet50(resnet50_model, image_path, imagenet_labels)

    clip_prediction = classify_with_clip(clip_model, preprocess, device, image_path, imagenet_labels)
    
    return resnet_prediction, clip_prediction

"""
Class 1 : Fruits
"""

folder_path = "../../data/interim/3/clip-yes-resnet-no/1"

for filename in os.listdir(folder_path):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        image_path = os.path.join(folder_path, filename)

        # Load image
        img = Image.open(image_path).convert("RGB")
        plt.imshow(img)
        plt.title(f"Image: {filename}")
        plt.axis('off')
        plt.show()

        resnet_prediction, clip_prediction = get_predictions(resnet50_imagenet, clip_RN50, preprocess, image_path, imagenet_labels, device)

        print(f"== Predictions for: {filename} ==")

        for label, score in resnet_prediction:
            print(f"ResNet Label: {label}, Score: {score:.4f}")

        print()

        for label, score in clip_prediction:
            print(f"CLIP Label: {label}, Score: {score:.4f}")
      

        print("\n" + "="*100 + "\n")

synset_to_label = dict(zip(synset_ids, human_readable_labels))

import os
from PIL import Image
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

# These will store the mismatched image paths
in_resnet_not_clip = []
not_resnet_in_clip = []

# Root directory with synset-style folders
root_folder = "../../data/external/3/imagenet-mini/val"

# Get list of folders first
synset_folders = [f for f in os.listdir(root_folder) if os.path.isdir(os.path.join(root_folder, f))]

for synset_id in tqdm(synset_folders, desc="Processing synsets"):

    class_path = os.path.join(root_folder, synset_id)

    # Get the human-readable label
    true_label = synset_to_label.get(synset_id, "").lower()


    if not true_label:
        continue  # Skip if mapping not found

    image_files = [f for f in os.listdir(class_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    for filename in tqdm(image_files, leave=False, desc=f"{synset_id}", 
                         postfix=lambda: {
                             "in_resnet_not_clip": len(in_resnet_not_clip),
                             "not_resnet_in_clip": len(not_resnet_in_clip)
                         }):
        image_path = os.path.join(class_path, filename)

        # Run predictions
        resnet_pred, clip_pred = get_predictions(
            resnet50_imagenet, clip_RN50, preprocess, image_path, imagenet_labels, device
        )

        # Get just the labels
        resnet_labels = [label.lower() for label, _ in resnet_pred]
        clip_labels = [label.lower() for label, _ in clip_pred]


        in_resnet = any(true_label in pred_label for pred_label in resnet_labels)
        in_clip = any(true_label in pred_label for pred_label in clip_labels)

        if in_resnet and not in_clip:
            in_resnet_not_clip.append(image_path)
            print(f"[Updated] in_resnet_not_clip: {len(in_resnet_not_clip)}")
        elif in_clip and not in_resnet:
            not_resnet_in_clip.append(image_path)
            print(f"[Updated] not_resnet_in_clip: {len(not_resnet_in_clip)}")


# Save results to file
with open("in_resnet_not_clip.txt", "w") as f:
    for path in in_resnet_not_clip:
        f.write(f"{path}\n")

with open("not_resnet_in_clip.txt", "w") as f:
    for path in not_resnet_in_clip:
        f.write(f"{path}\n")

print(f"✅ Done! {len(in_resnet_not_clip)} in ResNet but not CLIP")
print(f"✅ Done! {len(not_resnet_in_clip)} in CLIP but not ResNet")