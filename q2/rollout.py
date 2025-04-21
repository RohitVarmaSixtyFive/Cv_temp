import math
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse
import sys
import cv2
import requests
from io import BytesIO
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms as pth_transforms
import numpy as np
from PIL import Image
from ViT import DiffViT
import utils
from config import config1
import torch.nn.functional as F

def compute_rollout_attention(all_attentions, discard_ratio=0.0):
    """
    all_attentions: list of T tensors, each [batch, heads, seq, seq]
    returns: rollout [batch, seq, seq]
    """
    # 1. stack → [T, batch, heads, seq, seq]
    attn_stack = torch.stack(all_attentions, dim=0)
    # 2. avg heads → [T, batch, seq, seq]
    attn_avg = attn_stack.mean(dim=2)
    
    # 3. add identity & normalize
    T, B, S, _ = attn_avg.shape
    identity = torch.eye(S, device=attn_avg.device).unsqueeze(0).unsqueeze(0)  # [1,1,S,S]
    attn_aug = attn_avg + identity  # broadcast to [T,B,S,S]
    attn_aug = attn_aug / attn_aug.sum(dim=-1, keepdim=True)
    
    # 4. multiply through layers
    rollout = attn_aug[0]
    for i in range(1, T):
        rollout = torch.matmul(attn_aug[i], rollout)
    return rollout  # [batch, seq, seq]

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser('Visualize Self-Attention maps')
    parser.add_argument('--arch', default='vit_small', type=str,
        choices=['vit_tiny', 'vit_small', 'vit_base'], help='Architecture (support only ViT atm).')
    parser.add_argument('--patch_size', default=4, type=int, help='Patch resolution of the model.')
    parser.add_argument('--pretrained_weights', default='', type=str,
        help="Path to pretrained weights to load.")
    parser.add_argument("--checkpoint_key", default="teacher", type=str,
        help='Key to use in the checkpoint (example: "teacher")')
    parser.add_argument("--image_path", default="img.png", type=str, help="Path of the image to load.")
    parser.add_argument("--image_size", default=(224, 224), type=int, nargs="+", help="Resize image.")
    parser.add_argument('--output_dir', default='plots_2.4.3', help='Path where to save visualizations.')
    args = parser.parse_args()

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    config1={
        "image_size": 224,
        "patch_size": 4,
        "num_channels": 3,
        "hidden_dropout_prob": 0.1,
        "attention_probs_dropout_prob": 0.1,
        "initializer_range": 0.02,
        "qkv_bias": True,
        "num_classes": 10,
        "use_differential_attention": True,  
        "positional_embedding": "none",
        "epochs":50,
        "hidden_size": 128,
        "intermediate_size": 512,
        "num_hidden_layers":  8,
        "num_attention_heads": 4
    }


    config = config1
    model = DiffViT(config=config)

    for p in model.parameters():
        p.requires_grad = False
        
    model.eval()
    model.to(device)
    # model = nn.DataParallel(model)


    state_dict = torch.load(os.path.join("models", f"DiffViT-Run-{config['patch_size']}-{config['hidden_size']}-{config['num_attention_heads']}-{config['num_hidden_layers']}-{config['intermediate_size']}-{config['positional_embedding']}.pth"), map_location="cpu")
    model.load_state_dict(state_dict)
    print(f"Loaded weights from {os.path.join("models", f"ViT-Run-{config['patch_size']}-{config['hidden_size']}-{config['num_attention_heads']}-{config['num_hidden_layers']}-{config['intermediate_size']}-{config['positional_embedding']}.pth")}")

    # open image
    if args.image_path is None:
        # user has not specified any image - we use our own image
        print("Please use the `--image_path` argument to indicate the path of the image you wish to visualize.")
        print("Since no image path have been provided, we take the first image in our paper.")
        response = requests.get("https://dl.fbaipublicfiles.com/dino/img.png")
        img = Image.open(BytesIO(response.content))
        img = img.convert('RGB')
    elif os.path.isfile(args.image_path):
        with open(args.image_path, 'rb') as f:
            img = Image.open(f)
            img = img.convert('RGB')
    else:
        print(f"Provided image path {args.image_path} is non valid.")
        sys.exit(1)
        
    CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
    CIFAR_STD  = (0.2470, 0.2435, 0.2616)
    
    transform = pth_transforms.Compose([
        pth_transforms.Resize(args.image_size),
        pth_transforms.ToTensor(),
        pth_transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
    img = transform(img)

    # make the image divisible by the patch size
    w, h = img.shape[1] - img.shape[1] % args.patch_size, img.shape[2] - img.shape[2] % args.patch_size
    img = img[:, :w, :h].unsqueeze(0)

    w_featmap = img.shape[-2] // args.patch_size
    h_featmap = img.shape[-1] // args.patch_size

    # print(img)

    _, all_attentions = model(img.to(device), True)
    # all_attentions: list of length num_layers, each [1, heads, seq, seq]
    rollout = compute_rollout_attention(all_attentions)  # shape [1, seq, seq]

    # Now visualize CLS→patch map:
    cls_rollout = rollout[0, 0, 1:]             # drop CLS-to-CLS, keep CLS-to-patches
    grid_size  = int(math.sqrt(cls_rollout.shape[0]))
    cls_rollout_map = cls_rollout.reshape(grid_size, grid_size)

    # Upsample to image size
    attn_map_upsampled = F.interpolate(
        cls_rollout_map.unsqueeze(0).unsqueeze(0),    # [1,1,gh,gw]
        scale_factor=args.patch_size,
        mode="nearest"
    ).squeeze().cpu().numpy()  # [H, W]

    plt.figure(figsize=(6,6))
    plt.imshow(attn_map_upsampled, cmap='inferno')
    plt.axis('off')
    plt.title("Attention Rollout (CLS → patches)")
    plt.savefig(os.path.join(args.output_dir, "attention_rollout.png"), bbox_inches='tight', dpi=200)
    plt.show()
