# Copyright (c) Facebook, Inc. and its affiliates.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import sys
import argparse
import cv2
import random
import colorsys
import requests
from io import BytesIO
from ViT import DiffViT
import skimage.io
from skimage.measure import find_contours
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms as pth_transforms
import numpy as np
from PIL import Image
import torch.nn.functional as F

import utils


def apply_mask(image, mask, color, alpha=0.5):
    for c in range(3):
        image[:, :, c] = image[:, :, c] * (1 - alpha * mask) + alpha * mask * color[c] * 255
    return image


def random_colors(N, bright=True):
    """
    Generate random colors.
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.shuffle(colors)
    return colors


def display_instances(image, mask, fname="test", figsize=(5, 5), blur=False, contour=True, alpha=0.5):
    fig = plt.figure(figsize=figsize, frameon=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax = plt.gca()

    N = 1
    mask = mask[None, :, :]
    # Generate random colors
    colors = random_colors(N)

    # Show area outside image boundaries.
    height, width = image.shape[:2]
    margin = 0
    ax.set_ylim(height + margin, -margin)
    ax.set_xlim(-margin, width + margin)
    ax.axis('off')
    masked_image = image.astype(np.uint32).copy()
    for i in range(N):
        color = colors[i]
        _mask = mask[i]
        if blur:
            _mask = cv2.blur(_mask,(10,10))
        # Mask
        masked_image = apply_mask(masked_image, _mask, color, alpha)
        # Mask Polygon
        # Pad to ensure proper polygons for masks that touch image edges.
        if contour:
            padded_mask = np.zeros((_mask.shape[0] + 2, _mask.shape[1] + 2))
            padded_mask[1:-1, 1:-1] = _mask
            contours = find_contours(padded_mask, 0.5)
            for verts in contours:
                # Subtract the padding and flip (y, x) to (x, y)
                verts = np.fliplr(verts) - 1
                p = Polygon(verts, facecolor="none", edgecolor=color)
                ax.add_patch(p)
    ax.imshow(masked_image.astype(np.uint8), aspect='auto')
    fig.savefig(fname)
    print(f"{fname} saved.")
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Visualize Self-Attention maps')

    
    parser.add_argument('--patch_size', default=4, type=int, help='Patch resolution of the model.')
    parser.add_argument('--pretrained_weights', default='', type=str,
        help="Path to pretrained weights to load.")
    parser.add_argument("--checkpoint_key", default="teacher", type=str,
        help='Key to use in the checkpoint (example: "teacher")')
    parser.add_argument("--image_path", default="img.png", type=str, help="Path of the image to load.")
    parser.add_argument("--image_size", default=(224, 224), type=int, nargs="+", help="Resize image.")
    parser.add_argument('--output_dir', default='plots_2.4.2', help='Path where to save visualizations.')
    parser.add_argument("--threshold", type=float, default=None, help="""We visualize masks
        obtained by thresholding the self-attention maps to keep xx% of the mass.""")
    args = parser.parse_args()

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    config={
        "image_size": 32,
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

    model = DiffViT(config=config)

    for p in model.parameters():
        p.requires_grad = False
        
    ckpt = os.path.join(
        "models",
        f"DiffViT-Run-{config['patch_size']}-{config['hidden_size']}-"
        f"{config['num_attention_heads']}-{config['num_hidden_layers']}-"
        f"{config['intermediate_size']}-{config['positional_embedding']}.pth"
    )
    print(f"Loaded weights from {ckpt}")
    
    
    # model = nn.DataParallel(model)        # ← wrap here
    model.to(device).eval()

    state_dict = torch.load(ckpt, map_location=device)
    model.load_state_dict(state_dict)


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

    print(img.shape)

    _, all_attentions = model(img.to(device), True)

    num_layers = len(all_attentions)
    batch_size, nh, seq_len, _ = all_attentions[0].shape

    torchvision.utils.save_image(
        torchvision.utils.make_grid(img, normalize=True, scale_each=True),
        os.path.join(args.output_dir, "output.png")
    )

    for i, attentions in enumerate(all_attentions):  # Iterate over layers
        # Get attention for CLS token to all patches (excluding itself)
        cls_attn = attentions[:, :, 0, 1:]  # shape: [batch, heads, 1, num_patches]

        # Assume square image and square patch grid
        w_featmap = h_featmap = int(cls_attn.shape[-1] ** 0.5)

        # Reshape to [batch, heads, h_featmap, w_featmap]
        cls_attn_reshaped = cls_attn.reshape(batch_size, nh, w_featmap, h_featmap)

        # Interpolate attention map to match image resolution
        attn_maps = F.interpolate(
            cls_attn_reshaped,
            scale_factor=args.patch_size,
            mode="nearest"
        ).cpu().numpy()  # shape: [batch, heads, H, W]

        for j in range(nh):
            fname = os.path.join(args.output_dir, f"layer_{i}_attn-head{j}.png")
            plt.imsave(fname=fname, arr=attn_maps[0, j], format="png")
            print(f"{fname} saved.")