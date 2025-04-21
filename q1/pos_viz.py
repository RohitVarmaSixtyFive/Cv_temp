import math
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse
import sys
from ViT import ViTForClassfication
from config import config1
# import seaborn as sns

def visualize_positional_embedding(model):

    pos_embed = model.module.embedding.position_embedding.position_embeddings.detach().cpu()
    
    print(f"pos_embed shape: ", pos_embed.shape)
    # Remove the class token embedding (first position)
    pos_embed = pos_embed[0, 1:, :]
    print(f"pos_embed shape: ", pos_embed.shape)
    # Normalize the embeddings for cosine similarity
    pos_embed_norm = F.normalize(pos_embed, p=2, dim=1)
    
    similarity = torch.matmul(pos_embed_norm, pos_embed_norm.transpose(0,1)).numpy()
    
    n = pos_embed.shape[0]
    grid_size = int(math.sqrt(n))
    
    plt.figure(figsize=(10, 8))
    # sns.heatmap(similarity, cmap='viridis')
    plt.imshow(similarity, cmap='hot')
    plt.title("Positional Embedding Similarities")
    plt.xlabel("Position Index")
    plt.ylabel("Position Index")
    plt.colorbar(label="Similarity")  # This adds the bar with a label

    plt.savefig(os.path.join(args.output_dir, "pos_embed_similarity.png"), bbox_inches='tight', dpi=200)
    
    # Additionally, visualize as a 2D grid to see spatial relationships
    plt.figure(figsize=(12, 5))
    
    # Plot for each position, its similarity with all other positions
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()
    
    # Sample some positions to visualize (e.g., corners and center)
    sample_positions = [0, grid_size-1, n-grid_size, n-1, n//2]  # top-left, top-right, bottom-left, bottom-right, center
    sample_names = ["Top-Left", "Top-Right", "Bottom-Left", "Bottom-Right", "Center"]
    
    for i, (pos, name) in enumerate(zip(sample_positions, sample_names)):
        if i >= len(axes):
            break
            
        # Reshape similarity vector to 2D grid
        sim_map = similarity[pos].reshape(grid_size, grid_size)
        
        # Plot
        im = axes[i].imshow(sim_map, cmap='hot')
        axes[i].set_title(f"Similarity with {name} Position")
        axes[i].axis('off')
        fig.colorbar(im, ax=axes[i])
    
    # Plot full similarity matrix in the last subplot
    if len(axes) > len(sample_positions):
        im = axes[len(sample_positions)].imshow(similarity, cmap='hot')
        axes[len(sample_positions)].set_title("Full Similarity Matrix")
        fig.colorbar(im, ax=axes[len(sample_positions)])
    
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "pos_embed_grid_similarity.png"), bbox_inches='tight', dpi=200)
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Visualize Positional Embeddings')
    parser.add_argument('--patch_size', default=4, type=int, help='Patch resolution of the model.')
    parser.add_argument('--output_dir', default='plots_1.4.4', help='Path where to save visualizations.')
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    
    
    config1 = {
        "image_size": 32, 
        "patch_size": 4,
        "num_channels": 3,
        "hidden_size": 256,
        "num_attention_heads": 4,
        "intermediate_size": 1024,
        "num_hidden_layers": 8,
        "qkv_bias": True,
        "attention_probs_dropout_prob": 0.1,
        "hidden_dropout_prob": 0.1,
        "initializer_range": 0.02,
        "num_classes": 10, 
        "use_faster_attention": True,
        "positional_embedding": "1d_learned"
    }
    
    config = config1
    model = ViTForClassfication(config=config)
    model.eval()
    model.to("cuda")
    model = torch.nn.DataParallel(model)

    state_dict = torch.load(os.path.join("models", f"ViT-Run-{config['patch_size']}-{config['hidden_size']}-{config['num_attention_heads']}-{config['num_hidden_layers']}-{config['intermediate_size']}-{config['positional_embedding']}.pth"), map_location="cpu")
    model.load_state_dict(state_dict)
    print(f"Loaded weights from {os.path.join('models', f'ViT-Run-{config['patch_size']}-{config['hidden_size']}-{config['num_attention_heads']}-{config['num_hidden_layers']}-{config['intermediate_size']}-{config['positional_embedding']}.pth')}")
    
    # model.eval()
    # model.to(device)
    
    visualize_positional_embedding(model)
    
    print(f"Visualizations saved to {args.output_dir}")