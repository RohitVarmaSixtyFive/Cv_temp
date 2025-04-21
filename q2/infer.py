import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from ViT import ViTForClassfication
import matplotlib.pyplot as plt
from config import config1, config2, config3, config4, config5, config6, config7


config = config1

transform = transforms.Compose([
    transforms.Resize((config["image_size"], config["image_size"])),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# For displaying images (without normalization)
display_transform = transforms.Compose([
    transforms.Resize((config["image_size"], config["image_size"])),
    transforms.ToTensor(),
])

display_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

# image dimensions from dataset
image_size = display_dataset[0][0].shape[1]
print(f"Image size: {image_size}x{image_size}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

def visualize_predictions(model, display_dataset, num_images=8):
    # Get a batch of test images
    dataiter = iter(torch.utils.data.DataLoader(display_dataset, batch_size=num_images, shuffle=True))
    images, labels = next(dataiter)
    
    # Get model predictions
    model.eval()
    with torch.no_grad():
        outputs, _ = model((images).to(device))
        _, predicted = torch.max(outputs, 1)
    
    # Convert images for display
    def imshow(img):
        img = img.numpy().transpose((1, 2, 0))  # Convert from Tensor image
        return img

    # Plot the images and predictions
    fig = plt.figure(figsize=(15, 6))
    for idx in range(num_images):
        ax = fig.add_subplot(2, num_images//2, idx+1, xticks=[], yticks=[])
        img = images[idx].cpu().numpy().transpose((1, 2, 0))  # Convert from Tensor image
        img = imshow(images[idx])
        ax.imshow(img)
        
        # Add color-coded prediction - green if correct, red if wrong
        pred_class = classes[predicted[idx]]
        true_class = classes[labels[idx]]
        color = 'green' if pred_class == true_class else 'red'
        
        ax.set_title(f'Pred: {pred_class}\nTrue: {true_class}', color=color)
    
    plt.tight_layout()
    plt.savefig('predictions/vit_predictions.png')
    print("Saved predictions visualization to 'vit_predictions.png'")
    
    # Print detailed classification results
    print("\nDetailed Classification Results:")
    print("--------------------------------")
    for idx in range(num_images):
        pred_class = classes[predicted[idx]]
        true_class = classes[labels[idx]]
        result = "CORRECT" if pred_class == true_class else "WRONG"
        print(f"Image {idx+1}: Predicted '{pred_class}', True '{true_class}' - {result}")
    
    # Show class probabilities for the first 3 images
    print("\nClass Probabilities for First 3 Images:")
    print("--------------------------------------")
    
    # Get probabilities using softmax
    probabilities = torch.nn.functional.softmax(outputs[:3], dim=1)
    
    for i in range(3):
        print(f"\nImage {i+1} (True: {classes[labels[i]]}):")
        # Show top 3 probabilities
        top_probs, top_idxs = torch.topk(probabilities[i], 3)
        for j in range(3):
            print(f"  {classes[top_idxs[j]]}: {top_probs[j].item()*100:.2f}%")

# Run the visualization after training
print("\nVisualizing model predictions on test images...")

model = ViTForClassfication(config).to(device)

model.load_state_dict(torch.load('models/config1.pth')['model_state_dict'])

visualize_predictions(model, display_dataset, 8)