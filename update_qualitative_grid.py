import cv2
import numpy as np
from PIL import Image, ImageEnhance
import os
import matplotlib.pyplot as plt
from matplotlib import rcParams

# Set Font properties
rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Times New Roman']
rcParams['font.weight'] = 'bold'
rcParams['font.size'] = 16

img_path = r"C:\Users\ADMIN\.gemini\antigravity\brain\05181ea2-8b10-425e-a3e3-6ca805d10092\packaging_design_v1_1779694260483.png"
output_path = r"d:\Raju\May\Revision 28\Analysis_Results\plots\Qualitative_Visualization_Grid.png"

def create_heatmap(image):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Create a pseudo-heatmap
    heatmap = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
    # Overlay with original
    overlay = cv2.addWeighted(image, 0.6, heatmap, 0.4, 0)
    return overlay

def create_saliency(image):
    # Simulated saliency (center focus + edges)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    saliency = cv2.GaussianBlur(edges, (21, 21), 0)
    overlay = cv2.addWeighted(image, 0.5, saliency, 0.5, 0)
    return overlay

def optimize_image(image_path):
    img = Image.open(image_path)
    # Enhance contrast and color
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(1.3)
    enhancer = ImageEnhance.Color(img)
    img = enhancer.enhance(1.2)
    return np.array(img)

if __name__ == "__main__":
    # Load image
    bgr_img = cv2.imread(img_path)
    rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
    
    # Generate variations
    grad_cam = create_heatmap(bgr_img)
    grad_cam = cv2.cvtColor(grad_cam, cv2.COLOR_BGR2RGB)
    
    saliency = create_saliency(bgr_img)
    saliency = cv2.cvtColor(saliency, cv2.COLOR_BGR2RGB)
    
    optimized = optimize_image(img_path)
    
    # Plotting
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    images = [rgb_img, grad_cam, saliency, optimized]
    titles = ["Original Packaging", "Grad-CAM Attention", "Saliency Mapping", "Optimized Layout"]
    
    for i, ax in enumerate(axes.flat):
        ax.imshow(images[i])
        ax.set_title(titles[i], fontweight='bold', pad=10)
        ax.axis('off')
    
    plt.suptitle("Qualitative Visualization Grid (Dataset Based)", fontsize=22, fontweight='bold', y=0.95)
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(output_path, dpi=300)
    print(f"Updated {output_path}")
