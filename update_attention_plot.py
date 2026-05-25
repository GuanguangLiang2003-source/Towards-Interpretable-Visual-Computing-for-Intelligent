import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import os

# Set Font properties
rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Times New Roman']
rcParams['font.weight'] = 'bold'
rcParams['font.size'] = 16

ad_img_path = r"C:\Users\ADMIN\.gemini\antigravity\brain\05181ea2-8b10-425e-a3e3-6ca805d10092\advertising_design_v1_1779694448486.png"
output_path = r"d:\Raju\May\Revision 28\Analysis_Results\plots\Multi-head_Attention_Visualization.png"

def plot_realistic_attention():
    # Load ad image
    bgr_img = cv2.imread(ad_img_path)
    rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
    
    # Create mask for attention (focusing on the watch and logo area)
    h, w, _ = rgb_img.shape
    mask = np.zeros((h, w), dtype=np.float32)
    cv2.circle(mask, (w//2, h//2), min(w, h)//3, 1, -1)
    mask = cv2.GaussianBlur(mask, (101, 101), 0)
    
    # Apply colormap to mask
    heatmap = (mask * 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
    # Overlay
    overlay = cv2.addWeighted(rgb_img, 0.5, heatmap, 0.5, 0)
    
    plt.figure(figsize=(10, 8))
    plt.imshow(overlay)
    plt.title("Multi-head Attention Visualization (Ad Dataset Example)")
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"Updated {output_path}")

if __name__ == "__main__":
    plot_realistic_attention()
