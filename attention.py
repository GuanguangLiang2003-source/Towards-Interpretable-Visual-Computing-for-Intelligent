import cv2
import numpy as np

def generate_grad_cam(model, image_tensor):
    """
    Simulated Grad-CAM implementation.
    In a real scenario, this would compute gradients of the class score 
    with respect to the feature maps of the last convolutional layer.
    """
    # Generating a dummy focus region for demonstration
    h, w = image_tensor.shape[2:]
    heatmap = np.zeros((h, w), dtype=np.float32)
    # Focus on logical packaging centers (logo, slogan)
    cv2.circle(heatmap, (w//2, h//2), 60, 1, -1)
    heatmap = cv2.GaussianBlur(heatmap, (51, 51), 0)
    
    # Normalize and colormap
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
    heatmap = (heatmap * 255).astype(np.uint8)
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    return heatmap_color

def overlay_attention(image, heatmap, alpha=0.5):
    """Overlays heatmap on the original image."""
    return cv2.addWeighted(image, 1-alpha, heatmap, alpha, 0)
