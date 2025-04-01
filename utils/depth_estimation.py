import torch
import cv2
from PIL import Image
from transformers import AutoProcessor, AutoModelForDepthEstimation


class DepthEstimator:
    def __init__(self,image):
        self.device = device="cuda" if torch.cuda.is_available() else "cpu"

        self.processor = AutoProcessor.from_pretrained("apple/DepthPro-hf")
        self.model = AutoModelForDepthEstimation.from_pretrained("apple/DepthPro-hf").to(device)
    
        # Convert BGR to RGB and converting into numpy array
        self.image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    def get_depth_map(self):
        inputs = self.processor(images=self.image, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            depth_map = outputs.predicted_depth.squeeze().cpu().numpy()

        depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())

        return depth_map


