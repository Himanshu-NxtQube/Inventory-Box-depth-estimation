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

        # Resize depth map to match input image size
        depth_map = cv2.resize(depth_map, (self.image.width, self.image.height))

        # Normalize depth values between 0 and 1, then scale to 0-255
        depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())
        depth_map = (depth_map * 255).astype("uint8")

if __name__=="__main__":
    image = cv2.imread('/content/drive/MyDrive/3_ground_rack/DJI_0785.JPG')
    print(image.shape)

    image = cv2.resize(image, (1080, 720))
    print(image.shape)

    depth_map = DepthEstimator(image).get_depth_map()
    print(depth_map.shape)

    # Resize depth map to match input image
    # depth_map = cv2.resize(depth_map, (1080, 720))

    # Convert to uint8 before saving
    # depth_map = (depth_map * 255).astype("uint8")
    cv2.imwrite("depth_map.png", depth_map)

