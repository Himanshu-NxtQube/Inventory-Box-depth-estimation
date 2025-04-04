import torch
import cv2
from PIL import Image
from transformers import AutoProcessor, AutoModelForDepthEstimation
from transformers import pipeline


class DepthEstimator:
    def __init__(self,model):
        self.device = device="cuda" if torch.cuda.is_available() else "cpu"
        self.model = model
        
        if self.model == "apple_depth_pro":
            self.apple_depth_processor = AutoProcessor.from_pretrained("apple/DepthPro-hf")
            self.apple_depth_model = AutoModelForDepthEstimation.from_pretrained("apple/DepthPro-hf").to(device)
        elif self.model == "depth_anything_v2":
            self.depth_estimator = pipeline(task="depth-estimation", model="depth-anything/Depth-Anything-V2-Large-hf")


    
    def get_depth_map(self,image):
        # Convert BGR to RGB and converting into numpy array
        image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        if self.model == "apple_depth-pro":

            inputs = self.apple_depth_processor(images=image, return_tensors="pt").to(self.device)

            with torch.no_grad():
                outputs = self.apple_depth_model(**inputs)
                depth_map = outputs.predicted_depth.squeeze().cpu().numpy()

            # Resize depth map to match input image size
            depth_map = cv2.resize(depth_map, (image.width, image.height))

            # Normalize depth values between 0 and 1, then scale to 0-255
            depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())
            depth_map = (depth_map * 255).astype("uint8")

            return depth_map
        elif self.model == "depth_anything_v2":
            

            depth_map = depth_estimator(image)['depth']
            return depth_map

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
