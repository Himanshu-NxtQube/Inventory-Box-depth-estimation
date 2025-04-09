import cv2
import torch
import numpy as np
import os
from pathlib import Path
import csv
from ultralytics import YOLO
import matplotlib.pyplot as plt
from PIL import Image
from transformers import AutoProcessor, AutoModelForDepthEstimation

def load_models(device="cuda" if torch.cuda.is_available() else "cpu"):
    """Load the YOLO and DepthPro depth estimation models"""
    # Load the custom YOLO model
    yolo_model = YOLO("/content/drive/MyDrive/Requirement files/Copy of gbox_detection.pt")
    orange_model = YOLO("/content/drive/MyDrive/Requirement files/orange_bar_detection (1).pt")
    blue_model = YOLO("/content/drive/MyDrive/Requirement files/blue_bar_detection.pt")

    # Load the DepthPro model
    processor = AutoProcessor.from_pretrained("apple/DepthPro-hf")
    model = AutoModelForDepthEstimation.from_pretrained("apple/DepthPro-hf").to(device)

    return yolo_model, orange_model, blue_model, model, processor, device

def get_orange_bar_boundaries(image, model, confidence_threshold=0.5):
    """Get upper and lower Y boundaries from detections"""
    image_height, image_width = image.shape[:2]

    # Predict bounding boxes
    results = model(image)
    detections = results[0].boxes.data  # Format: (x1, y1, x2, y2, confidence, class)

    upper_y_coord = 0
    lower_y_coord = image_height

    # Divide the image vertically
    y_mid = image_height // 2

    # Filter detections for upper and lower halves
    upper_detections = [box for box in detections if (box[1] <= y_mid and (box[4] > confidence_threshold))]  # y1 < y_mid
    lower_detections = [box for box in detections if (box[1] >= y_mid and (box[4] > confidence_threshold))]  # y1 >= y_mid

    # Find nearest to upper edge
    if upper_detections:
        upper_y_coord = min(upper_detections, key=lambda box: box[1])[1]
    else:
        upper_y_coord = 0  # Top edge

    # Find nearest to lower edge
    if lower_detections:
        lower_y_coord = max(lower_detections, key=lambda box: box[1])[1]
    else:
        lower_y_coord = image_height  # Bottom edge

    # Return coordinates
    return int(upper_y_coord), int(lower_y_coord)



def get_blue_bar_boundaries(image, model, confidence_threshold=0.3, merge_threshold=50):
    """Get left and right X boundaries from detections"""
    img_height, img_width, _ = image.shape
    results = model(image)
    bbox_centers_x = []
    for box in results[0].boxes:
        if box.conf.item() >= confidence_threshold:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            bbox_center_x = (x1 + x2) // 2
            bbox_centers_x.append(bbox_center_x)

    bbox_centers_x.sort()
    merged_centers = []
    for center in bbox_centers_x:
        if not merged_centers or center - merged_centers[-1] > merge_threshold:
            merged_centers.append(center)

    left_blue = False
    right_blue = False

    if len(merged_centers) >= 2:
        left_roi_x = merged_centers[0]
        left_blue = True
        right_roi_x = merged_centers[-1]
        right_blue = True
    elif len(merged_centers) == 1:
        # Determine if the single detection is on the left or right side of the image
        center_x = merged_centers[0]
        midpoint = img_width // 2

        if center_x < midpoint:  # Detection is on the left side
            left_roi_x = center_x
            left_blue = True
            right_roi_x = img_width - 1  # Use right edge of image
        else:  # Detection is on the right side
            left_roi_x = 0  # Use left edge of image
            right_roi_x = center_x
            right_blue = True
    else:
        # If no bboxes are detected, use left and right edges of the image
        left_roi_x = 0
        right_roi_x = img_width - 1

    return left_roi_x, right_roi_x, left_blue, right_blue

def detect_objects_and_depth(image_path, orange_model, blue_model, yolo_model, depth_model, depth_processor, device, conf_threshold=0.5):
    """Detect gboxes and estimate their depth using DepthPro"""
    # Read the image
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(image_rgb)

    # Get ROI boundaries
    upper_y, lower_y = get_orange_bar_boundaries(image, orange_model)
    left_x, right_x, left_blue, right_blue = get_blue_bar_boundaries(image, blue_model)

    # YOLO detection with confidence threshold
    results = yolo_model(image_rgb, conf=conf_threshold)  # Set confidence threshold # Box detection yolo model

    # DepthPro depth estimation
    inputs = depth_processor(images=pil_image, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = depth_model(**inputs)
        predicted_depth = outputs.predicted_depth

    # Move depth map to CPU and convert to numpy
    depth_map = predicted_depth.squeeze().cpu().numpy()

    # Resize depth map to match image dimensions if needed
    if depth_map.shape != image.shape[:2]:
        depth_map = cv2.resize(depth_map, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_CUBIC)

    # Calculate depth scale factor (approximate conversion to meters)
    depth_scale = 10.0  # This is a placeholder; calibrate for your setup

    # Normalize depth map for visualization
    depth_map_normalized = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min()) #min_max_normalizer
    depth_colormap = cv2.applyColorMap((depth_map_normalized * 255).astype(np.uint8), cv2.COLORMAP_TURBO)

    # Return updated boundaries with flags
    return image, results, depth_map, depth_colormap, depth_scale, (upper_y, lower_y, left_x, right_x, left_blue, right_blue)

def calculate_blue_bar_distance(depth_map, depth_scale, left_x, right_x, left_blue, right_blue):
    """Calculate distance to blue bars using depth estimation"""
    img_height, img_width = depth_map.shape
    distances = []

    # Calculate distance to left blue bar if detected
    if left_blue:
        # Take a sample of depth values near the left blue bar
        left_depth_samples = depth_map[:, max(0, left_x-5):left_x+5]
        if left_depth_samples.size > 0:
            left_median_depth = np.median(left_depth_samples)
            # Convert depth to distance
            if left_median_depth > 0:
                left_distance = 1.0 / (left_median_depth * depth_scale)
                distances.append(left_distance)

    # Calculate distance to right blue bar if detected
    if right_blue:
        # Take a sample of depth values near the right blue bar
        right_depth_samples = depth_map[:, max(0, right_x-5):min(img_width, right_x+5)]
        if right_depth_samples.size > 0:
            right_median_depth = np.median(right_depth_samples)
            # Convert depth to distance
            if right_median_depth > 0:
                right_distance = 1.0 / (right_median_depth * depth_scale)
                distances.append(right_distance)

    # Calculate average distance if both bars detected
    if len(distances) > 0:
        return sum(distances) / len(distances)
    else:
        return None  # No blue bars detected

def calculate_box_distance(box, depth_map, depth_scale):
    """Calculate the distance of a bounding box from the camera in meters,
    where distance is inversely proportional to depth"""
    x1, y1, x2, y2 = box
    box_depth = depth_map[int(y1):int(y2), int(x1):int(x2)]

    # Use median to be more robust to outliers
    median_depth = np.median(box_depth)

    # Avoid division by zero
    if median_depth == 0:
        return float('inf')  # Return infinity for zero depth

    # Calculate distance as inversely proportional to depth
    # You may need to adjust this constant based on your specific requirements
    constant = 1.0  # This could be a calibration factor
    distance_meters = constant / (median_depth * depth_scale)

    return distance_meters

def is_box_in_roi(box, boundaries):
    """Check if a box is within the ROI boundaries"""
    x1, y1, x2, y2 = box
    upper_y, lower_y, left_x, right_x = boundaries

    # Calculate center of the box
    center_x = (x1 + x2) // 2
    center_y = (y1 + y2) // 2

    # Check if center is within boundaries
    return (left_x <= center_x <= right_x) and (upper_y <= center_y <= lower_y)

def categorize_boxes_by_gradient(boxes, distances, boundaries, depth_map, blue_bar_distance):
    """
    Categorize boxes using the depth gradient approach:
    1. Sort boxes by distance
    2. Calculate the gradient (rate of change) between consecutive distances
    3. Identify the point where the gradient increases sharply
    4. Categorize boxes before this point as front and after as back
    """
    # First, filter boxes within ROI
    roi_boxes_with_idx = []
    ignored_boxes = []

    for i, (box, distance) in enumerate(zip(boxes, distances)):
        if is_box_in_roi(box, boundaries):
            roi_boxes_with_idx.append((i, box, distance))
        else:
            ignored_boxes.append((i, box, distance))

    # Handle case with too few boxes
    if len(roi_boxes_with_idx) < 3:
        return [], [], roi_boxes_with_idx, ignored_boxes, {
            'front_cluster_center': None,
            'back_cluster_center': None,
            'max_valid_distance': None,
            'threshold_gradient': None
        }

    # Sort boxes by distance
    roi_boxes_with_idx.sort(key=lambda x: x[2])

    # Extract distances for gradient calculation
    sorted_distances = [box[2] for box in roi_boxes_with_idx]
    print("sorted distances", sorted_distances)

    # Calculate gradients (differences between consecutive distances)
    # gradients = [sorted_distances[i+1] - sorted_distances[i] for i in range(len(sorted_distances)-1)]
    gradients = []
    for i in range(len(sorted_distances)-1):
        gradient = sorted_distances[i+1] - sorted_distances[i]
        gradients.append(gradient)
    print("gradients:", gradients)
    # Calculate average gradient for normalization
    avg_gradient = sum(gradients) / len(gradients) if gradients else 0

    # Identify the point where gradient exceeds threshold
    # (Usually a multiple of the average gradient indicates a significant change)
    # gradient_threshold = max(0.01, avg_gradient * 2)  # At least 0.1 or 2x average gradient
    gradient_threshold  = 0.01
    print("avg_gradient", avg_gradient)
    # print("gradient_threshold", gradient_threshold)
    # Find the first point where gradient exceeds threshold
    split_index = None
    for i, gradient in enumerate(gradients):
        print("gradient", gradient)
        # grad_th = max(0.01, gradient_threshold)
        if gradient > gradient_threshold:
            split_index = i + 1  # +1 because gradient is between i and i+1
            # split_index = i
            break

    # If no significant gradient found, use statistical approach
    # if split_index <= 1:
    #     # Use mean + std as fallback
    #     mean_dist = np.mean(sorted_distances)
    #     std_dist = np.std(sorted_distances)
    #     threshold_distance = mean_dist + 0.5 * std_dist
    #     split_index = next((i for i, d in enumerate(sorted_distances) if d > threshold_distance), len(sorted_distances) // 2)

    # Split boxes into front and back

    if split_index is not None:
      front_boxes = roi_boxes_with_idx[:split_index]  #green

      remaining_boxes = roi_boxes_with_idx[split_index:] #yellow
    else:
      front_boxes = roi_boxes_with_idx
      remaining_boxes = []

    front_threshold = front_boxes[-1][2]
    print("split index", split_index)
    print("front boxes", front_boxes)
    print("remaining boxes", remaining_boxes)

    back_boxes = []
    other_boxes = []

    back_threshold = None
    print("remaining boxes i got", remaining_boxes)
    if remaining_boxes:
      if blue_bar_distance is not None:
        max_valid_distance = 0.10
        print("box[2]:", [(box[2]) for box in remaining_boxes])

        for box in remaining_boxes:
          if abs(box[2] - blue_bar_distance) <= max_valid_distance:
            back_boxes.append(box)
          else:
            other_boxes.append(box)

        if back_boxes:
            back_threshold = back_boxes[-1][2]
      else:
        for box in remaining_boxes:
          if box[2] > 0.47:
            back_boxes.append(box)
          else:
            other_boxes.append(box)

        if back_boxes:
            back_threshold = back_boxes[-1][2]
    else:
        max_valid_distance = float('inf')

    print(f"Split index: {split_index}")

    print(f"Front boxes count: {len(front_boxes)}")
    print(f"Back boxes count: {len(back_boxes)}")
    print(f"Other boxes count: {len(other_boxes)}")

    thresholds = {
            'front_threshold': front_threshold if front_threshold else 'None',
            'back_threshold': back_threshold
    }


    return front_boxes, back_boxes, other_boxes, ignored_boxes, thresholds
def visualize_results(image, results, depth_map, depth_colormap, depth_scale, boundaries, image_name):
    """Visualize detection results with distances and categorized boxes using adaptive thresholds"""
    # Unpack boundaries, now including the blue bar detection flags

    upper_y, lower_y, left_x, right_x, left_blue, right_blue = boundaries

    # Calculate blue bar distance if any blue bars are detected
    blue_bar_distance = None
    if left_blue or right_blue or (left_blue and right_blue):
        blue_bar_distance = calculate_blue_bar_distance(depth_map, depth_scale, left_x, right_x, left_blue, right_blue)
        print(f"Blue bar distance: {blue_bar_distance:.2f} meters")

    roi_image = image.copy()

    # Get the depth colormap for visualization
    depth_vis = depth_colormap.copy()

    # Lists to store data for each box
    box_distances = []
    boxes = []

    # Process each detected box
    if len(results[0].boxes) > 0:
        for i, det in enumerate(results[0].boxes.data):
            x1, y1, x2, y2, conf, cls = det
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            boxes.append([x1, y1, x2, y2])

            # Calculate distance in meters
            distance = calculate_box_distance([x1, y1, x2, y2], depth_map, depth_scale)
            box_distances.append(distance)

    else:
        print("No boxes detected in the image")

    # Now categorize boxes with the blue_bar_distance parameter
    front_boxes, back_boxes, other_boxes, ignored_boxes, thresholds = categorize_boxes_by_gradient(
        boxes, box_distances, (upper_y, lower_y, left_x, right_x), depth_map, blue_bar_distance
    )
    print("front boxes", front_boxes)
    print("back boxes", back_boxes)
    print("other boxes", other_boxes)

    # Draw ROI boundaries on the ROI image
    cv2.line(roi_image, (left_x, 0), (left_x, image.shape[0]), (255, 0, 0), 2)  # Left boundary
    cv2.line(roi_image, (right_x, 0), (right_x, image.shape[0]), (255, 0, 0), 2)  # Right boundary
    cv2.line(roi_image, (0, upper_y), (image.shape[1], upper_y), (0, 165, 255), 2)  # Upper boundary
    cv2.line(roi_image, (0, lower_y), (image.shape[1], lower_y), (0, 165, 255), 2)  # Lower boundary

    # Draw blue bar indicators if detected
    if left_blue:
        cv2.line(roi_image, (left_x, 0), (left_x, image.shape[0]), (255, 0, 0), 4)  # Thicker line for detected blue bar
        cv2.putText(roi_image, "Blue Bar", (left_x + 5, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    if right_blue:
        cv2.line(roi_image, (right_x, 0), (right_x, image.shape[0]), (255, 0, 0), 4)  # Thicker line for detected blue bar
        cv2.putText(roi_image, "Blue Bar", (right_x - 100, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    # Front boxes count multiplied by 2 as requested
    front_boxes_count = len(front_boxes) * 2
    back_boxes_count = len(back_boxes)

    # Store information about all categorized boxes
    categorized_boxes_info = []

    # Draw categorized boxes on the categorized image
    for idx, box, distance in front_boxes:
        x1, y1, x2, y2 = box
        # Draw front boxes in green
        # cv2.rectangle(categorized_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.rectangle(roi_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"Front: {distance:.2f} m"
        # cv2.putText(categorized_image, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        # Store box info for output display
        categorized_boxes_info.append((idx, "Front", distance))

    for idx, box, distance in back_boxes:
        x1, y1, x2, y2 = box
        # Draw back boxes in yellow
        # cv2.rectangle(categorized_image, (x1, y1), (x2, y2), (0, 255, 255), 2)
        cv2.rectangle(roi_image, (x1, y1), (x2, y2), (0, 255, 255), 2)
        label = f"Back: {distance:.2f} m"
        # cv2.putText(categorized_image, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        # Store box info for output display
        categorized_boxes_info.append((idx, "Back", distance))

    for idx, box, distance in other_boxes:
        x1, y1, x2, y2 = box
        # Draw other boxes in red
        # cv2.rectangle(categorized_image, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.rectangle(roi_image, (x1, y1), (x2, y2), (0, 0, 255), 2)
        label = f"Other: {distance:.2f} m"
        # cv2.putText(categorized_image, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # Add count and threshold information to the categorized image
    blue_bar_status = "None"
    if left_blue and right_blue:
        blue_bar_status = "Both"
    elif left_blue:
        blue_bar_status = "Left"
    elif right_blue:
        blue_bar_status = "Right"
    # ---------------------------------------------------------

    info_text = [
        f"ROI Boundaries: X({left_x}-{right_x}), Y({upper_y}-{lower_y})",
        f"Blue Bars: {blue_bar_status}" + (f" - Distance: {blue_bar_distance:.2f}m" if blue_bar_distance else ""),
        f"Front Threshold: {thresholds['front_threshold']:.2f}m" if thresholds['front_threshold'] else "Front Threshold: None",
        f"Back Threshold: {thresholds['back_threshold']:.2f}m" if thresholds['back_threshold'] else "Back Threshold: None",
        f"Front Boxes: {len(front_boxes)} (Count x2 = {front_boxes_count})",
        f"Back Boxes: {back_boxes_count}",
        f"Total Boxes in ROI: {len(front_boxes) + len(back_boxes) + len(other_boxes)}",
        f"Boxes Outside ROI: {len(ignored_boxes)}"
    ]

    # for i, text in enumerate(info_text):
    #     cv2.putText(categorized_image, text, (10, 30 + i*30), cv2.FONT_HERSHEY_SIMPLEX,
    #                 0.7, (255, 255, 255), 2, cv2.LINE_AA)

    # Add count information to the ROI image
    roi_info_text = [
        f"Blue Bars: {blue_bar_status}" + (f" - Dist: {blue_bar_distance:.2f}m" if blue_bar_distance else ""),
        f"Front: {len(front_boxes)} (x2 = {front_boxes_count})"
    ]

    # Only add thresholds if they exist
    if thresholds['front_threshold']:
        roi_info_text[1] = f"Front (≤{thresholds['front_threshold']:.2f}m): {len(front_boxes)} (x2 = {front_boxes_count})"

    if thresholds['back_threshold']:
        roi_info_text.append(f"Back ({thresholds['back_threshold']:.2f}m): {back_boxes_count}")
    else:
        roi_info_text.append(f"Back: {back_boxes_count}")

    roi_info_text.append(f"Total in ROI: {len(front_boxes) + len(back_boxes) + len(other_boxes)}")

    for i, text in enumerate(roi_info_text):
        cv2.putText(roi_image, text, (10, 30 + i*30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255, 255, 255), 2, cv2.LINE_AA)

    # Create combined visualization
    # combined_image = np.hstack((output_image, depth_vis))

    return roi_image, box_distances, boxes, front_boxes_count, back_boxes_count, categorized_boxes_info, thresholds

def process_image(image_path, output_csv_path, roi_images_folder, yolo_model, orange_model, blue_model, depth_model, depth_processor, device):
    """Process a single image and save results to CSV and folder"""
    # Get image name from path
    image_name = os.path.basename(image_path)

    # Process image
    image, results, depth_map, depth_colormap, depth_scale, boundaries = detect_objects_and_depth(
        image_path, orange_model, blue_model, yolo_model, depth_model, depth_processor, device
    )

    # Check if blue bars were detected
    _, _, _, _, left_blue, right_blue = boundaries
    blue_bar_status = ""
    if left_blue and right_blue:
        blue_bar_status = "Both"
    elif left_blue:
        blue_bar_status = "Left"
    elif right_blue:
        blue_bar_status = "Right"
    else:
        blue_bar_status = "None"

    # Visualize results with adaptive thresholds
    roi_image, box_distances, boxes, front_boxes_count, back_boxes_count, categorized_boxes_info, thresholds = visualize_results(
        image, results, depth_map, depth_colormap, depth_scale, boundaries, image_name
    )

    # Save ROI image to folder
    roi_image_path = os.path.join(roi_images_folder, f"roi_{image_name}")
    cv2.imwrite(roi_image_path, roi_image)

    # Write to CSV with threshold information
    with open(output_csv_path, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        # If the file is new, write the header
        if os.path.getsize(output_csv_path) == 0:
            writer.writerow(['Image Name', 'Front Boxes Count', 'Back Boxes Count'])
        writer.writerow([
            image_name,
            front_boxes_count,
            back_boxes_count

        ])

    # Sort categorized boxes for consistent display
    categorized_boxes_info.sort(key=lambda x: x[0])

    # Display box information
    print(f"\n=== Box Information for {image_name} ===")
    print(f"Blue Bars Detected: {blue_bar_status}")
    # print(f"Adaptive Thresholds: Front ≥{thresholds['front_threshold']:.2f}m, Back {thresholds['back_threshold_min']:.2f}m-{thresholds['back_threshold_max']:.2f}m")

    for idx, category, distance in categorized_boxes_info:
        print(f"Box {idx+1} ({category}): Distance: {distance:.2f} meters")

    print(f"\nFront Boxes: {front_boxes_count//2} (x2 = {front_boxes_count})")
    print(f"Back Boxes: {back_boxes_count}")
    print(f"ROI image saved to: {roi_image_path}")
    print(f"Data saved to CSV: {output_csv_path}")

    return roi_image, front_boxes_count, back_boxes_count, categorized_boxes_info, thresholds

def main():
    # Define paths
    # input_dir = "/content/drive/MyDrive/weight updated/Images"
    output_csv_path = "/content/box_counts.csv"
    roi_images_folder = "/content/drive/MyDrive/ground_rackROI_Images"


    # Load models
    yolo_model, orange_model, blue_model, depth_model, depth_processor, device = load_models()

    # Create output directory if it doesn't exist
    os.makedirs(roi_images_folder, exist_ok=True)

    # Initialize CSV file
    if not os.path.exists(output_csv_path):
        with open(output_csv_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Image Name', 'Front Boxes Count', 'Back Boxes Count'])

    image_directory = Path('/content/drive/MyDrive/2_ground_rack')

    image_extensions = ["*.png", "*.jpg", "*.jpeg", "*.bmp", "*.gif", "*.tiff","*JPG"]

    for ext in image_extensions:

        for image_path in image_directory.glob(ext):

            image_path = str(image_path)  # Convert Path object to string for compatibility

            print(f"Processing image: {image_path}")

            processed_results = process_image(image_path, output_csv_path, roi_images_folder, yolo_model, orange_model, blue_model, depth_model, depth_processor, device)

    # Process image

    # Uncomment to process all images in the directory
    # process_directory(input_dir, output_csv_path, roi_images_folder)

if __name__ == "__main__":
    main()