import os 
import cv2
import pandas as pd
import numpy as np
from utils.detections import Detections
from utils.depth_estimation import DepthEstimator
from utils.depth_based_box_classifier import BoxDepthAnalyzer


OUTPUT_DIR = 'output/'
IMAGE_DIR = 'test images/'


def plot_boxes_with_depths(image,categorization_list,output_path):
	for entry in categorization_list:
		box_coordinate, box_depth, depth_level = entry
		x1, y1, x2, y2 = box_coordinate
		cv2.putText(image, str(box_depth), (int(np.mean((x1, x2))), int(np.mean((y1, y2)))), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
		
		if depth_level == 1:
			cv2.rectangle(image,(x1,y1),(x2,y2),(0,255,0),2)
		elif depth_level == 2:
			cv2.rectangle(image,(x1,y1),(x2,y2),(0,255,255),2)
		else:
			cv2.rectangle(image,(x1,y1),(x2,y2),(0,0,255),2)


	cv2.imwrite(output_path,image)

if __name__ == '__main__':
	os.makedirs(OUTPUT_DIR,exist_ok=True)
	box_counts_csv = open(OUTPUT_DIR+'/'+'box counts.csv','w')
	box_counts_csv.write("Image Path,Front Boxes,Back Boxes,Total Boxes\n")
	box_counts_csv.close()
	depth_estimator = DepthEstimator()
	detector = Detections()
	depth_analyzer = BoxDepthAnalyzer()

	for filename in os.listdir(IMAGE_DIR):
		box_counts_csv = open(OUTPUT_DIR+'/'+'box counts.csv','a')
		image_path = os.path.join(IMAGE_DIR, filename)
		image = cv2.imread(image_path)
		resized_image = cv2.resize(image,(1080,720))

		# depth_map = cv2.imread(f"test images/Depth Maps/depth_map{i}.png",cv2.IMREAD_GRAYSCALE)
		

		depth_map = depth_estimator.get_depth_map(resized_image)
		box_coordinates = detector.get_box_boundaries(resized_image)

		box_counts = depth_analyzer.count_boxes(depth_map = depth_map,
												left_roi_x = detector.left_roi_x,
												right_roi_x = detector.right_roi_x,
												upper_roi_y = detector.upper_roi_y,
												lower_roi_y = detector.lower_roi_y,
												box_coordinates = box_coordinates)
		# optional
		plot_boxes_with_depths(resized_image,
							depth_analyzer.categorization_list,
							os.path.join(OUTPUT_DIR,'Depth-boxes-'+filename))
		
		box_counts_csv.write(f"{image_path},{box_counts['front_box_count']},{box_counts['back_box_count']},{box_counts['total_box_count']}\n")

		box_counts_csv.close()
	