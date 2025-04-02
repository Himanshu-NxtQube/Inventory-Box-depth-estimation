# this file for box detection, blue bar and orange bar detections
from ultralytics import YOLO
import cv2
import numpy as np

class detections:
    def __init__(self,image):
        self.blue_bar_detect_model_path = 'models/blue_bar_detection.pt'
        self.orange_bar_detect_model_path = 'models/orange_bar_detection.pt'
        self.box_detect_model_path = 'models/gbox_detection.pt'
        self.image = image
        self.img_height, self.img_width,_ = image.shape # returns heigth, width, no. of color channels

    def __get_blue_bar_boundaries(self,confidence_threshold=0.3,merge_threshold=50):
        blue_bar_detect_model = YOLO(self.blue_bar_detect_model_path)
        blue_bar_detections = blue_bar_detect_model(self.image)

        bbox_centers_x = []
        for box in blue_bar_detections[0].boxes:
            if box.conf.item() >= confidence_threshold:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                bbox_center_x = (x1 + x2) // 2
                bbox_centers_x.append(bbox_center_x)

        bbox_centers_x.sort()
        merged_centers = []
        for center in bbox_centers_x:
            if not merged_centers or center - merged_centers[-1] > merge_threshold:
                merged_centers.append(center)

        if len(merged_centers) >= 2:
            left_roi_x = merged_centers[0]
            right_roi_x = merged_centers[-1]
        elif len(merged_centers) == 1:
            # Determine if the single detection is on the left or right side of the image
            blue_bar_center = merged_centers[0]
            midpoint = self.img_width // 2

            if blue_bar_center < midpoint:  # Detection is on the left side
                left_roi_x = blue_bar_center
                right_roi_x = self.img_width - 1  # Use right edge of image
            else:  # Detection is on the right side
                left_roi_x = 0  # Use left edge of image
                right_roi_x = blue_bar_center
        else:
            left_roi_x = 0
            right_roi_x = self.img_width - 1

        # return int(left_roi_x), int(right_roi_x)
        self.left_roi_x = int(left_roi_x)
        self.right_roi_x = int(right_roi_x)
    
    def __get_orange_bar_boundaries(self,confidence_threshold=0.3):
        orange_bar_detect_model = YOLO(self.orange_bar_detect_model_path)
        orange_bar_detections = orange_bar_detect_model(self.image)

        detections = orange_bar_detections[0].boxes.data  # Format: (x1, y1, x2, y2, confidence, class)

        # Divide the image vertically
        y_mid = self.img_height // 2

        # Filter detections for upper and lower halves
        upper_detections = [(box[1]+box[3])/2 for box in detections if (box[1] <= y_mid and (box[4] > confidence_threshold))]  # y1 <= y_mid
        lower_detections = [(box[1]+box[3])/2 for box in detections if (box[1] >= y_mid and (box[4] > confidence_threshold))]  # y1 >= y_mid

        # Find nearest to upper edge
        if upper_detections:
            upper_roi_y = min(upper_detections)
        else:
            upper_roi_y = 0  # Top edge

        # Find nearest to lower edge
        if lower_detections:
            lower_roi_y = max(lower_detections)
        else:
            lower_roi_y = self.img_height - 1  # Bottom edge

        # Return coordinates
        # return int(upper_roi_y), int(lower_roi_y)
        self.upper_roi_y = int(upper_roi_y)
        self.lower_roi_y = int(lower_roi_y)

    def get_box_boundaries(self,confidence_threshold=0.5):
        self.__get_blue_bar_boundaries()
        self.__get_orange_bar_boundaries()

        
        box_detect_model = YOLO(self.box_detect_model_path)
        box_detections = box_detect_model(self.image)

        box_coordinates = []
        # bbox means bounding box (co-ordinates of detected box)
        for bbox in box_detections[0].boxes.data:
            x1,y1,x2,y2,conf,_ = bbox # x1,y1,x2,y2,confidence,class

            box_mid_x = np.mean((x1,x2))
            box_mid_y = np.mean((y1,y2))

            # print(f'{(x1,y1,x2,y2)=}')
            # print(f'{box_mid_x=}')
            # print(f'{box_mid_y=}')

            if conf < confidence_threshold:
                continue

            # print(f'{self.right_roi_x=}')
            # print(f'{self.left_roi_x=}')
            # print(f'{self.upper_roi_y=}')
            # print(f'{self.lower_roi_y=}')

            # Below conditions doesn't work
            if ((self.right_roi_x < box_mid_x) or (box_mid_x < self.left_roi_x)):
                continue
                 
            if ((self.upper_roi_y > box_mid_y) or (box_mid_y > self.lower_roi_y)):
                continue

            # if ((self.right_roi_x > box_mid_x) and (box_mid_x > self.left_roi_x)):
            #     box_coordinates.append([x1,y1,x2,y2]) 

            # if ((self.upper_roi_y > box_mid_y) and (box_mid_x > self.lower_roi_y)):
            #     box_coordinates.append([x1,y1,x2,y2])

            box_coordinates.append([x1,y1,x2,y2])



        return box_coordinates

if __name__=='__main__':
    image = cv2.imread("test images/DJI_0782.JPG")
    resized_image = cv2.resize(image,(1080,720))

    h,w,_ = resized_image.shape

    obj = detections(resized_image)
    # left_x, right_x = obj.get_blue_bar_boundaries()
    # upper_x, lower_x = obj. get_orange_bar_boundaries()
    box_coordinates = obj.get_box_boundaries()
    print(len(box_coordinates))
    # cv2.line(resized_image, (left_x,0), (left_x,h),(0,0,255),3)
    # cv2.line(resized_image, (right_x,0), (right_x,h),(0,0,255),3)

    # cv2.line(resized_image, (0,upper_x), (w,upper_x),(0,255,0),3)
    # cv2.line(resized_image, (0,lower_x), (w,lower_x),(0,255,0),3)

    for box_coordinate in box_coordinates:
        x1, y1, x2, y2 = box_coordinate
        cv2.rectangle(resized_image,(int(x1),int(y1)),(int(x2),int(y2)),color=(0, 255, 255),thickness=3)

    cv2.imshow("Bar Detections",resized_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()