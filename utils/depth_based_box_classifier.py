import numpy as np

class BoxDepthAnalyzer:
    def __init__(self,GYthreashold=20,YRthreashold=65):
        # self.depth_map = depth_map
        self.GYthreashold = GYthreashold
        self.YRthreashold = YRthreashold
        self.categorization_list = []
        self.front_boxes = 0
        self.back_boxes = 0

    def count_boxes(self,depth_map,left_roi_x,right_roi_x,upper_roi_y,lower_roi_y,box_coordinates):
        height, width = depth_map.shape

        left_depth = depth_map[height//2][left_roi_x] if left_roi_x != 0 else -1
        right_depth = depth_map[height//2][right_roi_x] if right_roi_x != width - 1 else -1
        upper_depth = depth_map[upper_roi_y][width//2] if upper_roi_y != 0 else -1
        lower_depth = depth_map[lower_roi_y][width//2] if lower_roi_y != height - 1 else -1

        # Apple-depth-pro model outputs values as per depths, closer objects have higher values and far objects have less values
        min_bar_depth = max(left_depth,right_depth,upper_depth,lower_depth)

        ### Below code for checking bar with min depth - - - -
        # maxidx = np.argmax([left_depth,right_depth,upper_depth,lower_depth])

        # if maxidx == 0:
        #     # print("left blue bar:",left_roi_x)
        #     return (left_roi_x,height//2)
        # elif maxidx == 1:
        #     # print("right blue bar:",right_roi_x)
        #     return (right_roi_x,height//2)
        # elif maxidx == 2:
        #     # print("upper orange bar:",upper_depth)
        #     return (width//2,upper_roi_y)
        # elif maxidx == 3:
        #     # print("lower orange bar:",lower_depth)
        #     return (width//2,lower_roi_y)
        ### - - - - - - - - - - - - - - - - - - - - - - - - - -

        for box_coordinate in box_coordinates:
            x1, y1, x2, y2 = box_coordinate
            box_mid_x = int(np.mean((x1,x2)))
            box_mid_y = int(np.mean((y1,y2)))
            box_depth = int(min_bar_depth)- int(depth_map[box_mid_y][box_mid_x])


            # (box_coordinates, box_depth, depth_level)
            # depth_level == 3 -> far
            # depth_level == 2 -> not far, not close
            # depth_level == 1 -> close
            if box_depth > self.YRthreashold:
                self.categorization_list.append((box_coordinate,box_depth,3))
            elif box_depth > self.GYthreashold:
                self.categorization_list.append((box_coordinate,box_depth,2))
                self.back_boxes+=1
            else:
                self.categorization_list.append((box_coordinate,box_depth,1))
                self.front_boxes+=1

        self.total_boxes = (self.front_boxes * 1) + (self.back_boxes * 2)
        return {"front_box_count": self.front_boxes,
                "back_box_count": self.back_boxes,
                "total_box_count": self.total_boxes}