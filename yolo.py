import os
import rospy
import cv2
import math
from sensor_msgs.msg import CompressedImage
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
import numpy as np
from ultralytics import YOLO

class PersonDetector:
    def __init__(self):

        # YOLO model initialization
        yolo_model_path = os.path.join('yolo', 'yolov8m-pose.pt')
        self.yolo_model = YOLO(yolo_model_path)
        
        # Camera intrinsic parameters
        self.camera_matrix = np.array([[335.49106984455955, 0.0, 329.76315999999997],
                                     [0.0, 376.1876816184971, 239.82100277456647],
                                     [0.0, 0.0, 1.0]])
        
        # Camera pose parameters
        self.camera_height = 1.3  # meters above ground
        self.camera_tilt = 0.0    # radians

        rospy.init_node('person_detector', anonymous=True)
        
        # Subscribers
        self.camera_sub = rospy.Subscriber('/vizzy/l_camera/suppressed_image_rect_color_sd/compressed', CompressedImage, self.camera_callback)
        
        # Publisher for detected people coordinates and bounding boxes
        self.people_pub = rospy.Publisher('/detected_people', MarkerArray, queue_size=10)


    def get_ground_coordinates(self, bbox, keypoints, camera_matrix, camera_height, camera_tilt=0.0):
        """
        Get 3D ground coordinates (X, Z) for a detected person.
        
        Args:
            bbox: (x1, y1, x2, y2) bounding box
            keypoints: List of keypoints
            camera_matrix: 3x3 camera intrinsic matrix
            camera_height: Camera height above ground
            camera_tilt: Camera tilt angle
        
        Returns:
            (x, z) coordinates on ground plane in meters, or None if calculation fails
        """
        x1, y1, x2, y2 = bbox
        
        # Method 1: Use foot keypoints if available (ankle keypoints: 15, 16)
        if len(keypoints) > 16:
            left_ankle = keypoints[15] if len(keypoints[15]) == 2 else None
            right_ankle = keypoints[16] if len(keypoints[16]) == 2 else None
            
            if left_ankle and right_ankle:
                # Use average of both ankles
                foot_x = (left_ankle[0] + right_ankle[0]) / 2
                foot_y = (left_ankle[1] + right_ankle[1]) / 2
                ground_pos = self.pixel_to_ground_plane(foot_x, foot_y, camera_matrix, camera_height, camera_tilt)
                if ground_pos:
                    return ground_pos
        
        # Method 2: Use bottom center of bounding box
        bbox_center_x = (x1 + x2) / 2
        bbox_bottom_y = y2
        ground_pos = self.pixel_to_ground_plane(bbox_center_x, bbox_bottom_y, camera_matrix, camera_height, camera_tilt)
        return ground_pos

    def pixel_to_ground_plane(self, pixel_x, pixel_y, camera_matrix, camera_height, camera_tilt=0.0):
        """
        Convert pixel coordinates to ground plane coordinates (X, Z).
        
        Returns:
            (x, z) coordinates in meters, or None if point is above horizon
        """
        # Extract camera parameters
        fx, fy = camera_matrix[0, 0], camera_matrix[1, 1]
        cx, cy = camera_matrix[0, 2], camera_matrix[1, 2]
        
        # Convert to normalized coordinates
        x_norm = (pixel_x - cx) / fx
        y_norm = (pixel_y - cy) / fy
        
        # Apply camera tilt
        cos_tilt = math.cos(camera_tilt)
        sin_tilt = math.sin(camera_tilt)
        
        # Ray direction in camera coordinates
        ray_x = x_norm
        ray_y = y_norm * cos_tilt + sin_tilt  
        ray_z = -y_norm * sin_tilt + cos_tilt
        
        # Check if we can intersect with ground (need forward ray with downward component)
        if ray_y <= 0 or ray_z <= 0:
            return None
        
        # Calculate intersection with ground plane
        t = -camera_height / ray_y
        
        # Ground coordinates (X, Z)
        ground_x = ray_x * t
        ground_z = ray_z * t
        
        return (ground_x, ground_z)

    def process_yolo_detections(self, cv_image):
        """
        Process image with YOLO and return detection results
        
        Returns:
            List of detections with keypoints, bboxes, and ground coordinates
        """
        # Run YOLOv8 pose inference
        results = self.yolo_model(cv_image)
        detections = []
        
        if results[0].boxes is not None:
            for i, (box, keypoints) in enumerate(zip(results[0].boxes, results[0].keypoints)):
                # Get class name and confidence
                class_id = int(box.cls[0])
                class_name = self.yolo_model.names[class_id]
                confidence = float(box.conf[0])
                
                # Only process person detections
                if class_name == 'person':
                    # Get bounding box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    bbox = [x1, y1, x2, y2]
                    
                    # Extract keypoints
                    keypoints_list = []
                    if keypoints is not None:
                        keypoints_xy = keypoints.xy[0].tolist()
                        keypoints_list = keypoints_xy
                    
                    # Get 3D ground position
                    ground_coords = self.get_ground_coordinates(bbox, keypoints_list, self.camera_matrix, self.camera_height, self.camera_tilt)
                    
                    if ground_coords:
                        x, y = ground_coords
                        
                        detection = {
                            'id': i + 1,
                            'confidence': confidence,
                            'bbox': bbox,
                            'keypoints': keypoints_list,
                            'ground_position': {
                                'x': x,
                                'y': y,
                                'z': 0.0  # Always on ground plane
                            },
                            'class_name': class_name,
                            'image_width': cv_image.shape[1],
                            'image_height': cv_image.shape[0]
                        }
                        detections.append(detection)
        print(detection['ground_position']['x'])
        print(detection['ground_position']['y'])
        print(detection['confidence'])
        print("--------------------------------------------------")

        return detections

    def camera_callback(self, msg):
        # Convert CompressedImage data to numpy array
        np_arr = np.frombuffer(msg.data, np.uint8)
        # Decode to OpenCV image
        cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        
        # Process image with YOLO
        detections = self.process_yolo_detections(cv_image)
        
        if not detections:
            rospy.logdebug("No person detections found")
            return
            
        rospy.logdebug(f"Detected {len(detections)} person(s)")

    def spin(self):
        rospy.loginfo('Person Detector spinning...')
        rospy.spin()

if __name__ == '__main__':
    detector = PersonDetector()
    detector.spin()
