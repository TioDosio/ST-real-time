#!/usr/bin/env python3

import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image, CompressedImage
from ultralytics import YOLO
import torch

class YOLOv8PoseDetector:
    def __init__(self):
        rospy.init_node('yolov8_pose_detector', anonymous=True)
        
        # Load YOLOv8 pose model
        rospy.loginfo("Loading YOLOv8-pose model...")
        self.model = YOLO('yolov8n-pose.pt')  # You can use yolov8s-pose.pt, yolov8m-pose.pt, etc.
        rospy.loginfo("Model loaded successfully!")
        
        # Set device (GPU if available)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        rospy.loginfo(f"Using device: {self.device}")
        
        # Publishers
        self.image_pub = rospy.Publisher('/yolov8_pose/annotated_image', Image, queue_size=1)
        self.compressed_pub = rospy.Publisher('/yolov8_pose/annotated_image/compressed', CompressedImage, queue_size=1)
        
        # Subscriber
        self.image_sub = rospy.Subscriber('/vizzy/l_camera/suppressed_image_rect_color_sd/compressed', 
                                         CompressedImage, self.image_callback, queue_size=1)
        
        # Parameters
        self.confidence_threshold = rospy.get_param('~confidence_threshold', 0.5)
        self.publish_compressed = rospy.get_param('~publish_compressed', True)
        
        rospy.loginfo("YOLOv8 Pose Detector initialized!")
        rospy.loginfo("Subscribed to: /vizzy/l_camera/suppressed_image_rect_color_sd/compressed")
        rospy.loginfo("Publishing to: /yolov8_pose/annotated_image")
        
    def image_callback(self, msg):
        try:
            # Convert compressed image to CV2 format
            np_arr = np.frombuffer(msg.data, np.uint8)
            cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            
            if cv_image is None:
                rospy.logerr("Failed to decode compressed image")
                return
            
            # Run YOLOv8 pose detection
            results = self.model(cv_image, device=self.device, conf=self.confidence_threshold)
            
            # Annotate image
            annotated_image = self.annotate_image(cv_image, results[0])
            
            # Publish annotated image
            self.publish_results(annotated_image, msg.header)
            
        except Exception as e:
            rospy.logerr(f"Error in image callback: {str(e)}")
    
    def annotate_image(self, image, result):
        """
        Annotate image with bounding boxes and pose keypoints
        """
        annotated_img = image.copy()
        
        # Get boxes and keypoints
        boxes = result.boxes
        keypoints = result.keypoints
        
        if boxes is not None:
            for i, box in enumerate(boxes):
                # Get box coordinates and confidence
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                conf = box.conf[0].cpu().numpy()
                cls = box.cls[0].cpu().numpy().astype(int)
                
                # Draw bounding box
                cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Add label
                label = f"Person {conf:.2f}"
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                cv2.rectangle(annotated_img, (x1, y1 - label_size[1] - 5), 
                             (x1 + label_size[0], y1), (0, 255, 0), -1)
                cv2.putText(annotated_img, label, (x1, y1 - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        # Draw keypoints if available
        if keypoints is not None:
            for keypoint_set in keypoints:
                self.draw_keypoints(annotated_img, keypoint_set.xy[0].cpu().numpy(), 
                                  keypoint_set.conf[0].cpu().numpy())
        
        return annotated_img
    
    def draw_keypoints(self, image, keypoints, confidences):
        """
        Draw pose keypoints and skeleton
        """
        # COCO pose keypoint connections
        skeleton = [
            [16, 14], [14, 12], [17, 15], [15, 13], [12, 13],
            [6, 12], [7, 13], [6, 7], [6, 8], [7, 9],
            [8, 10], [9, 11], [2, 3], [1, 2], [1, 3],
            [2, 4], [3, 5], [4, 6], [5, 7]
        ]
        
        # Colors for different keypoints
        keypoint_colors = [
            (255, 0, 0),    # nose
            (255, 85, 0),   # eyes
            (255, 170, 0),
            (255, 255, 0),  # ears  
            (170, 255, 0),
            (85, 255, 0),   # shoulders
            (0, 255, 0),
            (0, 255, 85),   # elbows
            (0, 255, 170),
            (0, 255, 255),  # wrists
            (0, 170, 255),
            (0, 85, 255),   # hips
            (0, 0, 255),
            (85, 0, 255),   # knees
            (170, 0, 255),
            (255, 0, 255),  # ankles
            (255, 0, 170)
        ]
        
        # Draw keypoints
        for i, (x, y) in enumerate(keypoints):
            if confidences[i] > 0.5:  # Only draw confident keypoints
                color = keypoint_colors[i] if i < len(keypoint_colors) else (255, 255, 255)
                cv2.circle(image, (int(x), int(y)), 5, color, -1)
                cv2.circle(image, (int(x), int(y)), 5, (0, 0, 0), 1)
        
        # Draw skeleton connections
        for connection in skeleton:
            kpt_a, kpt_b = connection
            if (kpt_a-1 < len(keypoints) and kpt_b-1 < len(keypoints) and 
                confidences[kpt_a-1] > 0.5 and confidences[kpt_b-1] > 0.5):
                
                x1, y1 = keypoints[kpt_a-1]
                x2, y2 = keypoints[kpt_b-1]
                cv2.line(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), 2)
    
    def cv2_to_imgmsg(self, cv_image, encoding="bgr8"):
        """
        Convert OpenCV image to ROS Image message without cv_bridge
        """
        img_msg = Image()
        img_msg.height = cv_image.shape[0]
        img_msg.width = cv_image.shape[1]
        img_msg.encoding = encoding
        img_msg.is_bigendian = 0
        img_msg.step = cv_image.shape[1] * cv_image.shape[2]
        img_msg.data = cv_image.tobytes()
        return img_msg
    
    def publish_results(self, annotated_image, header):
        """
        Publish annotated image without using cv_bridge 
        """
        try:
            # Convert to ROS Image message without cv_bridge
            ros_image = self.cv2_to_imgmsg(annotated_image, "bgr8")
            ros_image.header = header
            ros_image.header.frame_id = "camera_frame"  # Adjust frame_id as needed
            ros_image.header.stamp = rospy.Time.now()
            
            # Publish uncompressed image
            self.image_pub.publish(ros_image)
            
            # Publish compressed image if enabled
            if self.publish_compressed:
                compressed_msg = CompressedImage()
                compressed_msg.header = ros_image.header
                compressed_msg.format = "jpeg"
                
                # Compress image
                encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
                _, compressed_data = cv2.imencode('.jpg', annotated_image, encode_param)
                compressed_msg.data = compressed_data.tobytes()
                
                self.compressed_pub.publish(compressed_msg)
                
        except Exception as e:
            rospy.logerr(f"Error publishing results: {str(e)}")

    def run(self):
        rospy.loginfo("YOLOv8 Pose Detector running...")
        rospy.spin()

if __name__ == '__main__':
    try:
        detector = YOLOv8PoseDetector()
        detector.run()
    except rospy.ROSInterruptException:
        rospy.loginfo("YOLOv8 Pose Detector shutting down...")