#!/usr/bin/env python2
import rospy
import cv2
import numpy as np
import tf2_ros
import tf2_geometry_msgs
from sensor_msgs.msg import CompressedImage
from geometry_msgs.msg import PointStamped, TransformStamped
from std_msgs.msg import Header
from cv_bridge import CvBridge
from ultralytics import YOLO
from tf2_msgs.msg import TFMessage

class YOLOTFPersonDetector:
    def __init__(self):
        rospy.init_node('yolo_tf_person_detector', anonymous=True)

        # Camera intrinsic matrix
        self.K = np.array([
            [335.49106984455955, 0.0, 329.76315999999997],
            [0.0, 376.1876816184971, 239.82100277456647],
            [0.0, 0.0, 1.0]
        ])

        # Inverse of intrinsic matrix
        self.K_inv = np.linalg.inv(self.K)

        # Person height assumption (meters)
        self.person_height = 1.7

        # Camera setup parameters (adjust these based on your robot setup)
        self.camera_height = 1.3  # Height of camera above ground (meters)
        self.camera_tilt_angle = 0.1  # Camera tilt angle in radians (positive = looking down)

        # Initialize YOLO model
        self.model = YOLO('yolov8m.pt')

        # TF2 setup
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        # Frame names - adjust these based on your robot configuration
        self.camera_frame = "l_camera_link"  # Adjust based on your camera frame
        self.target_frame = "odom"  # Target frame for publishing coordinates
        self.base_frame = "base_footprint"  # Robot base frame

        # Publishers
        self.floor_coord_pub = rospy.Publisher('/person_coordinates_transformed', PointStamped, queue_size=10)
        self.debug_image_pub = rospy.Publisher('/person_detection_debug', CompressedImage, queue_size=10)
        self.tf_pub = rospy.Publisher('/combined_tf', TFMessage, queue_size=10)

        # Subscribers
        self.image_sub = rospy.Subscriber('/vizzy/l_camera/suppressed_image_rect_color_sd/compressed', 
                                         CompressedImage, self.image_callback)

        # Timer for TF publishing (optional - can be removed if not needed)
        self.tf_timer = rospy.Timer(rospy.Duration(0.02), self.publish_tf)  # 50 Hz

        rospy.loginfo("YOLO TF Person Detector Initialized!")

    def image_callback(self, msg):
        try:
            # Convert compressed image to cv2 format
            np_arr = np.frombuffer(msg.data, np.uint8)
            cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            # Run YOLO detection
            results = self.model(cv_image, verbose=False)
            
            # Process detections
            for r in results:
                boxes = r.boxes
                if boxes is not None:
                    for box in boxes:
                        # Check if detection is a person (class 0 in COCO dataset)
                        if int(box.cls) == 0 and float(box.conf) > 0.5:
                            # Get bounding box coordinates
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            
                            # Calculate floor coordinates in camera frame
                            camera_coords = self.pixel_to_camera_coordinates((x1 + x2) / 2, y2)
                            
                            if camera_coords is not None:
                                # Transform coordinates to target frame
                                transformed_coords = self.transform_coordinates(camera_coords, msg.header.stamp)
                                
                                if transformed_coords is not None:
                                    # Publish transformed coordinates
                                    self.publish_floor_coordinates(transformed_coords, msg.header.stamp)

                                    # Draw debug visualization
                                    cv_image = self.draw_detection(cv_image, x1, y1, x2, y2, 
                                                                 transformed_coords, float(box.conf))

                                    rospy.loginfo(f"Person at floor (transformed): x={transformed_coords[0]:.4f}m, "
                                                f"y={transformed_coords[1]:.4f}m")

            # Publish debug image
            self.publish_debug_image(cv_image, msg.header)

        except Exception as e:
            rospy.logerr(f"Error in image callback: {e}")

    def pixel_to_camera_coordinates(self, pixel_x, pixel_y):
        """
        Convert pixel coordinates to 3D coordinates in camera frame
        """
        try:
            # Convert pixel to homogeneous coordinates
            pixel_coords = np.array([pixel_x, pixel_y, 1.0])

            # Convert to normalized camera coordinates
            camera_coords = self.K_inv @ pixel_coords

            # Extract normalized coordinates
            x_norm = camera_coords[0]
            y_norm = camera_coords[1]

            # Calculate the ray from camera center through the pixel
            cy = self.K[1, 2]  # Principal point Y
            fy = self.K[1, 1]  # Focal length Y
            
            # Angle from camera optical axis to the pixel
            angle_y = np.arctan((pixel_y - cy) / fy)
            
            # Effective downward angle (relative to horizontal)
            downward_angle = angle_y + self.camera_tilt_angle
            
            if downward_angle <= 0:  # Looking up or horizontal, won't hit ground
                return None
            
            # Distance forward to ground intersection
            ground_distance = self.camera_height / np.tan(downward_angle)
            
            if ground_distance <= 0:
                return None
            
            # Lateral position (X coordinate)
            cx = self.K[0, 2]  # Principal point X  
            fx = self.K[0, 0]  # Focal length X
            angle_x = np.arctan((pixel_x - cx) / fx)
            lateral_distance = ground_distance * np.tan(angle_x)
            
            # Return 3D coordinates in camera frame
            # Camera frame: X-right, Y-down, Z-forward
            camera_coords_3d = np.array([lateral_distance, 0.0, ground_distance])
            return camera_coords_3d

        except Exception as e:
            rospy.logerr(f"Error calculating camera coordinates: {e}")
            return None

    def transform_coordinates(self, camera_coords, timestamp):
        """
        Transform coordinates from camera frame to target frame using TF
        """
        try:
            # Create a PointStamped in camera frame
            point_camera = PointStamped()
            point_camera.header.stamp = timestamp
            point_camera.header.frame_id = self.camera_frame
            point_camera.point.x = camera_coords[0]
            point_camera.point.y = camera_coords[1]
            point_camera.point.z = camera_coords[2]

            # Transform to target frame
            try:
                # Wait for transform with timeout
                transform = self.tf_buffer.lookup_transform(
                    self.target_frame, 
                    self.camera_frame, 
                    timestamp, 
                    rospy.Duration(0.1)
                )
                
                # Apply transformation
                point_transformed = tf2_geometry_msgs.do_transform_point(point_camera, transform)
                
                # Return as numpy array [x, y, z]
                return np.array([
                    point_transformed.point.x,
                    point_transformed.point.y,
                    point_transformed.point.z
                ])
                
            except (tf2_ros.LookupException, tf2_ros.ConnectivityException, 
                    tf2_ros.ExtrapolationException) as e:
                rospy.logwarn(f"TF lookup failed: {e}")
                return None

        except Exception as e:
            rospy.logerr(f"Error transforming coordinates: {e}")
            return None

    def publish_floor_coordinates(self, coords, timestamp):
        """Publish transformed floor coordinates"""
        point_msg = PointStamped()
        point_msg.header.stamp = timestamp
        point_msg.header.frame_id = self.target_frame
        point_msg.point.x = coords[0]
        point_msg.point.y = coords[1] 
        point_msg.point.z = coords[2]
        self.floor_coord_pub.publish(point_msg)

    def draw_detection(self, image, x1, y1, x2, y2, transformed_coords, confidence):
        """Draw detection on image"""
        # Bounding box
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

        # Foot position (bottom center)
        foot_x, foot_y = int((x1 + x2) / 2), int(y2)
        cv2.circle(image, (foot_x, foot_y), 5, (0, 0, 255), -1)

        # Text information
        text = f"Person {confidence:.2f}"
        cv2.putText(image, text, (int(x1), int(y1) - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        coord_text = f"Floor: ({transformed_coords[0]:.2f}, {transformed_coords[1]:.2f}, {transformed_coords[2]:.2f})"
        cv2.putText(image, coord_text, (int(x1), int(y2) + 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

        return image

    def publish_debug_image(self, cv_image, original_header):
        """Publish debug image"""
        try:
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
            result, encoded_img = cv2.imencode('.jpg', cv_image, encode_param)

            if result:
                debug_msg = CompressedImage()
                debug_msg.header = original_header
                debug_msg.format = "jpeg"
                debug_msg.data = encoded_img.tobytes()
                self.debug_image_pub.publish(debug_msg)

        except Exception as e:
            rospy.logerr(f"Error publishing debug image: {e}")

    def publish_tf(self, event):
        """
        Publish TF transforms (from tf.py functionality)
        This is optional and can be removed if TF is published elsewhere
        """
        transforms = []
        
        try:
            # Look up the transform from base_footprint to odom (robot's position in world)
            robot_trans = self.tf_buffer.lookup_transform(
                self.target_frame, 
                self.base_frame, 
                rospy.Time(0), 
                rospy.Duration(0.1)
            )
            robot_trans.child_frame_id = "robot_frame"
            transforms.append(robot_trans)
            
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, 
                tf2_ros.ExtrapolationException) as e:
            rospy.logdebug("Robot TF lookup error: %s" % str(e))
        except Exception as e:
            rospy.logerr("Unexpected error in robot TF: %s" % str(e))
        
        try:
            # Look up the transform from camera to odom
            camera_trans = self.tf_buffer.lookup_transform(
                self.target_frame, 
                self.camera_frame, 
                rospy.Time(0), 
                rospy.Duration(0.1)
            )
            camera_trans.child_frame_id = "camera_frame"
            transforms.append(camera_trans)
            
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, 
                tf2_ros.ExtrapolationException) as e:
            rospy.logdebug("Camera TF lookup error: %s" % str(e))
        except Exception as e:
            rospy.logerr("Unexpected error in camera TF: %s" % str(e))
        
        # Publish both transforms in a single message if we have any
        if transforms:
            tf_message = TFMessage()
            tf_message.transforms = transforms
            self.tf_pub.publish(tf_message)

def main():
    try:
        detector = YOLOTFPersonDetector()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass

if __name__ == '__main__':
    main()