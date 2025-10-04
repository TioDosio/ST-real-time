import rospy
import cv2
import numpy as np
from sensor_msgs.msg import CompressedImage
from geometry_msgs.msg import PointStamped
from std_msgs.msg import Header
from cv_bridge import CvBridge
from ultralytics import YOLO

class SimplePersonFloorDetector:
    def __init__(self):
        rospy.init_node('simple_person_floor_detector', anonymous=True)

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

        # Publishers
        self.floor_coord_pub = rospy.Publisher('/person_floor_coordinates', PointStamped, queue_size=10)
        self.debug_image_pub = rospy.Publisher('/person_detection_debug', CompressedImage, queue_size=10)

        # Subscriber
        self.image_sub = rospy.Subscriber('/vizzy/l_camera/suppressed_image_rect_color_sd/compressed', CompressedImage, self.image_callback)

        rospy.loginfo("Detector Initialized!")

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
                            # Calculate floor coordinates using the bottom center of bounding box
                            floor_coords = self.pixel_to_floor_coordinates((x1 + x2) / 2, y2)
                            if floor_coords is not None:
                                # Publish floor coordinates
                                self.publish_floor_coordinates(floor_coords, msg.header.stamp)

                                # Draw debug visualization
                                #cv_image = self.draw_detection(cv_image, x1, y1, x2, y2, floor_coords, float(box.conf))

                                rospy.loginfo(f"Person at floor: x={floor_coords[1]:.4f}m, "f"y={floor_coords[0]:.4f}m")

            # Publish debug image
            #self.publish_debug_image(cv_image, msg.header)

        except Exception as e:
            rospy.logerr(f"Error in image callback: {e}")

    def pixel_to_floor_coordinates(self, pixel_x, pixel_y):
        """
        Convert pixel coordinates to floor coordinates
        Assumes the bottom of the bounding box represents the person's feet on the ground
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
            # Camera coordinate system: X-right, Y-down, Z-forward
            # For floor projection, we need to account for the camera height and orientation

            # The ray direction in camera frame
            ray_direction = np.array([x_norm, y_norm, 1.0])
            ray_direction = ray_direction / np.linalg.norm(ray_direction)

            # Camera position: at height camera_height, looking down towards the ground
            # In world coordinates: camera is at (0, 0, camera_height)
            # Ground plane is at z = 0
            
            # For a camera looking forward/down, we need to transform coordinates
            # The y-coordinate in image corresponds to depth, x to lateral position
            
            # Simple pinhole model for floor projection:
            # Given pixel (x,y), find where the ray hits the ground plane
            
            # Using similar triangles approach:
            # The ray goes from camera center through the pixel to the ground
            # Camera height = 1.3m, pixel_y tells us the angle downward
            
            # Distance to ground point (forward distance)
            # tan(angle) = (pixel_y - cy) / fy
            # where cy is principal point Y, fy is focal length Y
            cy = self.K[1, 2]  # Principal point Y
            fy = self.K[1, 1]  # Focal length Y
            
            # Angle from camera optical axis to the pixel
            angle_y = np.arctan((pixel_y - cy) / fy)
            
            # For a forward-looking camera at height camera_height:
            # - Positive angle_y means looking down (pixel_y > cy, bottom of image)
            # - We need the angle relative to horizontal to find ground intersection
            # - Camera tilt angle should be considered (assuming camera looks slightly down)
            
            # Effective downward angle (relative to horizontal)
            downward_angle = angle_y + self.camera_tilt_angle
            
            if downward_angle <= 0:  # Looking up or horizontal, won't hit ground
                return None
            
            # Distance forward to ground intersection
            ground_distance = self.camera_height / np.tan(downward_angle)
            
            if ground_distance <= 0:
                print("Negative ground distance")
                return None
            
            # Lateral position (X coordinate)
            cx = self.K[0, 2]  # Principal point X  
            fx = self.K[0, 0]  # Focal length X
            angle_x = np.arctan((pixel_x - cx) / fx)
            lateral_distance = ground_distance * np.tan(angle_x)
            
            
            # Return floor coordinates (x=lateral, y=forward)
            floor_coords = np.array([lateral_distance, ground_distance])
            return floor_coords

        except Exception as e:
            rospy.logerr(f"Error calculating floor coordinates: {e}")
            return None

    def publish_floor_coordinates(self, coords, timestamp):
        """Publish floor coordinates"""
        point_msg = PointStamped()
        point_msg.header.stamp = timestamp
        point_msg.header.frame_id = "odom"  # Adjust frame_id as needed
        point_msg.point.x = coords[0]
        point_msg.point.y = coords[1] 
        self.floor_coord_pub.publish(point_msg)

    def draw_detection(self, image, x1, y1, x2, y2, floor_coords, confidence):
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

        coord_text = f"Floor: ({floor_coords[0]:.2f}, {floor_coords[1]:.2f})"
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

def main():
    try:
        detector = SimplePersonFloorDetector()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass

if __name__ == '__main__':
    main()

