import rospy
import numpy as np
import torch
import random
import math
import traceback
import os
from geometry_msgs.msg import PoseArray, Pose
from tf2_msgs.msg import TFMessage
from geometry_msgs.msg import TransformStamped
from view_predictions import TrajectoryEvaluator
from evaluate_jta import load_model_and_config, evaluate_real_time_data
from utils.utils import create_logger

class RealTimeDataCollector:
    def __init__(self, checkpoint_path="realeases/jta/checkpoint/checkpoint.pth.tar"):
        self.local_frames = []
        self.robot_transform = None
        self.seq_len = 10
        self.interval = 5  # detection frequency is 5Hz, so interval of 5 means 1 second
        self.debug = True
        self.last_predictions = []
        self.last_orientation = None
        
        # Table display variables
        self.trajectory_table_data = []
        self.frame_counter = 0
        self.max_table_rows = 10

        # Set random seeds for reproducibility
        random.seed(0)
        np.random.seed(0)
        torch.manual_seed(0)

        rospy.init_node('real_time_data_collector', anonymous=True)
        
        # Initialize logger
        self.logger = create_logger('')
        
        # Load model and configuration
        self.logger.info("Initializing model for real-time prediction...")
        self.model, self.config = load_model_and_config(checkpoint_path, self.logger)
        self.model.eval()  # Set to evaluation mode
        self.logger.info("Model loaded successfully!")
        
        # Initialize trajectory evaluator
        self.trajectory_evaluator = TrajectoryEvaluator()
        
        # Subscribers
        self.image_detections_sub = rospy.Subscriber('/raw_bodies', PoseArray, self.image_detections_callback)
        self.tf_sub = rospy.Subscriber('/tf', TFMessage, self.tf_callback)

        # Publisher for goal pose
        self.goal_pose_pub = rospy.Publisher('/goal_pose', PoseArray, queue_size=10)

    def tf_callback(self, msg):
        """Extract robot transform from TF message"""
        for transform in msg.transforms:
            if transform.child_frame_id == "base_footprint":
                self.robot_transform = transform
                break            

    def image_detections_callback(self, msg):
        #print(f"[DEBUG] Image detections callback received with {len(msg.poses)} poses")
        if not msg.poses:
            return
        self.create_local_frame(msg)
        #print(f"[DEBUG] Total local frames stored: {len(self.local_frames)}")
        self.process_frames()

    def process_frames(self):
        # Process frames when we have enough data
        if len(self.local_frames) > self.seq_len * self.interval + 1:
            try:
                # Extract trajectory data from local frames
                X = self.extract_trajectory_data()
                
                if X is not None and len(X) > 0:
                    # Use the loaded model for real-time prediction
                    predictions = self.predict_trajectories(X)
                    
                    # Create data for visualization
                    observations = [X] if X is not None else []
                    ground_truth = []  # No ground truth in real-time
                    
                    # Only visualize if we have predictions
                    if predictions:
                        # Transform predictions to odom frame before visualization
                        transformed_predictions = self.transform_predictions_to_odom(predictions)
                        self.last_predictions.append(transformed_predictions)

                        # Display trajectory table in terminal
                        self.update_trajectory_table(observations, transformed_predictions)

                        if self.debug:
                            self.trajectory_evaluator.publish_trajectories_to_rviz(observations, ground_truth, transformed_predictions)
                    else:
                        print("WARNING: Skipping visualization due to empty predictions")
                        return
                else:
                    print("WARNING: No trajectory data extracted from frames")
                    
            except Exception as e:
                print(f"ERROR in prediction pipeline: {e}")
                print(traceback.format_exc())

    def extract_trajectory_data(self):
        """Extract trajectory data from recent local frames"""

        # Take the most recent frames according to sequence length and interval
        recent_frames = self.local_frames[-self.seq_len * self.interval:]
        # Extract person trajectories (simplified version)
        trajectories = []
        
        # Group frames by interval to get the sequence
        for i in range(0, len(recent_frames), self.interval):
            if i < len(recent_frames):
                frame = recent_frames[i]
                frame_data = []
                if len(frame['coordinates']) > 0:
                    coords = frame['coordinates'][0]
                    frame_data.append([coords['x'], coords['y']])
                    #print(f"[DEBUG] Frame {i//self.interval}: person at x={coords['x']:.3f}, y={coords['y']:.3f}")
                
                if frame_data:
                    trajectories.append(frame_data)
        
        # Convert to numpy array format expected by model
        if trajectories and len(trajectories) >= 9:  # Need at least input_track_size frames
            # Take the first person's trajectory for simplicity
            # In a real implementation, you'd handle multiple people
            person_trajectory = []
            for frame_data in trajectories[:9]:  # input_track_size = 9
                if frame_data:  # If there's at least one person in this frame
                    person_trajectory.append(frame_data[0])  # Take first person
                else:
                    # If no person in frame, use last known position or zero
                    if person_trajectory:
                        person_trajectory.append(person_trajectory[-1])
                        print(f"Missing person in frame, using last known position")
                    else:
                        person_trajectory.append([0.0, 0.0])
                        print(f"Missing person in frame, using zero position")
            
            result = np.array(person_trajectory)
            return result
        else:
            print(f"Not enough trajectory frames: {len(trajectories)} < 9")
            return None

    def rotation_tf(self, orientation):
        if self.robot_transform is None:
            print("No robot TF available for rotation")
            return orientation
            
        try:
            # Get robot's orientation quaternion from transform
            robot_qx = self.robot_transform.transform.rotation.x
            robot_qy = self.robot_transform.transform.rotation.y
            robot_qz = self.robot_transform.transform.rotation.z
            robot_qw = self.robot_transform.transform.rotation.w
            
            # Quaternion multiplication to combine orientations: q_result = q_robot * q_local
            q1_w, q1_x, q1_y, q1_z = robot_qw, robot_qx, robot_qy, robot_qz  # robot orientation
            q2_w, q2_x, q2_y, q2_z = orientation['w'], orientation['x'], orientation['y'], orientation['z']  # local orientation
            
            # Quaternion multiplication formula
            result_w = q1_w * q2_w - q1_x * q2_x - q1_y * q2_y - q1_z * q2_z
            result_x = q1_w * q2_x + q1_x * q2_w + q1_y * q2_z - q1_z * q2_y
            result_y = q1_w * q2_y - q1_x * q2_z + q1_y * q2_w + q1_z * q2_x
            result_z = q1_w * q2_z + q1_x * q2_y - q1_y * q2_x + q1_z * q2_w
            
            # Return transformed orientation in same format as input
            transformed_orientation = {
                'x': result_x,
                'y': result_y,
                'z': result_z,
                'w': result_w
            }
            
            return transformed_orientation
            
        except Exception as e:
            print(f"Error in rotation transformation: {e}")
            return orientation


    def translation_tf(self, coords):
        if self.robot_transform is None:
            print("No robot TF available for translation")
            return coords
        
        try:
            # Get robot position from transform (base_footprint to odom)
            robot_x = self.robot_transform.transform.translation.x
            robot_y = self.robot_transform.transform.translation.y
            robot_z = self.robot_transform.transform.translation.z
            
            # For proper transformation, we need to apply rotation first, then translation
            # Get the rotation quaternion
            qx = self.robot_transform.transform.rotation.x
            qy = self.robot_transform.transform.rotation.y
            qz = self.robot_transform.transform.rotation.z
            qw = self.robot_transform.transform.rotation.w
            
            # Convert quaternion to rotation matrix elements we need
            # For 2D transformation, we mainly need the yaw rotation
            yaw = math.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
            cos_yaw = math.cos(yaw)
            sin_yaw = math.sin(yaw)
            
            # Apply rotation to the local coordinates
            x_local = coords['x']
            y_local = coords['y']
            
            # Rotate the local coordinates to align with robot's orientation
            x_rotated = cos_yaw * x_local - sin_yaw * y_local
            y_rotated = sin_yaw * x_local + cos_yaw * y_local
            
            # Create a copy to avoid modifying the original
            transformed_coords = coords.copy()
            
            # Transform coordinates to odom frame (rotation + translation)
            transformed_coords['x'] = x_rotated + robot_x
            transformed_coords['y'] = y_rotated + robot_y
            
            # Handle z coordinate if it exists in input
            if 'z' in coords:
                transformed_coords['z'] = coords['z'] + robot_z
            else:
                # If z not in input, add it with robot's z offset
                transformed_coords['z'] = robot_z
                
            return transformed_coords
            
        except Exception as e:
            print(f"Error in translation transformation: {e}")
            return coords

    def transform_predictions_to_odom(self, predictions):
        """Transform prediction coordinates from robot frame to odom frame"""
        if not predictions or self.robot_transform is None:
            return predictions
        
        transformed_predictions = []
        try:
            for pred_point in predictions:
                # Create coordinate dict in the format expected by translation_tf
                coord_dict = {
                    'x': pred_point[0],
                    'y': pred_point[1],
                    'z': 0.0  # Assuming 2D predictions
                }
                
                # Transform to odom frame
                transformed_coord = self.translation_tf(coord_dict)
                
                # Convert back to list format
                transformed_predictions.append([transformed_coord['x'], transformed_coord['y']])
                
            return transformed_predictions
            
        except Exception as e:
            print(f"Error transforming predictions to odom frame: {e}")
            return predictions

    def clear_terminal(self):
        """Clear the terminal screen"""
        os.system('cls' if os.name == 'nt' else 'clear')

    def print_trajectory_table(self, observations, predictions, current_pos=None):
        """Print a formatted table showing observations, current position, and predictions"""
        self.clear_terminal()
        
        print("=" * 110)
        print("REAL-TIME TRAJECTORY PREDICTION TABLE")
        print("=" * 110)
        print()
        
        # Header - updated to include current position
        header = f"{'Frame':<8} {'Obs x':<10} {'Obs y':<10} {'Current x':<10} {'Current y':<10} {'GT x':<10} {'GT y':<10} {'Pred x':<10} {'Pred y':<10}"
        print(header)
        print("-" * 110)
        
        max_frames = 15  # Display up to 15 frames to accommodate more predictions
        
        # Observations (input frames)
        obs_len = 0
        if observations and len(observations) > 0:
            obs_data = observations[0]  # Get the trajectory data
            obs_len = len(obs_data)
            for i, point in enumerate(obs_data):
                if i >= max_frames:
                    break
                frame_num = i + 1
                obs_x = f"{point[0]:.4f}" if len(point) > 0 else "—"
                obs_y = f"{point[1]:.4f}" if len(point) > 1 else "—"
                current_x = "—"
                current_y = "—"
                gt_x = "—"
                gt_y = "—"
                pred_x = "—"
                pred_y = "—"
                print(f"{frame_num:<8} {obs_x:<10} {obs_y:<10} {current_x:<10} {current_y:<10} {gt_x:<10} {gt_y:<10} {pred_x:<10} {pred_y:<10}")
        
        # Current position (transition frame between observation and prediction)
        if current_pos and obs_len < max_frames:
            frame_num = obs_len + 1
            obs_x = "—"
            obs_y = "—"
            current_x = f"{current_pos[0]:.4f}"
            current_y = f"{current_pos[1]:.4f}"
            gt_x = "—"
            gt_y = "—"
            pred_x = "—"
            pred_y = "—"
            print(f"{frame_num:<8} {obs_x:<10} {obs_y:<10} {current_x:<10} {current_y:<10} {gt_x:<10} {gt_y:<10} {pred_x:<10} {pred_y:<10}")
            current_frame_count = 1
        else:
            current_frame_count = 0
        
        # Predictions (future frames) - limit to 5 points
        if predictions and len(predictions) > 0:
            pred_start_frame = obs_len + current_frame_count + 1
            for i, point in enumerate(predictions[:5]):  # Limit to 5 prediction points
                frame_num = pred_start_frame + i
                if frame_num > max_frames:
                    break
                obs_x = "—"
                obs_y = "—"
                current_x = "—"
                current_y = "—"
                gt_x = "—"
                gt_y = "—"
                pred_x = f"{point[0]:.4f}"
                pred_y = f"{point[1]:.4f}"
                print(f"{frame_num:<8} {obs_x:<10} {obs_y:<10} {current_x:<10} {current_y:<10} {gt_x:<10} {gt_y:<10} {pred_x:<10} {pred_y:<10}")
        
        # Fill remaining rows with empty data if needed
        total_shown = obs_len + current_frame_count + min(5, len(predictions) if predictions else 0)
        for i in range(total_shown, max_frames):
            frame_num = i + 1
            print(f"{frame_num:<8} {'—':<10} {'—':<10} {'—':<10} {'—':<10} {'—':<10} {'—':<10} {'—':<10} {'—':<10}")
        
        print("-" * 110)
        print(f"Observation frames: {obs_len}")
        print(f"Current position: {1 if current_pos else 0}")
        print(f"Prediction frames: {min(5, len(predictions)) if predictions else 0}")
        print()

    def update_trajectory_table(self, observations, predictions):
        """Update and display the trajectory table"""
        try:
            # Get current position (most recent detection)
            current_pos = None
            if self.local_frames and len(self.local_frames) > 0:
                latest_frame = self.local_frames[-1]
                if latest_frame['coordinates'] and len(latest_frame['coordinates']) > 0:
                    latest_coords = latest_frame['coordinates'][0]
                    current_pos = [latest_coords['x'], latest_coords['y']]
            
            # Clear screen and print updated table
            self.print_trajectory_table(observations, predictions, current_pos)
            
            # Print additional status information
            print("STATUS INFORMATION:")
            print("-" * 50)
            
            # Robot transform info
            if self.robot_transform:
                robot_x = self.robot_transform.transform.translation.x
                robot_y = self.robot_transform.transform.translation.y
                robot_z = self.robot_transform.transform.translation.z
                print(f"Robot position (odom): x={robot_x:.3f}, y={robot_y:.3f}, z={robot_z:.3f}")
                
                # Show rotation
                qx = self.robot_transform.transform.rotation.x
                qy = self.robot_transform.transform.rotation.y
                qz = self.robot_transform.transform.rotation.z
                qw = self.robot_transform.transform.rotation.w
                yaw = math.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
                print(f"Robot orientation: yaw={math.degrees(yaw):.1f}°")
            else:
                print("Robot position: No TF data available")
            
            # Frame processing info
            print(f"Total stored frames: {len(self.local_frames)}")
            print(f"Sequence length: {self.seq_len}")
            print(f"Frame interval: {self.interval}")
            
            # Timestamp
            current_time = rospy.Time.now()
            print(f"Last update: {current_time.secs}.{current_time.nsecs//1000000:03d}")
            
            print()
            print("Coordinates are in odom frame (global coordinates)")
            print("Press Ctrl+C to stop...")
            
        except Exception as e:
            print(f"Error updating trajectory table: {e}")

    def predict_trajectories(self, X):
        """Use the loaded model to predict trajectories from processed frame data"""
        try:
            if X is None or len(X) == 0:
                print("ERROR: Input X is None or empty")
                return []
                
            # Convert input data to the format expected by the model
            # Model expects (B, in_F, NJ, K) where:
            # B = batch size, in_F = input frames, NJ = num_people * num_joints, K = coordinates (4D)
            input_joints = torch.tensor(X, dtype=torch.float32)
            
            # The model expects 47 joints per person (token_num from config)
            # and 4D coordinates (x, y, z, confidence/visibility)
            # We only have 2D trajectory points, so we'll extend to 4D
            
            B = 1  # batch size
            in_F = input_joints.shape[0]  # input frames (9)
            J = 47  # number of joints per person (from config token_num)
            N = 1   # number of people
            NJ = N * J  # total joints (1 person * 47 joints = 47)
            K = 4   # coordinate dimensions (x, y, z, confidence) - model expects 4D
            
            # Create padded joint data: (in_F, NJ, K)
            padded_joints = torch.zeros(in_F, NJ, K)
            
            # Place our trajectory data in the first joint (index 0) of each frame
            # Convert 2D (x, y) to 4D (x, y, z=0, confidence=1)
            padded_joints[:, 0, 0] = input_joints[:, 0]  # x coordinate
            padded_joints[:, 0, 1] = input_joints[:, 1]  # y coordinate
            padded_joints[:, 0, 2] = 0.0                 # z coordinate (2D data, so z=0)
            padded_joints[:, 0, 3] = 1.0                 # confidence/visibility (valid data)
            
            # Reshape to model format: (B, in_F, NJ, K)
            input_joints = padded_joints.unsqueeze(0)  # (in_F, NJ, K) -> (1, in_F, NJ, K)
            
            # Create padding mask with correct dimensions (B, N) not (B, NJ)
            # B = batch size, N = number of people (not total joints)
            N = 1  # number of people
            padding_mask = torch.zeros(B, N, dtype=torch.bool)  # All people are valid (not padded)
            
            # Get predictions using the real-time evaluation function
            predictions = evaluate_real_time_data(self.model, self.config, input_joints, padding_mask)
            
            if predictions is not None:
                # Convert predictions back to numpy and return
                pred_numpy = predictions.detach().numpy()
                
                # Check if predictions are reasonable
                if pred_numpy.size == 0:
                    print("ERROR: Predictions tensor is empty")
                    return []
                
                # Extract the trajectory data from the prediction tensor
                # Predictions likely have shape (B, N, F, J, K) - extract the trajectory coordinates
                if len(pred_numpy.shape) == 5:
                    # Extract first person's trajectory coordinates: (B, N, F, J, K) -> (F, K)
                    trajectory_pred = pred_numpy[0, 0, :, 0, :2]  # Get first person, first joint, x,y coords
                elif len(pred_numpy.shape) == 4:
                    # If shape is (B, F, J, K)
                    trajectory_pred = pred_numpy[0, :, 0, :2]
                elif len(pred_numpy.shape) == 3:
                    # If shape is (B, F, K)
                    trajectory_pred = pred_numpy[0, :, :2]
                else:
                    # Fallback: flatten and reshape
                    trajectory_pred = pred_numpy.reshape(-1, 2)
                
                # Convert to list and ensure we have exactly 5 prediction points
                pred_list = trajectory_pred.tolist()
                
                # If we have more than 5 predictions, take the first 5
                if len(pred_list) > 5:
                    pred_list = pred_list[:5]
                # If we have fewer than 5 predictions, extend by repeating the last point
                elif len(pred_list) < 5 and len(pred_list) > 0:
                    last_point = pred_list[-1]
                    while len(pred_list) < 5:
                        pred_list.append(last_point.copy())
                
                return pred_list
            else:
                print("ERROR: evaluate_real_time_data returned None")
                return []
            
        except Exception as e:
            print(f"ERROR in trajectory prediction: {e}")
            import traceback
            print(traceback.format_exc())
            return []

    def create_local_frame(self, msg):
        local_frame = {
            'timestamp': msg.header.stamp,
            'coordinates': [],
            'orientation': [],
        }
        
        # Extract coordinates and orientation from all poses in the PoseArray
        for i, pose in enumerate(msg.poses):
            coords = {
                'x': pose.position.x,
                'y': pose.position.y,
                'z': pose.position.z
            }
            orientation = {
                'x': pose.orientation.x,
                'y': pose.orientation.y,
                'z': pose.orientation.z,
                'w': pose.orientation.w
            }

            # Transform coordinates and orientation to odom frame
            transformed_coords = self.translation_tf(coords)
            transformed_orientation = self.rotation_tf(orientation)

            local_frame['coordinates'].append(transformed_coords)
            local_frame['orientation'].append(transformed_orientation)

        # Only add frame if it has at least one pose
        if local_frame['coordinates'] and local_frame['orientation']:
            self.last_orientation = local_frame['orientation'][0]
            self.local_frames.append(local_frame)

    def move_or_stop(self, x, y, flag, orientation):
        """Send goal pose to move the robot"""
        pose_array = PoseArray()
        pose_array.header.stamp = rospy.Time.now()
        pose_array.header.frame_id = "odom"  # Using odom frame for goal
        
        pose = Pose()
        pose.position.x = x
        pose.position.y = y
        pose.position.z = flag

        if isinstance(orientation, dict):
            pose.orientation.x = orientation.get('x', 0.0)
            pose.orientation.y = orientation.get('y', 0.0)
            pose.orientation.z = orientation.get('z', 0.0)
            pose.orientation.w = orientation.get('w', 1.0)
        
        pose_array.poses.append(pose)
        self.goal_pose_pub.publish(pose_array)
        print("Goal pose published successfully!!")

    def spin(self):
        rospy.loginfo('RealTimeDataCollector spinning...')
        rospy.spin()

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="realeases/jta/checkpoint/checkpoint.pth.tar", 
                       help="Path to model checkpoint")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    
    args = parser.parse_args()
    
    try:
        collector = RealTimeDataCollector(checkpoint_path=args.checkpoint)
        if args.debug:
            collector.debug = True
        collector.spin()
    except Exception as e:
        print(f"Error initializing collector: {e}")
        import traceback
        traceback.print_exc()
