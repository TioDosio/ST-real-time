import rospy
import numpy as np
import torch
import random
import math
import traceback
from geometry_msgs.msg import PoseArray, Pose
from tf2_msgs.msg import TFMessage
from view_predictions import TrajectoryEvaluator
from evaluate_jta import load_model_and_config, evaluate_real_time_data
from utils.utils import create_logger

class RealTimeDataCollector:
    def __init__(self, checkpoint_path="realeases/jta/checkpoint/checkpoint.pth.tar"):
        self.local_frames = []
        self.odom_base = None
        self.seq_len = 10
        self.interval = 5  # detection frequency is 5Hz, so interval of 5 means 1 second
        self.debug = False
        self.last_predictions = []
        self.last_orientation = None
        self.min_distance = 1.8
        self.max_distance = 10
        self.can_move = True

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
        self.tf_sub = rospy.Subscriber('/odom', TFMessage, self.tf_callback)

        # Publisher for goal pose
        self.goal_pose_pub = rospy.Publisher('/goal_pose', PoseArray, queue_size=10)

    def tf_callback(self, msg):
        """Extract robot transform from TFMessage"""
        if msg.transforms:
            # Since we only have one transform now, take the first one
            if msg.header.child_frame_id == "base_footprint":
                self.odom_base = msg.transforms[0]

    def image_detections_callback(self, msg):
        if not msg.poses:
            return
        self.create_local_frame(msg)
        self.approach()

    def approach(self):
        # Check if we have enough data to make decisions
        if len(self.local_frames) == 0:
            return False

        self.process_frames()
        
        # Calculate target position from latest predictions
        if self.last_predictions:
            # Get the most recent prediction
            latest_prediction = self.last_predictions[-1]
            
            if latest_prediction:
                pred_array = np.array(latest_prediction)
                target_x = np.mean(pred_array[:, 0])
                target_y = np.mean(pred_array[:, 1])
                
                # Get current robot position
                if self.odom_base is not None:
                    robot_x = self.odom_base.transform.translation.x
                    robot_y = self.odom_base.transform.translation.y
                    
                    # Check if we're close enough to the target (within 0.5m)
                    distance_to_target = math.sqrt((robot_x - target_x)**2 + (robot_y - target_y)**2)
                    
                    if distance_to_target < 0.5:
                        # Stop the robot - we're close enough
                        self.move_or_stop(0, 0, 0, 0)
                    else:
                        # Move towards the predicted position
                        self.move_or_stop(target_x, target_y, 1, self.last_orientation)
                else:
                    # No robot TF available, can't determine distance
                    self.move_or_stop(target_x, target_y, 1, self.last_orientation)

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
                        # Since local_frames are already in odom frame, no transformation needed
                        self.last_predictions.append(predictions)

                        if self.debug:
                            self.trajectory_evaluator.publish_trajectories_to_rviz(observations, ground_truth, predictions)
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
                coords = frame['coordinates'][0]
                frame_data.append([coords['x'], coords['y']])
                
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
        if self.odom_base is None:
            print("No robot TF available for rotation")
            return orientation
            
        try:
            # Get robot's orientation quaternion from transform
            robot_trans = self.odom_base.transform
            robot_qx = robot_trans.rotation.x
            robot_qy = robot_trans.rotation.y
            robot_qz = robot_trans.rotation.z
            robot_qw = robot_trans.rotation.w
            
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
        if self.odom_base is None:
            print("No robot TF available for translation")
            return coords
        
        try:
            # Get robot transform (base_footprint to odom)
            robot_trans = self.odom_base.transform
            robot_x = robot_trans.translation.x
            robot_y = robot_trans.translation.y
            robot_z = robot_trans.translation.z
            
            # Create a copy to avoid modifying the original
            transformed_coords = coords.copy()
            
            # Transform coordinates to odom frame
            transformed_coords['x'] = coords['x'] + robot_x
            transformed_coords['y'] = coords['y'] + robot_y
            
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
                
                pred_list = trajectory_pred.tolist()
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
        for pose in msg.poses:
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
