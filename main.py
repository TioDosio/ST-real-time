import rospy
import numpy as np
import torch
import random
import math
from geometry_msgs.msg import PointStamped
from visualization_msgs.msg import MarkerArray
from tf2_msgs.msg import TFMessage
from view_predictions import TrajectoryEvaluator
from evaluate_jta import load_model_and_config, evaluate_real_time_data
from utils.utils import create_logger
from move_vizzy import move_robot_to_coordinate

class RealTimeDataCollector:
    def __init__(self, checkpoint_path="realeases/jta/checkpoint/checkpoint.pth.tar"):
        self.local_frames = []
        self.latest_robot_tf = None
        self.seq_len = 10
        self.interval = 5  # detection frequency is 5Hz, so interval of 5 means 1 second
        self.debug = False
        self.last_predictions = []
        self.last_yaw = 0

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
        self.image_detections_sub = rospy.Subscriber('/person_coordinates', PointStamped, self.image_detections_callback)
        self.tf_sub = rospy.Subscriber('/combined_tf', TFMessage, self.tf_callback)

    def tf_callback(self, msg):
        """Extract robot transform from TFMessage"""
        if msg.transforms:
            # Since we only have one transform now, take the first one
            self.latest_robot_tf = msg.transforms[0]

    def image_detections_callback(self, msg):
        self.create_local_frame(msg)
        
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
                        # Transform observations and predictions to odom frame before visualization
                        transformed_observations = self.transform_to_odom_frame(observations)
                        transformed_predictions = self.transform_to_odom_frame(predictions)
                        
                        # Calculate direction from trajectory predictions
                        self.last_yaw = self.calculate_direction_from_trajectory(transformed_predictions)
                        
                        self.last_predictions.append(transformed_predictions)

                        # Visualize trajectories with transformed coordinates
                        if len(self.last_predictions) > 3:
                            move_robot_to_coordinate(self.last_predictions, self.last_yaw)
                        self.trajectory_evaluator.publish_trajectories_to_rviz(transformed_observations, ground_truth, transformed_predictions)
                        
                    else:
                        print("WARNING: Skipping visualization due to empty predictions")
                        return
                else:
                    print("WARNING: No trajectory data extracted from frames")
                    
            except Exception as e:
                print(f"ERROR in prediction pipeline: {e}")
                import traceback
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

    def transform_to_odom_frame(self, points):
        """Transform points from robot frame to odom frame using latest robot TF"""
        if self.latest_robot_tf is None or points is None or len(points) == 0:
            print("No robot TF available or no points to transform")
            return points
            
        try:
            # Get robot transform (base_footprint to odom)
            robot_trans = self.latest_robot_tf.transform
            robot_x = robot_trans.translation.x
            robot_y = robot_trans.translation.y
            
            # For simplicity, assuming no rotation (can be extended if needed)
            # Transform points: robot_frame_point + robot_position_in_odom = odom_frame_point
            transformed_points = []
            
            if isinstance(points, list) and len(points) > 0:
                # Check if it's a list of trajectories (observations format) or single trajectory (predictions format)
                if isinstance(points[0], list) and len(points[0]) > 0 and isinstance(points[0][0], list):
                    # List of trajectories format: [[trajectory1], [trajectory2], ...]
                    for trajectory in points:
                        transformed_traj = []
                        for point in trajectory:
                            if isinstance(point, list) and len(point) >= 2:
                                transformed_point = [point[0] + robot_x, point[1] + robot_y]
                                transformed_traj.append(transformed_point)
                        transformed_points.append(transformed_traj)
                elif isinstance(points[0], list) and len(points[0]) >= 2:
                    # Single trajectory format: [[x1, y1], [x2, y2], ...]
                    for point in points:
                        if isinstance(point, list) and len(point) >= 2:
                            transformed_point = [point[0] + robot_x, point[1] + robot_y]
                            transformed_points.append(transformed_point)
                else:
                    # Handle other list formats
                    transformed_points = points
                    
            elif hasattr(points, 'shape'):  # Numpy array
                points_copy = np.copy(points)
                if len(points_copy.shape) >= 2 and points_copy.shape[1] >= 2:
                    points_copy[:, 0] += robot_x  # Add robot x to all x coordinates
                    points_copy[:, 1] += robot_y  # Add robot y to all y coordinates
                transformed_points = points_copy
            else:
                transformed_points = points
            
            return transformed_points
            
        except Exception as e:
            print(f"Error transforming to odom frame: {e}")
            import traceback
            print(traceback.format_exc())
            return points

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
    
    def calculate_direction_from_trajectory(self, predictions):
        """Calculate movement direction from trajectory predictions"""
        try:
            if not predictions or len(predictions) < 2:
                return self.last_yaw  # Keep previous direction if not enough data
            
            # Get the first and last points of the trajectory
            if isinstance(predictions[0], list) and len(predictions[0]) >= 2:
                # Single trajectory format
                start_point = predictions[0]
                end_point = predictions[-1]
            else:
                # Handle other formats
                return self.last_yaw
            
            # Calculate direction vector
            dx = end_point[0] - start_point[0]
            dy = end_point[1] - start_point[1]
            
            # Calculate yaw angle from direction vector
            yaw = math.atan2(dy, dx)
            
            return yaw
            
        except Exception as e:
            print(f"Error calculating direction: {e}")
            return self.last_yaw

    def create_local_frame(self, msg):
        local_frame = {
            'timestamp': msg.header.stamp,
            'coordinates': []
        }
        coords = {
            'x': msg.point.x,
            'y': msg.point.y
        }
        
        local_frame['coordinates'].append(coords)
        self.local_frames.append(local_frame)

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
