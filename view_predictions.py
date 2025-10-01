import rospy
import numpy as np
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
from std_msgs.msg import ColorRGBA

class TrajectoryEvaluator:
    def __init__(self):
        """
        Initialize publishers once when the class is instantiated
        """
        # Publishers for RViz visualization - initialized once
        self.observed_pub = rospy.Publisher('/trajectory_observed', MarkerArray, queue_size=10)
        self.gt_pub = rospy.Publisher('/trajectory_ground_truth', MarkerArray, queue_size=10)
        self.pred_pub = rospy.Publisher('/trajectory_predictions', MarkerArray, queue_size=10)
        
        print("TrajectoryEvaluator initialized with publishers")


    def create_trajectory_markers(self, points, namespace, color, marker_id_offset=0):
        """
        Create MarkerArray for trajectory visualization
        """
        marker_array = MarkerArray()
        
        if points is None or len(points) == 0:
            return marker_array
            
        # Convert to numpy array if it's a list
        if isinstance(points, list):
            points = np.array(points[0]) if len(points) > 0 else np.array([])
        
        if len(points) == 0:
            return marker_array
            
        # Create line strip for trajectory
        line_marker = Marker()
        line_marker.header.frame_id = "odom"  # Changed to odom to match RViz fixed frame
        line_marker.header.stamp = rospy.Time.now()
        line_marker.ns = f"{namespace}_line"
        line_marker.id = marker_id_offset
        line_marker.type = Marker.LINE_STRIP
        line_marker.action = Marker.ADD
        line_marker.pose.orientation.w = 1.0
        line_marker.scale.x = 0.05  # Line width
        line_marker.color = color
        
        # Add points to line strip
        for point in points:
            p = Point()
            p.x = float(point[0])
            p.y = float(point[1])
            p.z = 0.1  # Slightly above ground
            line_marker.points.append(p)
        
        marker_array.markers.append(line_marker)
        
        # Create sphere markers for individual points
        for i, point in enumerate(points):
            sphere_marker = Marker()
            sphere_marker.header.frame_id = "odom"
            sphere_marker.header.stamp = rospy.Time.now()
            sphere_marker.ns = f"{namespace}_points"
            sphere_marker.id = marker_id_offset + i + 1
            sphere_marker.type = Marker.SPHERE
            sphere_marker.action = Marker.ADD
            sphere_marker.pose.position.x = float(point[0])
            sphere_marker.pose.position.y = float(point[1])
            sphere_marker.pose.position.z = 0.1
            sphere_marker.pose.orientation.w = 1.0
            sphere_marker.scale.x = 0.1
            sphere_marker.scale.y = 0.1
            sphere_marker.scale.z = 0.1
            sphere_marker.color = color
            
            marker_array.markers.append(sphere_marker)
        
        return marker_array

    def publish_trajectories_to_rviz(self, observations, ground_truth, predictions):
        """
        Publish trajectory visualizations to RViz using pre-initialized publishers
        """
        # Define colors with adjusted opacity
        blue_color = ColorRGBA(0.0, 0.0, 1.0, 0.3)    # Blue for observations (reduced opacity)
        green_color = ColorRGBA(0.0, 1.0, 0.0, 0.8)   # Green for ground truth
        red_color = ColorRGBA(1.0, 0.0, 0.0, 1.0)     # Red for predictions (full opacity)
        
        # Create and publish observed trajectory
        obs_markers = self.create_trajectory_markers(observations, "observed", blue_color, 0)
        self.observed_pub.publish(obs_markers)
        
        # Create and publish ground truth trajectory
        gt_markers = self.create_trajectory_markers(ground_truth, "ground_truth", green_color, 100)
        self.gt_pub.publish(gt_markers)
        
        # Create and publish predicted trajectory
        pred_markers = self.create_trajectory_markers([predictions], "predictions", red_color, 200)
        self.pred_pub.publish(pred_markers)
        
        print("Published trajectory markers to RViz")

    def detailed_position_analysis(self, observations, ground_truth, predictions, local_frames, seq_len, interval):
        """
        Detailed position analysis showing recent positions from all data sources
        """
        print("\n--- Detailed Position Analysis ---")
        
        # Get the latest person detection data for comparison
        if not local_frames:
            print("No local frames available for comparison")
            return
            
        # Extract the recent detection coordinates for comparison
        number_frames = seq_len * interval + 1
        if len(local_frames) >= number_frames:
            recent_frames = local_frames[-number_frames:]
            
            # Extract person positions from the detection frames
            original_positions = []
            for i, frame in enumerate(recent_frames):
                if frame and 'coordinates' in frame and frame['coordinates']:
                    person = frame['coordinates'][0]  # First person
                    if 'x' in person and 'z' in person:
                        original_positions.append([person['x'], person['z']])
                    else:
                        print(f"Warning: Missing x/z in frame {i}")
                        continue
                else:
                    print(f"Warning: No coordinates in frame {i}")
                    continue
            
            if not original_positions:
                print("No valid original positions found")
                return
                
            original_positions = np.array(original_positions)
            print(f"Extracted {len(original_positions)} original positions")
        else:
            print(f"Not enough frames for analysis: {len(local_frames)}/{number_frames}")
            return
        
        # Recent original detection positions
        print("Recent original detection positions:")
        for i, pos in enumerate(original_positions[-5:]):  # Show last 5 positions
            frame_idx = len(original_positions) - 5 + i
            print(f"  Frame {frame_idx}: x={pos[0]:.4f}, y={pos[1]:.4f}")
        
        # Corresponding observation positions
        if observations and len(observations) > 0:
            try:
                obs_array = np.array(observations[0])
                print(f"Observation positions (total: {len(obs_array)}):")
                for i, pos in enumerate(obs_array[-5:]):  # Show last 5 positions
                    frame_idx = len(obs_array) - 5 + i
                    print(f"  Frame {frame_idx}: x={pos[0]:.4f}, y={pos[1]:.4f}")
            except Exception as e:
                print(f"Error processing observations: {e}")
        else:
            print("No observations available")
        
        # Corresponding ground truth positions
        if ground_truth and len(ground_truth) > 0:
            try:
                gt_array = np.array(ground_truth[0])
                print(f"Ground truth positions (total: {len(gt_array)}):")
                for i, pos in enumerate(gt_array[-5:]):  # Show last 5 positions
                    frame_idx = len(gt_array) - 5 + i
                    print(f"  Frame {frame_idx}: x={pos[0]:.4f}, y={pos[1]:.4f}")
            except Exception as e:
                print(f"Error processing ground truth: {e}")
        else:
            print("No ground truth available")
        
        # Predicted future positions
        if predictions is not None and len(predictions) > 0:
            try:
                pred_array = np.array(predictions)
                print(f"Predicted future positions (total: {len(pred_array)}):")
                for i, pos in enumerate(pred_array):
                    print(f"  Future step {i+1}: x={pos[0]:.4f}, y={pos[1]:.4f}")
            except Exception as e:
                print(f"Error processing predictions: {e}")
        else:
            print("No predictions available")
        
        # Side-by-side comparison of last known positions
        print("\n--- Side-by-Side Comparison (Last Known Positions) ---")
        if len(original_positions) > 0:
            last_original = original_positions[-1]
            print(f"Original:     x={last_original[0]:.4f}, y={last_original[1]:.4f}")
            
            if observations and len(observations) > 0:
                try:
                    obs_array = np.array(observations[0])
                    if len(obs_array) > 0:
                        last_obs = obs_array[-1]
                        print(f"Observation:  x={last_obs[0]:.4f}, y={last_obs[1]:.4f}")
                        
                        # Calculate difference
                        diff = np.linalg.norm(last_original - last_obs)
                        print(f"Obs-Orig diff: {diff:.4f}m")
                except Exception as e:
                    print(f"Error comparing observations: {e}")
            
            if ground_truth and len(ground_truth) > 0:
                try:
                    gt_array = np.array(ground_truth[0])
                    if len(gt_array) > 0:
                        last_gt = gt_array[-1]
                        print(f"Ground Truth: x={last_gt[0]:.4f}, y={last_gt[1]:.4f}")
                        
                        # Calculate difference
                        diff = np.linalg.norm(last_original - last_gt)
                        print(f"GT-Orig diff:  {diff:.4f}m")
                except Exception as e:
                    print(f"Error comparing ground truth: {e}")
            
            if predictions is not None and len(predictions) > 0:
                try:
                    pred_array = np.array(predictions)
                    first_pred = pred_array[0]
                    print(f"First Pred:   x={first_pred[0]:.4f}, y={first_pred[1]:.4f}")
                    
                    # Calculate difference from last known position
                    diff = np.linalg.norm(last_original - first_pred)
                    print(f"Pred-Orig diff: {diff:.4f}m")
                except Exception as e:
                    print(f"Error comparing predictions: {e}")
        
        print("--- End Position Analysis ---\n")
