#!/usr/bin/env python
import rospy
import actionlib
import math
import threading
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from geometry_msgs.msg import PoseArray

def quaternion_from_euler(roll, pitch, yaw):
    """
    Convert Euler angles to quaternion.
    Compatible replacement for tf.transformations.quaternion_from_euler
    """
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)

    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy

    return [x, y, z, w]

class AdaptiveRobotNavigator:
    def __init__(self, topic_name="/raw_bodies", distance_threshold=0.5):
        """
        Initialize the adaptive robot navigator.
        
        Args:
            topic_name (str): ROS topic name to subscribe to for pose messages
            distance_threshold (float): Distance threshold in meters to consider "close enough"
        """
        self.distance_threshold = distance_threshold
        self.current_target = None
        self.previous_distance = None
        self.is_moving = False
        self.lock = threading.Lock()
        
        # Initialize ROS node
        if not rospy.get_node_uri():
            rospy.init_node('adaptive_robot_navigator', anonymous=True)
        
        # Setup action client for move_base
        self.client = actionlib.SimpleActionClient('move_base', MoveBaseAction)
        rospy.loginfo("Waiting for move_base action server...")
        
        if not self.client.wait_for_server(rospy.Duration(5.0)):
            rospy.logerr("move_base action server not available")
            # Continue anyway to listen for poses
        
        # Subscribe to pose array topic
        self.pose_subscriber = rospy.Subscriber(topic_name, PoseArray, self.pose_callback)
        rospy.loginfo(f"Subscribed to {topic_name} with distance threshold: {distance_threshold}m")
        rospy.loginfo("Waiting for pose messages... Use 'rostopic echo /raw_bodies' to check if messages are being published")
    
    def pose_callback(self, msg):
        """
        Callback function for pose array messages.
        Assumes we want to track the first pose in the array.
        """
        print("=" * 50)
        print("RAW BODIES RECEIVED!")
        print(f"Message header: seq={msg.header.seq}, frame_id='{msg.header.frame_id}'")
        print(f"Timestamp: {msg.header.stamp.secs}.{msg.header.stamp.nsecs}")
        print(f"Number of poses: {len(msg.poses)}")
        
        if not msg.poses:
            print("WARNING: Received empty pose array!")
            rospy.logwarn("Received empty pose array!")
            return
        
        # Print all poses information
        for i, pose in enumerate(msg.poses):
            print(f"\nPose {i}:")
            print(f"  Position: x={pose.position.x:.6f}, y={pose.position.y:.6f}, z={pose.position.z:.6f}")
            print(f"  Orientation: x={pose.orientation.x:.6f}, y={pose.orientation.y:.6f}, z={pose.orientation.z:.6f}, w={pose.orientation.w:.6f}")
        
        # Get the first pose from the array for navigation
        target_pose = msg.poses[0]
        target_x = target_pose.position.x
        target_y = target_pose.position.y
        
        # Calculate distance to target
        current_distance = math.sqrt(target_x**2 + target_y**2)
        
        print(f"\nNavigation Info:")
        print(f"  Target coordinates: ({target_x:.6f}, {target_y:.6f})")
        print(f"  Distance to target: {current_distance:.6f}m")
        print(f"  Distance threshold: {self.distance_threshold}m")
        print("=" * 50)
        
        with self.lock:
            rospy.loginfo(f"Target at ({target_x:.2f}, {target_y:.2f}), distance: {current_distance:.2f}m")
            
            # Check if we should start moving, stop, or continue
            if current_distance <= self.distance_threshold:
                if self.is_moving:
                    rospy.loginfo("Close enough! Stopping robot.")
                    self.client.cancel_goal()
                    self.is_moving = False
            else:
                # Check if we're moving away (distance increasing)
                if self.previous_distance is not None:
                    if current_distance > self.previous_distance and not self.is_moving:
                        rospy.loginfo("Moving away from target, starting navigation.")
                        self.move_to_target(target_x, target_y, target_pose.orientation)
                elif not self.is_moving:
                    # First time or robot is not moving and target is far
                    rospy.loginfo("Target detected, starting navigation.")
                    self.move_to_target(target_x, target_y, target_pose.orientation)
            
            self.previous_distance = current_distance
    
    def move_to_target(self, x, y, orientation, frame="base_footprint"):
        """
        Send a goal to move_base to navigate to the target position.
        """
        try:
            # Build goal
            goal = MoveBaseGoal()
            goal.target_pose.header.frame_id = frame
            goal.target_pose.header.stamp = rospy.Time.now()
            goal.target_pose.pose.position.x = x
            goal.target_pose.pose.position.y = y
            goal.target_pose.pose.orientation = orientation
            
            rospy.loginfo(f"Sending goal: x={x:.2f}, y={y:.2f}")
            self.client.send_goal(goal)
            self.is_moving = True
            
        except Exception as e:
            rospy.logerr(f"Error sending goal: {e}")
            self.is_moving = False
    
    def set_distance_threshold(self, new_threshold):
        """
        Update the distance threshold for stopping.
        
        Args:
            new_threshold (float): New distance threshold in meters
        """
        with self.lock:
            self.distance_threshold = new_threshold
            rospy.loginfo(f"Distance threshold updated to: {new_threshold}m")
    
    def stop_robot(self):
        """
        Stop the robot by canceling current goal.
        """
        with self.lock:
            if self.is_moving:
                rospy.loginfo("Manually stopping robot...")
                self.client.cancel_goal()
                self.is_moving = False
    
    def get_status(self):
        """
        Get current status of the navigator.
        
        Returns:
            dict: Current status including distance threshold, moving state, etc.
        """
        with self.lock:
            return {
                'distance_threshold': self.distance_threshold,
                'is_moving': self.is_moving,
                'previous_distance': self.previous_distance,
                'has_target': self.current_target is not None
            }
    
    def spin(self):
        """
        Keep the node running and processing callbacks.
        """
        rospy.loginfo("Adaptive robot navigator ready. Waiting for pose messages...")
        rospy.spin()


def move_robot_to_coordinate(x, y, yaw=0.0, frame="base_footprint", timeout=60.0, cancel_after=None):
    try:
        # Initialize ROS node
        if not rospy.get_node_uri():
            rospy.init_node('robot_navigation', anonymous=True)
        
        client = actionlib.SimpleActionClient('move_base', MoveBaseAction)
        rospy.loginfo("Waiting for move_base action server...")
        
        if not client.wait_for_server(rospy.Duration(5.0)):
            rospy.logerr("move_base action server not available")
            return False
        
        # Build goal
        goal = MoveBaseGoal()
        goal.target_pose.header.frame_id = frame
        goal.target_pose.header.stamp = rospy.Time.now()
        goal.target_pose.pose.position.x = x
        goal.target_pose.pose.position.y = y

        q = quaternion_from_euler(0, 0, yaw)
        goal.target_pose.pose.orientation.x = q[0]
        goal.target_pose.pose.orientation.y = q[1]
        goal.target_pose.pose.orientation.z = q[2]
        goal.target_pose.pose.orientation.w = q[3]

        rospy.loginfo("Sending goal: x={:.2f}, y={:.2f}, yaw={:.2f}".format(x, y, yaw))
        client.send_goal(goal)

        # Optional: cancel goal after some time
        if cancel_after is not None:
            rospy.sleep(cancel_after)
            rospy.logwarn("Cancelling goal after {:.1f} seconds!".format(cancel_after))
            client.cancel_goal()

        # Wait for result
        finished = client.wait_for_result(rospy.Duration(timeout))
        if not finished:
            rospy.logwarn("Goal timed out after {} seconds.".format(timeout))
            client.cancel_goal()
            return False

        state = client.get_state()
        if state == actionlib.GoalStatus.SUCCEEDED:
            rospy.loginfo("Successfully reached target!")
            return True
        else:
            rospy.logwarn("Failed or cancelled. State: {}".format(state))
            return False

    except Exception as e:
        rospy.logerr("Error in move_robot_to_coordinate: {}".format(e))
        return False


if __name__ == '__main__':
    try:
        # Configuration
        TOPIC_NAME = "/raw_bodies"  # Your actual topic name
        DISTANCE_THRESHOLD = 1    # Distance in meters to consider "close enough"
        
        # Create and start the adaptive navigator
        navigator = AdaptiveRobotNavigator(
            topic_name=TOPIC_NAME,
            distance_threshold=DISTANCE_THRESHOLD
        )
        
        # Optional: You can change the threshold dynamically
        # navigator.set_distance_threshold(0.3)  # Change to 30cm
        
        # Start listening for messages and processing navigation
        navigator.spin()
        
    except rospy.ROSInterruptException:
        rospy.loginfo("Navigation interrupted by user")
    except KeyboardInterrupt:
        rospy.loginfo("Navigation stopped by user")
    except Exception as e:
        rospy.logerr(f"Error in main: {e}")
    
    # Legacy example using the original function
    # success = move_robot_to_coordinate(1.0, 0.0, yaw=0.0, cancel_after=5.0)
    # if success:
    #     print("Robot reached the target!")
    # else:
    #     print("Goal was cancelled or failed.")
