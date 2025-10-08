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
    def __init__(self):
        """
        Initialize the adaptive robot navigator.
        
        Args:
            topic_name (str): ROS topic name to subscribe to for pose messages
            distance_threshold (float): Distance threshold in meters to consider "close enough"
        """
        self.distance_threshold = 1.6
        self.max_distance_threshold = 10.0
        self.current_target = None
        self.previous_distance = None
        self.is_moving = False
        self.lock = threading.Lock()
        self.pose_count = None
        
        # Initialize ROS node
        if not rospy.get_node_uri():
            rospy.init_node('adaptive_robot_navigator', anonymous=True)
        
        # Setup action client for move_base
        self.client = actionlib.SimpleActionClient('move_base', MoveBaseAction)
        rospy.loginfo("Waiting for move_base action server...")
        if not self.client.wait_for_server(rospy.Duration(5.0)):
            rospy.logerr("move_base action server not available")
            # Continue anyway to listen for poses
        else:
            rospy.loginfo("Connected to move_base action server successfully")
            # Check if there's any current goal
            state = self.client.get_state()
            rospy.loginfo("Current move_base state: {}".format(state))
        
        # Subscribe to pose array topic
        self.pose_subscriber = rospy.Subscriber("/raw_bodies", PoseArray, self.pose_callback)
        rospy.loginfo("Subscribed to {} with distance threshold: {}m".format("/raw_bodies", self.distance_threshold))
        rospy.loginfo("Waiting for pose messages... Use 'rostopic echo /raw_bodies' to check if messages are being published")
    
    def pose_callback(self, msg):
        """
        Callback function for pose array messages.
        Assumes we want to track the first pose in the array.
        """
        if not msg.poses:
            rospy.logwarn("Received empty pose array!")
            return
        # Get the first pose from the array for navigation
        target_pose = msg.poses[0]
        target_x = target_pose.position.x
        target_y = target_pose.position.y
        
        # Calculate distance to target
        current_distance = math.sqrt(target_x**2 + target_y**2)
        
        # Only log every 5th pose message to reduce clutter
        self.pose_count += 1
        if self.pose_count % 5 == 1:
            rospy.loginfo("Received pose: x={:.2f}, y={:.2f}, distance={:.2f}m".format(
                target_x, target_y, current_distance))
        
        # Check if the pose is too far away (likely wrong)
        if current_distance > self.max_distance_threshold:
            rospy.logwarn("Target too far ({:.2f}m > {:.2f}m), ignoring".format(
                current_distance, self.max_distance_threshold))
            return
        
        with self.lock:
            # Check if we should start moving, stop, or continue
            if current_distance <= self.distance_threshold:
                # Target is close enough - stop if we're moving
                if self.is_moving:
                    rospy.loginfo("Target close enough ({:.2f}m <= {:.2f}m), stopping robot".format(
                        current_distance, self.distance_threshold))
                    if hasattr(self, 'client'):
                        self.client.cancel_goal()
                    self.is_moving = False
            else:
                # Target is far - start moving if we're not already
                if not self.is_moving:
                    rospy.loginfo("Target far enough ({:.2f}m > {:.2f}m), moving to target".format(
                        current_distance, self.distance_threshold))
                    self.move_to_target(target_x, target_y, target_pose.orientation)
            
            self.previous_distance = current_distance
    
    def move_to_target(self, x, y, orientation, frame="base_footprint"):
        """
        Send a goal to move_base to navigate to the target position.
        """
        try:
            rospy.loginfo("Sending navigation goal to: x={:.2f}, y={:.2f}".format(x, y))
            
            # Build goal
            goal = MoveBaseGoal()
            goal.target_pose.header.frame_id = frame
            goal.target_pose.header.stamp = rospy.Time.now()
            goal.target_pose.pose.position.x = x
            goal.target_pose.pose.position.y = y
            goal.target_pose.pose.position.z = 0.0  # Explicitly set z to 0
            goal.target_pose.pose.orientation = orientation
            
            if hasattr(self, 'client') and self.client:
                self.client.send_goal(goal, done_cb=self.goal_done_callback, 
                                    feedback_cb=self.goal_feedback_callback)
                self.is_moving = True
                rospy.loginfo("Goal sent successfully")
                
                # Check goal status after a short delay
                rospy.Timer(rospy.Duration(1.0), self.check_goal_status, oneshot=True)
            else:
                rospy.logwarn("Move_base client not available, simulating movement")
                # Still set is_moving to True for testing the logic
                self.is_moving = True
            
        except Exception as e:
            rospy.logerr("Error sending goal: {}".format(e))
            self.is_moving = False
    
    def goal_done_callback(self, status, result):
        """
        Callback when goal is completed (success, failure, or cancelled).
        """
        rospy.loginfo("Goal completed with status: {}".format(status))
        if status == actionlib.GoalStatus.SUCCEEDED:
            rospy.loginfo("Navigation goal reached successfully!")
        elif status == actionlib.GoalStatus.ABORTED:
            rospy.logwarn("Navigation goal was aborted")
        elif status == actionlib.GoalStatus.PREEMPTED:
            rospy.loginfo("Navigation goal was cancelled")
        else:
            rospy.logwarn("Navigation goal finished with status: {}".format(status))
        
        with self.lock:
            self.is_moving = False
    
    def goal_feedback_callback(self, feedback):
        """
        Callback for goal feedback (current robot position during navigation).
        """
        current_pos = feedback.base_position.pose.position
        rospy.loginfo("Robot moving - current position: x={:.2f}, y={:.2f}".format(
            current_pos.x, current_pos.y))
    
    def check_goal_status(self):
        """
        Check the current status of the goal after sending it.
        """
        if hasattr(self, 'client') and self.client:
            state = self.client.get_state()
            rospy.loginfo("Goal status after 1 second: {}".format(state))
            if state == actionlib.GoalStatus.PENDING:
                rospy.loginfo("Goal is pending...")
            elif state == actionlib.GoalStatus.ACTIVE:
                rospy.loginfo("Goal is active - robot should be moving")
            elif state == actionlib.GoalStatus.ABORTED:
                rospy.logwarn("Goal was aborted immediately - check for obstacles or invalid goal")
            elif state == actionlib.GoalStatus.REJECTED:
                rospy.logerr("Goal was rejected - check navigation stack configuration")
    
    def set_distance_threshold(self, new_threshold):
        """
        Update the distance threshold for stopping.
        
        Args:
            new_threshold (float): New distance threshold in meters
        """
        with self.lock:
            self.distance_threshold = new_threshold
    
    def stop_robot(self):
        """
        Stop the robot by canceling current goal.
        """
        with self.lock:
            if self.is_moving:
                if hasattr(self, 'client'):
                    self.client.cancel_goal()
                self.is_moving = False
    
    
    def spin(self):
        """
        Keep the node running and processing callbacks.
        """
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
        
        # Create and start the adaptive navigator
        navigator = AdaptiveRobotNavigator()
        
        # Start listening for messages and processing navigation
        navigator.spin()
        
    except rospy.ROSInterruptException:
        rospy.loginfo("Navigation interrupted by user")
    except KeyboardInterrupt:
        rospy.loginfo("Navigation stopped by user")
    except Exception as e:
        rospy.logerr("Error in main: {}".format(e))
    
    # Legacy example using the original function
    # success = move_robot_to_coordinate(1.0, 0.0, yaw=0.0, cancel_after=5.0)
    # if success:
    #     print("Robot reached the target!")
    # else:
    #     print("Goal was cancelled or failed.")
