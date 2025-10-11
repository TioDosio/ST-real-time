import rospy
import actionlib
import math
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
        self.pose_subscriber = rospy.Subscriber("/goal_pose", PoseArray, self.pose_callback)
        rospy.loginfo("Waiting for pose messages...s")

    def pose_callback(self, msg):
        """
        Callback function for pose array messages.
        Assumes we want to track the first pose in the array.
        """
        if not msg.poses:
            rospy.logwarn("Received empty pose array!")
            return

        if msg.poses[0].position.z == 0.0:
            self.client.cancel_goal()
        else:
            self.move_to_target(msg.poses[0].position.x, msg.poses[0].position.y, msg.poses[0].orientation)
    
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
            goal.target_pose.pose.position.z = 0.0
            goal.target_pose.pose.orientation = orientation
            
            if hasattr(self, 'client') and self.client:
                self.client.send_goal(goal, done_cb=self.goal_done_callback, 
                                    feedback_cb=self.goal_feedback_callback)
                rospy.loginfo("Goal sent successfully")
                
                # Check goal status after a short delay
                rospy.Timer(rospy.Duration(1.0), self.check_goal_status, oneshot=True)
            else:
                rospy.logwarn("Move_base client not available, simulating movement")
                # Still set is_moving to True for testing the logic
            
        except Exception as e:
            rospy.logerr("Error sending goal: {}".format(e))
    
    def goal_done_callback(self, status):
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
    
    def stop_robot(self):
        if hasattr(self, 'client'):
            self.client.cancel_goal()
    
    
    def spin(self):
        rospy.spin()

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
