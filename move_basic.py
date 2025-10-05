#!/usr/bin/env python
import rospy
import actionlib
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from tf.transformations import quaternion_from_euler

def move_robot_to_coordinate(x, y, yaw=0.0, frame="odom", timeout=60.0):
    """
    Complete function to move robot to a coordinate.
    
    :param x: target x coordinate in meters
    :param y: target y coordinate in meters
    :param yaw: target orientation in radians (default: 0.0)
    :param frame: reference frame (default: "map")
    :param timeout: max time to wait for result in seconds (default: 60.0)
    :return: True if successful, False otherwise
    """
    try:
        # Initialize node if not already done
        if not rospy.get_node_uri():
            rospy.init_node('robot_navigation', anonymous=True)
        
        # Create action client
        client = actionlib.SimpleActionClient('move_base', MoveBaseAction)
        rospy.loginfo("Waiting for move_base action server...")
        
        if not client.wait_for_server(rospy.Duration(5.0)):
            rospy.logerr("move_base action server not available")
            return False
        
        # Create goal
        goal = MoveBaseGoal()
        goal.target_pose.header.frame_id = frame
        goal.target_pose.header.stamp = rospy.Time.now()
        goal.target_pose.pose.position.x = x
        goal.target_pose.pose.position.y = y
        
        # Convert yaw to quaternion
        q = quaternion_from_euler(0, 0, yaw)
        goal.target_pose.pose.orientation.x = q[0]
        goal.target_pose.pose.orientation.y = q[1]
        goal.target_pose.pose.orientation.z = q[2]
        goal.target_pose.pose.orientation.w = q[3]
        
        # Send goal
        rospy.loginfo("Moving to: x={:.2f}, y={:.2f}, yaw={:.2f} rad".format(x, y, yaw))
        client.send_goal(goal)
        
        # Wait for result
        finished = client.wait_for_result(rospy.Duration(timeout))
        
        if not finished:
            rospy.logwarn("Goal timed out after {} seconds".format(timeout))
            client.cancel_goal()
            return False
        
        state = client.get_state()
        if state == actionlib.GoalStatus.SUCCEEDED:
            rospy.loginfo("Successfully reached target!")
            return True
        else:
            rospy.logwarn("Failed to reach target. State: {}".format(state))
            return False
            
    except Exception as e:
        rospy.logerr("Error in move_robot_to_coordinate: {}".format(e))
        return False

# Example usage:
if __name__ == '__main__':
    # Your trajectory prediction code here
    # x_pred, y_pred, yaw_pred = your_prediction_function()
    
    # Then simply call:
    success = move_robot_to_coordinate(0.5, 0.5, 0)  # x=0.5m, y=0.5m, yaw=0 
    
    if success:
        print("Robot reached the target!")
    else:
        print("Failed to reach target")
