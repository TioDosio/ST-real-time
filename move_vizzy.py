#!/usr/bin/env python
import rospy
import actionlib
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from tf.transformations import quaternion_from_euler

def mean_coordinates(coords, yaw):
        if not coords:
            print("No coordinates")
        # use the last 3 coordinates of the list and remove outliers
        last_coords = coords[-3:]
        x_mean = sum(c[0] for c in last_coords) / len(last_coords)
        y_mean = sum(c[1] for c in last_coords) / len(last_coords)

        return (x_mean, y_mean)

def move_robot_to_coordinate(coordinates, yaw):
    frame = "odom"
    timeout = 30
    x, y = mean_coordinates(coordinates)

    try:
        # Initialize node if not already done
        if not rospy.get_node_uri():
            rospy.init_node('robot_navigation', anonymous=True)
            print("Initialized ROS node 'robot_navigation'")
        
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
        rospy.loginfo(f"Moving to: x={x:.2f}, y={y:.2f}, yaw={yaw:.2f} rad")
        client.send_goal(goal)
        
        # Wait for result
        finished = client.wait_for_result(rospy.Duration(timeout))
        
        if not finished:
            rospy.logwarn(f"Goal timed out after {timeout} seconds")
            client.cancel_goal()
            return False
        
        state = client.get_state()
        if state == actionlib.GoalStatus.SUCCEEDED:
            rospy.loginfo("Successfully reached target!")
            return True
        else:
            rospy.logwarn(f"Failed to reach target. State: {state}")
            return False
            
    except Exception as e:
        rospy.logerr(f"Error in move_robot_to_coordinate: {e}")
        return False
