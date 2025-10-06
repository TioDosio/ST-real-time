#!/usr/bin/env python
import rospy
import actionlib
import math
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal

def yaw_to_quaternion(yaw):
    """
    Convert yaw angle to quaternion components.
    
    :param yaw: yaw angle in radians
    :return: tuple of (x, y, z, w) quaternion components
    """
    # For rotation around z-axis only (yaw)
    x = 0.0
    y = 0.0
    z = math.sin(yaw / 2.0)
    w = math.cos(yaw / 2.0)
    return x, y, z, w

def stop_robot(client):
    """
    Stop - cancels ALL goals on the move_base server.
    """
    if client.wait_for_server(rospy.Duration(0.5)):
        # Cancel all goals (including from other clients)
        client.cancel_all_goals()
        rospy.loginfo("All goals cancelled - emergency stop")
        return True
    else:
        rospy.logwarn("move_base action server not available")
        return False

def move_robot_to_coordinate(client, coordinates, yaw):
    frame = "base_footprint"  # or "map" if using a map frame

    try:
        # Initialize node if not already done
        if not rospy.get_node_uri():
            rospy.init_node('robot_navigation', anonymous=True)
            print("Initialized ROS node 'robot_navigation'")
        
        # Create action client
        rospy.loginfo("Waiting for move_base action server...")
        
        # Create goal
        goal = MoveBaseGoal()
        goal.target_pose.header.frame_id = frame
        goal.target_pose.header.stamp = rospy.Time.now()
        goal.target_pose.pose.position.x = coordinates[0]
        goal.target_pose.pose.position.y = coordinates[1]
        
        # Convert yaw to quaternion
        qx, qy, qz, qw = yaw_to_quaternion(yaw)
        goal.target_pose.pose.orientation.x = qx + math.pi
        goal.target_pose.pose.orientation.y = qy + math.pi
        goal.target_pose.pose.orientation.z = qz
        goal.target_pose.pose.orientation.w = qw
        
        # Send goal
        rospy.loginfo(f"Moving to: x={coordinates[0]:.2f}, y={coordinates[1]:.2f}, yaw={yaw:.2f} rad")
        
        client.send_goal(goal)
        
        # Return immediately without waiting
        rospy.loginfo("Goal sent successfully, not waiting for completion")
        return True
            
    except Exception as e:
        rospy.logerr(f"Error in move_robot_to_coordinate: {e}")
        return False
