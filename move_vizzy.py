#!/usr/bin/env python
import rospy
import actionlib
import math
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal

def quaternion_from_euler(roll, pitch, yaw):
    """Convert Euler angles to quaternion (manual implementation to avoid tf import issues)"""
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)
    
    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy
    
    return [qx, qy, qz, qw]

def mean_coordinates(coords, yaw):
    if not coords:
        print("No coordinates provided")
        return (0.0, 0.0)
    
    print(f"DEBUG: mean_coordinates received coords type: {type(coords)}")
    print(f"DEBUG: coords length: {len(coords) if hasattr(coords, '__len__') else 'unknown'}")
    if coords and len(coords) > 0:
        print(f"DEBUG: first element type: {type(coords[0])}")
        print(f"DEBUG: first element: {coords[0]}")
    
    # Handle nested list structure - coords might be a list of trajectory lists
    # Flatten the coordinates if needed
    flattened_coords = []
    for item in coords:
        if isinstance(item, list) and len(item) > 0:
            if isinstance(item[0], list):
                # This is a trajectory (list of points)
                flattened_coords.extend(item)
            else:
                # This is a single point
                flattened_coords.append(item)
    
    if not flattened_coords:
        print("No valid coordinates found after flattening")
        return (0.0, 0.0)
    
    print(f"DEBUG: flattened_coords length: {len(flattened_coords)}")
    print(f"DEBUG: flattened_coords sample: {flattened_coords[:3] if len(flattened_coords) >= 3 else flattened_coords}")
    
    # Use the last 3 coordinates of the list and remove outliers
    last_coords = flattened_coords[-3:]
    
    # Ensure all coordinates have at least 2 elements (x, y)
    valid_coords = [c for c in last_coords if isinstance(c, list) and len(c) >= 2]
    
    if not valid_coords:
        print("No valid coordinates with x,y values")
        return (0.0, 0.0)
    
    print(f"DEBUG: valid_coords for averaging: {valid_coords}")
    
    x_mean = sum(c[0] for c in valid_coords) / len(valid_coords)
    y_mean = sum(c[1] for c in valid_coords) / len(valid_coords)
    
    print(f"DEBUG: calculated mean: x={x_mean:.3f}, y={y_mean:.3f}")

    return (x_mean, y_mean)

def move_robot_to_coordinate(coordinates, yaw):
    frame = "odom"
    timeout = 0.05
    x, y = mean_coordinates(coordinates, yaw)

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
