#!/usr/bin/env python
import rospy
import actionlib
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from tf.transformations import quaternion_from_euler

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
    # Example: Move to (1.0, 0.0), cancel after 5 seconds
    success = move_robot_to_coordinate(1.0, 0.0, yaw=0.0, cancel_after=5.0)

    if success:
        print("Robot reached the target!")
    else:
        print("Goal was cancelled or failed.")
